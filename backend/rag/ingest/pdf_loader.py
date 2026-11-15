"""Đọc PDF THEO TỪNG TRANG bằng PyMuPDF và dựng Document giữ metadata trang/mục."""

from __future__ import annotations

import json
import logging
import re
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any

import pymupdf
import yaml

from rag.config import resolve_path
from rag.ingest.cleaner import clean_blocks, normalize_unicode
from rag.ingest.section_detector import (find_references_start, find_sections, is_appendix_heading,
                                         is_toc_page)
from rag.schemas import Document

logger = logging.getLogger(__name__)

# (text, y0, y1) với y đã chuẩn hóa về [0, 1] theo chiều cao trang
Block = tuple[str, float, float]


def _extract_page_blocks(page: pymupdf.Page) -> list[Block]:
    height = page.rect.height or 1.0
    blocks: list[Block] = []
    # "blocks" của PyMuPDF ~ từng đoạn văn; giữ thứ tự gốc của PDF (đúng với bài 2 cột)
    for b in page.get_text("blocks"):
        x0, y0, x1, y1, text, _block_no, block_type = b[:7]
        if block_type != 0 or not text.strip():   # bỏ khối ảnh
            continue
        blocks.append((text, y0 / height, y1 / height))
    return blocks


def _hf_key(text: str) -> str:
    return re.sub(r"\d+", "#", " ".join(text.lower().split()))


def remove_headers_footers(
    pages: list[list[Block]], min_ratio: float = 0.5, margin: float = 0.1
) -> list[list[Block]]:
    """Bỏ các khối nằm ở lề trên/dưới và lặp lại trên nhiều trang (header/footer)."""
    n = len(pages)
    if n < 3:
        return pages

    def in_margin(b: Block) -> bool:
        return b[2] <= margin or b[1] >= 1 - margin

    counts: Counter[str] = Counter()
    for blocks in pages:
        counts.update({_hf_key(b[0]) for b in blocks if in_margin(b)})
    threshold = max(2, min_ratio * n)
    repeated = {k for k, c in counts.items() if k and c >= threshold}
    return [[b for b in blocks if not (in_margin(b) and _hf_key(b[0]) in repeated)] for blocks in pages]


# Phần đánh số đứng trước tên mục trong văn bản: "4.2. ", "A.1 ", "B. ", "IV. "
_HEADING_NUMBER = re.compile(r"^(?:\d{1,2}(?:\.\d{1,2}){0,3}|[A-H](?:\.\d{1,2}){0,2}|[IVX]{1,4})?\.?\s*$")


def _find_heading_line(text: str, body: str, start: int, end: int) -> tuple[int, str] | None:
    """Tìm đoạn (dòng riêng) là tiêu đề có tên `body` trong text[start:end].

    Trả về (offset, tiêu đề đầy đủ kèm số mục như trong văn bản). Chấp nhận tiêu đề bị
    xuống dòng (đoạn chỉ chứa phần đầu của tên mục, tối thiểu 20 ký tự). Ưu tiên dòng có
    số mục (tên mục không số có thể chỉ là nhãn trong hình vẽ)."""
    low_body = body.lower()
    candidates: list[tuple[int, str, bool]] = []
    pos = start
    for para in text[start:end].split("\n\n"):
        line = para.strip()
        if line and "\n" not in line and len(line) <= len(body) + 12:
            low = line.lower()
            i = len(low) - len(low_body) if low.rstrip(" .:").endswith(low_body) else -1
            if i != -1:
                i = low.rstrip(" .:").rfind(low_body)
            if i == -1 and len(low) >= 20:                      # tiêu đề bị cắt xuống dòng
                j = next((k for k in range(min(8, len(low))) if low_body.startswith(low[k:])), -1)
                i = j if j != -1 and len(low) - j >= 20 else -1
            if i != -1 and _HEADING_NUMBER.match(line[:i]):
                prefix = line[:i].strip()
                candidates.append((pos + para.find(line), f"{prefix} {body}" if prefix else body, bool(prefix)))
        pos += len(para) + 2
    numbered = [c for c in candidates if c[2]]
    if numbered or candidates:
        off, title, _ = (numbered or candidates)[0]
        return off, title
    # Dự phòng: tiêu đề bị dính vào cuối đoạn văn trước, vd "... new generation. 2.2. Reversible Architectures"
    m = re.compile(r"(?<=\s)(\d{1,2}(?:\.\d{1,2}){0,3}\.?)\s+" + re.escape(body) + r"(?=\n\n)",
                   re.IGNORECASE).search(text, start, end)
    return (m.start(), f"{m.group(1)} {body}") if m else None


def _toc_sections(doc: pymupdf.Document, text: str, page_spans: list[tuple[int, int, int]]) -> list[tuple[int, str]]:
    """Lấy mục từ bookmark/TOC của PDF và định vị offset trong text.

    - Bookmark của nhiều bài LaTeX mất số mục (". Introduction") => lấy lại số mục từ văn bản.
    - Tìm tiêu đề trên đúng trang của bookmark, không thấy thì tìm tiếp phía sau mục trước
      (bookmark phụ lục đôi khi ghi sai số trang). Vẫn không thấy thì bỏ qua bookmark đó.
    """
    span_by_page = {p: (s, e) for s, e, p in page_spans}
    sections: list[tuple[int, str]] = []
    last = 0
    for _level, title, page_no in doc.get_toc(simple=True):
        body = " ".join(normalize_unicode(title).split()).lstrip(".:-–  ").strip()
        if not body:
            continue
        found = None
        if page_no in span_by_page:
            s, e = span_by_page[page_no]
            if e > last:
                found = _find_heading_line(text, body, max(s, last), e)
        if found is None:
            found = _find_heading_line(text, body, last, len(text))
        if found is None:
            logger.debug("Không định vị được bookmark %r (trang %s)", body, page_no)
            continue
        sections.append(found)
        last = found[0] + 1
    return sections


def drop_reference_section(document: Document) -> bool:
    """Xóa mục References: từ dòng tiêu đề 'References' tới tiêu đề mục kế tiếp (thường là
    Phụ lục) hoặc tới hết tài liệu. Phụ lục phía sau được giữ lại."""
    start = find_references_start(document.text)
    if start is None:
        return False
    end = next((off for off, _ in document.section_spans if off > start), None)
    if end is None:      # không có mục nào phía sau => tìm tiêu đề phụ lục ngay trong văn bản
        end = next((off for off, title in find_sections(document.text[start:])
                    if off > 0 and is_appendix_heading(title)), None)
        end = start + end if end is not None else len(document.text)
    document.remove_range(start, end)
    return True


def load_pdf(
    path: str | Path,
    meta: dict[str, Any] | None = None,
    remove_headers_footers_: bool = True,
    header_footer_min_ratio: float = 0.5,
    drop_references: bool = True,
    drop_toc_pages: bool = True,
) -> Document:
    path = Path(path)
    # Mở bằng bytes để tránh lỗi đường dẫn có dấu tiếng Việt trên Windows
    pdf = pymupdf.open(stream=path.read_bytes(), filetype="pdf")
    try:
        pages = [_extract_page_blocks(page) for page in pdf]
        if remove_headers_footers_:
            pages = remove_headers_footers(pages, header_footer_min_ratio)

        parts: list[str] = []
        page_spans: list[tuple[int, int, int]] = []
        skipped_toc: list[int] = []
        pos = 0
        for page_no, blocks in enumerate(pages, start=1):
            page_text = clean_blocks([b[0] for b in blocks])
            if not page_text:
                continue                         # trang trống / chỉ có ảnh
            if drop_toc_pages and is_toc_page(page_text):
                skipped_toc.append(page_no)      # mục lục / danh mục hình / bảng
                continue
            if parts:
                pos += 2                         # ký tự "\n\n" nối giữa các trang
            parts.append(page_text)
            page_spans.append((pos, pos + len(page_text), page_no))
            pos += len(page_text)
        text = "\n\n".join(parts)

        toc = _toc_sections(pdf, text, page_spans) if len(pdf.get_toc()) >= 3 else []
        sections = toc or find_sections(text)

        doc_meta = {"title": path.stem, **(meta or {})}
        doc_meta.setdefault("short_ref", doc_meta["title"])
        doc_meta["n_pages"] = pdf.page_count
        doc_meta["section_source"] = "toc" if toc else "heuristic"
        doc_meta["skipped_toc_pages"] = skipped_toc
    finally:
        pdf.close()

    document = Document(source=nfc(path.name), text=text, page_spans=page_spans,
                        section_spans=sections, meta=doc_meta)

    if drop_references and not drop_reference_section(document):
        logger.info("%s: không tìm thấy mục References.", path.name)

    if not document.text:
        logger.warning("%s: không trích xuất được chữ (có thể là PDF scan, cần OCR).", path.name)
    return document


def nfc(name: str) -> str:
    """Tên file tiếng Việt trên đĩa có thể ở dạng NFD (dấu tách rời) trong khi gõ tay là
    NFC => chuẩn hóa về NFC để so khớp đúng."""
    return unicodedata.normalize("NFC", name)


def load_sources(path: str | Path) -> dict[str, dict[str, Any]]:
    path = resolve_path(path)
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return {nfc(str(k)): v or {} for k, v in (data.get("documents") or {}).items()}


def load_all_sources(paths_cfg: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Gộp metadata soạn tay (sources.yaml) và metadata nhập qua giao diện (ưu tiên hơn)."""
    merged = load_sources(paths_cfg["sources_file"])
    uploaded = paths_cfg.get("uploaded_sources_file")
    if uploaded:
        for name, meta in load_sources(uploaded).items():
            merged[name] = {**merged.get(name, {}), **meta}
    return merged


def load_corpus(cfg: dict[str, Any], save: bool = True) -> list[Document]:
    """Nạp toàn bộ PDF trong data/raw kèm metadata từ data/sources.yaml."""
    paths_cfg, ingest_cfg = cfg["paths"], cfg.get("ingest", {})
    raw_dir = resolve_path(paths_cfg["raw_dir"])
    pdfs = sorted(raw_dir.glob("*.pdf"))
    if not pdfs:
        raise FileNotFoundError(f"Không có file PDF nào trong {raw_dir}")

    sources = load_all_sources(paths_cfg)
    docs: list[Document] = []
    for pdf_path in pdfs:
        name = nfc(pdf_path.name)
        if name not in sources:
            logger.warning("%s chưa khai báo trong sources.yaml - dùng tên file làm tiêu đề.", name)
        doc = load_pdf(
            pdf_path,
            meta=sources.get(name, {}),
            remove_headers_footers_=ingest_cfg.get("remove_headers_footers", True),
            header_footer_min_ratio=ingest_cfg.get("header_footer_min_ratio", 0.5),
            drop_references=ingest_cfg.get("drop_references", True),
            drop_toc_pages=ingest_cfg.get("drop_toc_pages", True),
        )
        logger.info("%s: %d trang, %d ký tự, %d mục (%s), bỏ trang mục lục: %s", doc.source,
                    len(doc.page_spans), len(doc.text), len(doc.section_spans),
                    doc.meta["section_source"], doc.meta["skipped_toc_pages"] or "không")
        docs.append(doc)

    if save:
        out_dir = resolve_path(paths_cfg["processed_dir"])
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(out_dir / "documents.jsonl", "w", encoding="utf-8") as f:
            for doc in docs:
                f.write(json.dumps(doc.to_dict(), ensure_ascii=False) + "\n")
    return docs
