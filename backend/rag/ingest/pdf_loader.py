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
from rag.ingest.section_detector import REFERENCE_HEADINGS, find_sections, heading_body, is_toc_page
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


def _toc_sections(doc: pymupdf.Document, text: str, page_spans: list[tuple[int, int, int]]) -> list[tuple[int, str]]:
    """Lấy mục từ bookmark/TOC của PDF và định vị offset trong text."""
    span_by_page = {p: (s, e) for s, e, p in page_spans}
    sections: list[tuple[int, str]] = []
    for _level, title, page_no in doc.get_toc(simple=True):
        title = " ".join(normalize_unicode(title).split())
        if not title or page_no not in span_by_page:
            continue
        s, e = span_by_page[page_no]
        idx = text.lower().find(title.lower(), s, e)
        sections.append((idx if idx != -1 else s, title))
    return sorted(sections)


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

    if drop_references:
        # Chỉ xóa riêng mục References (tới tiêu đề kế tiếp), giữ lại Phụ lục phía sau
        for i, (offset, title) in enumerate(document.section_spans):
            if heading_body(title).lower().rstrip(":. ") in REFERENCE_HEADINGS and offset > 0.3 * len(text):
                nxt = document.section_spans[i + 1][0] if i + 1 < len(document.section_spans) else len(document.text)
                document.remove_range(offset, nxt)
                break

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
