"""Nhận diện tiêu đề mục/chương/điều để gán metadata `section` và `chapter` cho chunk.

Hai nguồn:
1. Mục lục (bookmark/TOC) nhúng trong PDF - chính xác nhất nếu có (xem pdf_loader).
2. Heuristic trên từng dòng (module này) - dùng khi PDF không có TOC.

Đây là phần nên tinh chỉnh theo từng loại tài liệu của đề tài (bài báo, giáo trình,
văn bản hành chính).
"""

from __future__ import annotations

import re

# Tiêu đề phổ biến (so sánh không phân biệt hoa thường, đã bỏ số thứ tự ở đầu).
KEYWORD_HEADINGS: set[str] = {
    # Bài báo khoa học (tiếng Anh)
    "abstract", "introduction", "related work", "related works", "background",
    # (không có "model", "method", "data": đó thường là tiêu đề cột của bảng kết quả)
    "preliminaries", "methods", "methodology", "approach", "proposed method",
    "proposed approach", "model architecture", "architecture", "dataset",
    "datasets", "experiments", "experiment", "experimental setup",
    "experimental results", "experiments and results", "evaluation", "results",
    "results and discussion", "discussion", "analysis", "ablation study",
    "ablation studies", "limitations", "future work", "conclusion", "conclusions",
    "conclusion and future work", "conclusions and future work", "references",
    "bibliography", "acknowledgments", "acknowledgements", "appendix",
    "contents", "table of contents", "list of figures", "list of tables",
    # Tiếng Việt (báo cáo, luận văn, giáo trình)
    "tóm tắt", "giới thiệu", "lời mở đầu", "lời nói đầu", "mở đầu", "đặt vấn đề", "tổng quan",
    "cơ sở lý thuyết", "các nghiên cứu liên quan", "phương pháp", "phương pháp đề xuất",
    "phương pháp nghiên cứu", "dữ liệu", "thực nghiệm", "kết quả", "kết quả thực nghiệm",
    "thảo luận", "kết luận", "kết luận và kiến nghị", "kết luận và hướng phát triển",
    "tài liệu tham khảo", "lời cảm ơn", "phụ lục", "mục lục", "danh mục hình ảnh",
    "danh mục hình vẽ", "danh mục bảng biểu", "danh mục bảng", "danh mục từ viết tắt",
}

REFERENCE_HEADINGS: set[str] = {"references", "reference", "bibliography", "tài liệu tham khảo"}

# "1 Introduction", "2.1. Related Work", "IV. RESULTS", "A. Appendix", "A.1 Implementation Details"
_NUM_PREFIX = re.compile(r"^(?:\d{1,2}(?:\.\d{1,2}){0,3}\.?|[IVXLC]{1,6}\.|[A-H](?:\.\d{1,2}){1,2}\.?|[A-H]\.)\s+(\S.*)$")
# "A Appendix" (đánh chữ không có dấu chấm, kiểu LaTeX)
_LETTER_APPENDIX = re.compile(r"^[A-H]\s+appendix\b", re.IGNORECASE)
# Một mục trong danh sách tài liệu tham khảo: "12. Gu, J., Bradbury, J., ..." hoặc "[3] Bochkovskiy, A., ..."
_BIB_ENTRY = re.compile(r"^(?:\[\d{1,3}\]|\d{1,3}\.)\s+[^\s,]+,\s+(?:[A-Z]\.|[A-Z][a-z]+\s[A-Z]\.)")
# "Chương 2", "Điều 5.", "Mục II", "Chapter 3", "Phụ lục A"
# (không dùng "Section"/"Part" vì dễ trùng câu văn như "Section 3 describes ...")
_STRUCT = re.compile(
    r"^(?:chương|phần|mục|điều|chapter|appendix|phụ lục)\s+(?:\d+|[IVXLC]+|[A-Z])(?=[\s.:)]|$)",
    re.IGNORECASE,
)
# "PHỤ LỤC: PHÂN CÔNG CÔNG VIỆC", "Appendix: Proofs"
_APPENDIX = re.compile(r"^(?:phụ lục|appendix)\s*[:.\-–]\s*\S", re.IGNORECASE)
# Chú thích hình/bảng không phải tiêu đề mục
_CAPTION = re.compile(r"^(?:fig(?:ure)?|table|tab|bảng|hình|algorithm|eq)\.?\s*\d", re.IGNORECASE)
# Dòng mục lục: "Chương 4: THU THẬP DỮ LIỆU ........ 16" (dấu chấm dẫn + số trang)
DOT_LEADER = re.compile(r"(?:\.\s?){4,}\s*\d{0,4}\s*$|…{2,}\s*\d{0,4}\s*$")
DOT_LEADER_ANY = re.compile(r"(?:\.\s?){4,}\s*\d{1,4}|…{2,}\s*\d{1,4}")
# Chương cấp 1: "Chương 2", "Chapter 3", "Phần II"
_CHAPTER = re.compile(r"^(?:chương|chapter|phần|part)\s+(?:\d+|[IVXLC]+)\b", re.IGNORECASE)


def heading_body(line: str) -> str:
    """Bỏ phần đánh số ở đầu: '2.1 Related Work' -> 'Related Work'."""
    s = line.strip()
    m = _NUM_PREFIX.match(s)
    return (m.group(1) if m else s).strip()


def is_heading(line: str) -> bool:
    s = line.strip()
    if not s or len(s) > 90 or _CAPTION.match(s) or DOT_LEADER.search(s) or _BIB_ENTRY.match(s):
        return False
    if _LETTER_APPENDIX.match(s):
        return True

    body = heading_body(s)
    if body.lower().rstrip(":. ") in KEYWORD_HEADINGS:
        return True

    if _STRUCT.match(s) or _APPENDIX.match(s):
        return len(s.split()) <= 15 and not s.endswith((",", ";"))

    if _NUM_PREFIX.match(s):
        words = body.split()
        if not 1 <= len(words) <= 12:
            return False
        if body.endswith((".", ",", ";")):          # tiêu đề hiếm khi kết thúc bằng dấu câu
            return False
        if not (body[0].isalpha() and body[0].isupper()):
            return False
        digits = sum(ch.isdigit() for ch in body)
        return digits <= 0.3 * len(body)            # tránh dòng số liệu trong bảng
    return False


def heading_level(title: str) -> int:
    """Cấp tiêu đề: 1 = chương / mục lớn ("Chương 2", "3 Method", "MỞ ĐẦU"),
    2 = "2.1 ...", 3 = "2.1.3 ...". Dùng để suy ra metadata `chapter`."""
    s = title.strip()
    if _CHAPTER.match(s) or _APPENDIX.match(s):
        return 1
    m = re.match(r"^(\d{1,2}(?:\.\d{1,2})*)\.?\s", s)
    if m:
        return m.group(1).count(".") + 1
    return 1   # tiêu đề từ khóa không đánh số (MỞ ĐẦU, Abstract, References...)


_TOP_NUMBER = re.compile(r"^(\d{1,2})(?:[.\s]|$)")


def is_appendix_heading(title: str) -> bool:
    s = title.strip()
    return bool(_APPENDIX.match(s) or _LETTER_APPENDIX.match(s) or re.match(r"^[A-H](?:\.\d{1,2})*\.?\s", s)
                or heading_body(s).lower().rstrip(":. ") in {"appendix", "phụ lục"}
                or re.match(r"^(?:appendix|phụ lục)\b", s, re.IGNORECASE))


def find_sections(text: str) -> list[tuple[int, str]]:
    """Tìm tiêu đề trong text đã làm sạch (mỗi tiêu đề là một đoạn riêng).

    Tiêu đề đánh số phải đi đúng thứ tự: số mục cấp 1 chỉ được giữ nguyên hoặc tăng 1
    (3 -> 3.1 -> 4). Nhờ vậy loại được chú thích chân trang ("2 The input images...")
    và các dòng đánh số lẻ khác nằm giữa văn bản. Gặp Phụ lục thì đếm lại từ đầu.
    """
    sections: list[tuple[int, str]] = []
    current: int | None = None          # số mục cấp 1 gần nhất
    pos = 0
    for para in text.split("\n\n"):
        stripped = para.strip()
        if stripped and "\n" not in stripped and is_heading(stripped):
            m = _TOP_NUMBER.match(stripped)
            if m:
                n = int(m.group(1))
                if current is not None and not current <= n <= current + 1:
                    pos += len(para) + 2
                    continue
                current = n
            elif is_appendix_heading(stripped):
                current = None
            sections.append((pos + para.find(stripped), stripped))
        pos += len(para) + 2
    return sections


_REFERENCES_PARA = re.compile(
    r"(?:^|\n\n)[ \t]*(?:\d{1,2}\.?\s+|[IVX]{1,4}\.\s+)?(?:references|reference|bibliography|tài liệu tham khảo)"
    r"[ \t:.]*(?=\n\n|$)", re.IGNORECASE)


def find_references_start(text: str, min_ratio: float = 0.3) -> int | None:
    """Vị trí dòng tiêu đề 'References' (đứng riêng một đoạn) cuối cùng, nằm sau `min_ratio` văn bản."""
    starts = [m.start() + (2 if m.group(0).startswith("\n\n") else 0) for m in _REFERENCES_PARA.finditer(text)]
    starts = [s for s in starts if s >= min_ratio * len(text)]
    return starts[-1] if starts else None


def is_toc_page(page_text: str, min_entries: int = 3) -> bool:
    """Trang mục lục / danh mục hình / danh mục bảng: có nhiều dòng dạng
    'tiêu đề ........ số trang'. Các trang này chỉ gây nhiễu cho tìm kiếm."""
    return len(DOT_LEADER_ANY.findall(page_text)) >= min_entries
