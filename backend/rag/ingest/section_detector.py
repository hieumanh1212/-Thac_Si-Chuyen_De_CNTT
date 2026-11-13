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
    "preliminaries", "method", "methods", "methodology", "approach", "proposed method",
    "proposed approach", "model", "model architecture", "architecture", "dataset",
    "datasets", "data", "experiments", "experiment", "experimental setup",
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

# "1 Introduction", "2.1. Related Work", "IV. RESULTS", "A. Appendix"
_NUM_PREFIX = re.compile(r"^(?:\d{1,2}(?:\.\d{1,2}){0,3}\.?|[IVXLC]{1,6}\.|[A-H]\.)\s+(\S.*)$")
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
    if not s or len(s) > 90 or _CAPTION.match(s) or DOT_LEADER.search(s):
        return False

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


def find_sections(text: str) -> list[tuple[int, str]]:
    """Tìm tiêu đề trong text đã làm sạch (mỗi tiêu đề là một đoạn riêng)."""
    sections: list[tuple[int, str]] = []
    pos = 0
    for para in text.split("\n\n"):
        stripped = para.strip()
        if stripped and "\n" not in stripped and is_heading(stripped):
            sections.append((pos + para.find(stripped), stripped))
        pos += len(para) + 2
    return sections


def is_toc_page(page_text: str, min_entries: int = 3) -> bool:
    """Trang mục lục / danh mục hình / danh mục bảng: có nhiều dòng dạng
    'tiêu đề ........ số trang'. Các trang này chỉ gây nhiễu cho tìm kiếm."""
    return len(DOT_LEADER_ANY.findall(page_text)) >= min_entries
