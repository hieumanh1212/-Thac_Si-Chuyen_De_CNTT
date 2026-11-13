"""Làm sạch text trích xuất từ PDF (yêu cầu 3.1 của đề).

Xử lý:
- Chuẩn hóa Unicode NFC (quan trọng với tiếng Việt: dấu tổ hợp vs dựng sẵn).
- Ký tự ghép (ligature ﬁ, ﬂ...), soft hyphen, khoảng trắng không ngắt.
- Từ bị ngắt dòng bằng gạch nối: "infor-\\nmation" -> "information".
- Khoảng trắng dư thừa, dòng trống dư thừa.
- Dòng nhiễu: số trang đứng riêng, "Page 3 of 10", dấu arXiv bên lề.
- Nối các dòng bị ngắt cứng trong cùng một đoạn thành một dòng.
"""

from __future__ import annotations

import re
import unicodedata

from rag.ingest.section_detector import is_heading

_CHAR_MAP = {
    "ﬀ": "ff", "ﬁ": "fi", "ﬂ": "fl", "ﬃ": "ffi", "ﬄ": "ffl",
    "­": "",    # soft hyphen
    "​": "",    # zero-width space
    "﻿": "",    # BOM
    " ": " ",   # non-breaking space
    "\t": " ",
}
_CHAR_TRANS = str.maketrans(_CHAR_MAP)

_HYPHEN_BREAK = re.compile(r"(\w)-[ \t]*\n[ \t]*(\w)")
_MULTI_SPACE = re.compile(r"[ ]{2,}")
_NOISE_LINES = [
    re.compile(r"^\d{1,4}$"),                                          # số trang
    re.compile(r"^(?:page|trang)\s+\d+(?:\s*(?:of|/)\s*\d+)?$", re.IGNORECASE),
    re.compile(r"^arXiv:\d{4}\.\d{4,5}", re.IGNORECASE),               # dấu arXiv bên lề
    re.compile(r"^[^\w]$"),                                            # 1 ký tự lẻ
]


def normalize_unicode(text: str) -> str:
    return unicodedata.normalize("NFC", text).translate(_CHAR_TRANS)


def fix_hyphenation(text: str) -> str:
    """Nối từ bị ngắt dòng. Chỉ nối khi phần sau viết thường để không phá
    các cụm có gạch nối thật như 'BERT-\\nLarge'."""

    def _join(m: re.Match[str]) -> str:
        left, right = m.group(1), m.group(2)
        return left + right if right.islower() else f"{left}-{right}"

    return _HYPHEN_BREAK.sub(_join, text)


def is_noise_line(line: str) -> bool:
    return any(p.match(line) for p in _NOISE_LINES)


def _collapse_spaces(line: str) -> str:
    return _MULTI_SPACE.sub(" ", line).strip()


def _is_heading_continuation(prev: str, line: str) -> bool:
    """Tiêu đề in hoa quá dài bị xuống dòng:
    'Chương 2: KIẾN TRÚC HỆ THỐNG VÀ SƠ ĐỒ' + 'PIPELINE' => nối lại thành một tiêu đề."""
    if not is_heading(prev) or is_heading(line):
        return False
    prev_words = [w for w in prev.split() if any(ch.isalpha() for ch in w)]
    return (
        bool(prev_words) and prev_words[-1].isupper()
        and line.isupper() and len(line.split()) <= 6
        and not line.endswith((".", ":", ";", ","))
    )


def clean_blocks(blocks: list[str]) -> str:
    """Làm sạch danh sách khối văn bản của một trang (mỗi khối ~ một đoạn).

    Trả về text của trang, các đoạn cách nhau đúng một dòng trống ("\\n\\n");
    tiêu đề mục luôn đứng thành đoạn riêng để section_detector nhận diện.
    """
    paragraphs: list[str] = []
    for raw in blocks:
        text = fix_hyphenation(normalize_unicode(raw))
        lines = [_collapse_spaces(line) for line in text.split("\n")]
        lines = [line for line in lines if line and not is_noise_line(line)]
        if not lines:
            continue
        # Dòng đầu của khối là tiêu đề => tách thành đoạn riêng
        if len(lines) > 1 and is_heading(lines[0]):
            paragraphs.append(lines[0])
            lines = lines[1:]
        paragraphs.append(" ".join(lines))

    merged: list[str] = []
    for para in paragraphs:
        if merged and _is_heading_continuation(merged[-1], para):
            merged[-1] = f"{merged[-1]} {para}"
        else:
            merged.append(para)
    return "\n\n".join(merged)


def clean_page_text(raw: str) -> str:
    """Làm sạch text thô của một trang (các đoạn phân tách bằng dòng trống)."""
    blocks = re.split(r"\n[ \t]*\n", raw)
    return clean_blocks(blocks)
