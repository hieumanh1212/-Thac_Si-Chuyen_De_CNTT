"""Tokenizer cho BM25 (tiếng Việt + tiếng Anh).

- simple: chữ thường + tách theo ký tự chữ/số. Tiếng Việt bị tách theo âm tiết
  ("học phí" -> ["học", "phí"]) nhưng BM25 vẫn hoạt động khá tốt.
- pyvi: tách từ ghép tiếng Việt ("học phí" -> "học_phí"), chính xác hơn cho
  tài liệu tiếng Việt. Cần `pip install pyvi`.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Callable

_WORD = re.compile(r"\w+", re.UNICODE)

STOPWORDS_EN = {
    "a", "an", "the", "and", "or", "of", "to", "in", "on", "for", "with", "by", "at", "from",
    "is", "are", "was", "were", "be", "been", "being", "this", "that", "these", "those", "it",
    "its", "as", "which", "what", "how", "does", "do", "did", "can", "we", "our", "they",
    "their", "there", "than", "then", "also", "into", "such", "has", "have", "had",
}
STOPWORDS_VI = {
    "là", "của", "và", "các", "những", "được", "cho", "trong", "với", "một", "này", "đó",
    "thì", "mà", "để", "từ", "theo", "về", "như", "khi", "đã", "sẽ", "bị", "nào", "gì",
    "ở", "ra", "vào", "lại", "cũng", "nhiều", "rất", "hay", "hoặc",
}
STOPWORDS = STOPWORDS_EN | STOPWORDS_VI

Tokenizer = Callable[[str], list[str]]


def get_tokenizer(name: str = "simple", remove_stopwords: bool = True) -> Tokenizer:
    if name == "pyvi":
        from pyvi import ViTokenizer

        def segment(text: str) -> str:
            return ViTokenizer.tokenize(text)
    elif name == "simple":
        def segment(text: str) -> str:
            return text
    else:
        raise ValueError(f"Tokenizer không hợp lệ: {name}")

    def tokenize(text: str) -> list[str]:
        text = unicodedata.normalize("NFC", text).lower()
        tokens = _WORD.findall(segment(text))
        if remove_stopwords:
            tokens = [t for t in tokens if t not in STOPWORDS]
        return tokens

    return tokenize
