"""Ghép Top-K chunk thành context có đánh số + chuyển [n] thành citation đầy đủ."""

from __future__ import annotations

import re
import string
from collections import defaultdict

from rag.schemas import Chunk, RetrievedChunk

DEFAULT_TEMPLATE = "{short_ref}, {section}, trang {pages}"
_CITE = re.compile(r"\[(\d+(?:\s*[,;]\s*\d+)*)\]")


def format_pages(chunk: Chunk) -> str:
    return str(chunk.page) if chunk.end_page == chunk.page else f"{chunk.page}-{chunk.end_page}"


def citation_label(chunk: Chunk, template: str = DEFAULT_TEMPLATE) -> str:
    """Điền template citation từ metadata. Trường thiếu => bỏ trống rồi dọn dấu phẩy thừa."""
    fields: dict[str, object] = defaultdict(str)
    fields.update({k: v for k, v in chunk.metadata.items() if v is not None})
    fields.update({
        "source": chunk.source,
        "section": chunk.section,
        "page": chunk.page,
        "pages": format_pages(chunk),
        "short_ref": chunk.metadata.get("short_ref") or chunk.metadata.get("title") or chunk.source,
    })
    label = string.Formatter().vformat(template, (), fields)
    label = re.sub(r"(?:\s*,\s*){2,}", ", ", label)      # "A, , trang 3" -> "A, trang 3"
    return label.strip(" ,")


def build_context(results: list[RetrievedChunk], template: str = DEFAULT_TEMPLATE) -> tuple[str, list[str]]:
    """Trả về (context_text, labels) với labels[i] là citation của đoạn [i+1]."""
    labels = [citation_label(r.chunk, template) for r in results]
    blocks = [f"[{i}] Nguồn: {label}\n{r.chunk.text}" for i, (r, label) in enumerate(zip(results, labels), start=1)]
    return "\n\n---\n\n".join(blocks), labels


def cited_numbers(answer: str) -> list[int]:
    nums: list[int] = []
    for m in _CITE.finditer(answer):
        for n in re.split(r"\s*[,;]\s*", m.group(1)):
            if int(n) not in nums:
                nums.append(int(n))
    return nums


def render_citations(answer: str, labels: list[str]) -> str:
    """'... 91,2% [1][3]' -> '... 91,2% [Nguyen et al., 2024, 4.2 Results, trang 8][...]'."""

    def _replace(m: re.Match[str]) -> str:
        parts = []
        for n in re.split(r"\s*[,;]\s*", m.group(1)):
            i = int(n)
            parts.append(f"[{labels[i - 1]}]" if 1 <= i <= len(labels) else f"[{n}]")
        return "".join(parts)

    return _CITE.sub(_replace, answer)


_ENCODER = None


def count_tokens(text: str) -> int:
    """Ước lượng số token context (tiktoken cl100k). Tokenizer của từng LLM có thể lệch ±20%."""
    global _ENCODER
    try:
        if _ENCODER is None:
            import tiktoken
            _ENCODER = tiktoken.get_encoding("cl100k_base")
        return len(_ENCODER.encode(text))
    except Exception:  # không có tiktoken / không tải được bảng mã
        return max(1, len(text) // 4)
