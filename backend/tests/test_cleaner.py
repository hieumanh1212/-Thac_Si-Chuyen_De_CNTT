import unicodedata

from rag.ingest.cleaner import clean_blocks, clean_page_text, fix_hyphenation
from rag.ingest.section_detector import find_sections, is_heading


def test_fix_hyphenation_joins_broken_words():
    assert fix_hyphenation("infor-\nmation retrieval") == "information retrieval"


def test_fix_hyphenation_keeps_real_compound():
    assert fix_hyphenation("BERT-\nLarge model") == "BERT-Large model"


def test_clean_joins_lines_and_collapses_spaces():
    raw = "This   is a\nbroken    line.\n\n\n\nNew   paragraph here."
    assert clean_page_text(raw) == "This is a broken line.\n\nNew paragraph here."


def test_clean_removes_noise_and_ligatures():
    raw = "The ﬁrst result is good.\n\n12\n\nPage 3 of 10"
    assert clean_page_text(raw) == "The first result is good."


def test_clean_normalizes_vietnamese_to_nfc():
    decomposed = unicodedata.normalize("NFD", "Học phí")
    assert clean_page_text(decomposed) == unicodedata.normalize("NFC", "Học phí")


def test_heading_split_from_block():
    text = clean_blocks(["1 Introduction\nDeep learning has achieved\nstrong results."])
    assert text == "1 Introduction\n\nDeep learning has achieved strong results."
    assert find_sections(text) == [(0, "1 Introduction")]


def test_is_heading():
    assert is_heading("2.1 Related Work")
    assert is_heading("ABSTRACT")
    assert is_heading("Chương 3 Mạng nơ-ron")
    assert not is_heading("Table 2: BLEU scores")
    assert is_heading("Điều 5. Điều kiện tốt nghiệp")
    assert not is_heading("Section 3 describes the proposed method")
    assert not is_heading("Phần mềm được cài đặt trên máy chủ")
    assert not is_heading("3 layers are stacked on top of each other.")
    assert not is_heading("We propose a new method for this task.")
