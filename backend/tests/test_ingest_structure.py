"""Test các quy tắc xử lý cấu trúc tài liệu (mục lục, tiêu đề xuống dòng, References)."""
from rag.ingest.cleaner import clean_blocks
from rag.ingest.section_detector import heading_level, is_heading, is_toc_page
from rag.schemas import Document


def test_toc_line_is_not_heading():
    assert not is_heading("Chương 4: THU THẬP DỮ LIỆU (Sqoop + Flume) ........................ 16")
    assert not is_heading("5.3. So sánh Hive và Pig ..................................... 20")
    assert is_heading("Chương 4: THU THẬP DỮ LIỆU (Sqoop + Flume)")


def test_new_keyword_headings():
    assert is_heading("KẾT LUẬN VÀ KIẾN NGHỊ")
    assert is_heading("PHỤ LỤC: PHÂN CÔNG CÔNG VIỆC")
    assert is_heading("DANH MỤC HÌNH ẢNH")


def test_detect_toc_page():
    toc = ("MỤC LỤC\n\nLỜI CẢM ƠN ............ 2\n\nMỞ ĐẦU ............ 7\n\n"
           "Chương 1: GIỚI THIỆU ............ 8")
    figures = "Hình 1. Kiến trúc ........ 10 Hình 2. Phiên bản ........ 11 Hình 3. Giao diện ........ 12"
    body = "Apache Hive cho phép truy vấn SQL... Kết quả được lưu lại. Xem thêm Hình 1."
    assert is_toc_page(toc) and is_toc_page(figures)
    assert not is_toc_page(body)


def test_wrapped_heading_is_merged():
    text = clean_blocks(["Chương 2: KIẾN TRÚC HỆ THỐNG VÀ SƠ ĐỒ", "PIPELINE", "2.1. Kiến trúc tổng thể",
                         "Hệ thống gồm 4 tầng."])
    assert text.split("\n\n") == ["Chương 2: KIẾN TRÚC HỆ THỐNG VÀ SƠ ĐỒ PIPELINE", "2.1. Kiến trúc tổng thể",
                                  "Hệ thống gồm 4 tầng."]


def test_uppercase_body_line_not_merged_into_normal_heading():
    text = clean_blocks(["2.1. Kiến trúc tổng thể", "HDFS"])
    assert text.split("\n\n") == ["2.1. Kiến trúc tổng thể", "HDFS"]


def test_heading_level_and_chapter():
    assert heading_level("Chương 5: XỬ LÝ DỮ LIỆU") == 1
    assert heading_level("5.3. So sánh Hive và Pig") == 2
    assert heading_level("3 Experiments") == 1
    assert heading_level("MỞ ĐẦU") == 1
    text = "Chương 5: XỬ LÝ\n\n5.3. So sánh\n\nHive và Pig khác nhau."
    doc = Document("d.pdf", text, [(0, len(text), 1)], [(0, "Chương 5: XỬ LÝ"), (17, "5.3. So sánh")])
    assert doc.chapter_at(40) == "Chương 5: XỬ LÝ"
    assert doc.section_at(40) == "5.3. So sánh"


def test_remove_range_keeps_appendix():
    parts = ["Kết luận nhóm.", "TÀI LIỆU THAM KHẢO\n\n[1] Hadoop docs", "PHỤ LỤC: PHÂN CÔNG\n\nToàn 25%"]
    text = "\n\n".join(parts)
    p2 = len(parts[0]) + 2
    p3 = p2 + len(parts[1]) + 2
    doc = Document("d.pdf", text, [(0, len(parts[0]), 25), (p2, p3 - 2, 26), (p3, len(text), 27)],
                   [(p2, "TÀI LIỆU THAM KHẢO"), (p3, "PHỤ LỤC: PHÂN CÔNG")])
    doc.remove_range(p2, p3)
    assert doc.text == "Kết luận nhóm.\n\nPHỤ LỤC: PHÂN CÔNG\n\nToàn 25%"
    assert [p for _, _, p in doc.page_spans] == [25, 27]
    assert doc.section_spans == [(p2, "PHỤ LỤC: PHÂN CÔNG")]
    assert doc.page_at(doc.text.index("Toàn")) == 27


# ---------------- bài báo khoa học: References, phụ lục, tiêu đề giả ----------------
from rag.ingest.pdf_loader import drop_reference_section  # noqa: E402
from rag.ingest.section_detector import find_references_start, find_sections  # noqa: E402


def _doc(paras: list[str]) -> Document:
    text = "\n\n".join(paras)
    return Document("p.pdf", text, [(0, len(text), 1)], find_sections(text))


def test_bibliography_entry_and_table_header_are_not_headings():
    assert not is_heading("1. Al-Rfou, R., Choe, D., Constant, N., Guo, M., Jones, L.: Character-level language")
    assert not is_heading("12. Gu, J., Bradbury, J., Xiong, C., Li, V.O., Socher, R.: Non-autoregressive")
    assert not is_heading("Model")
    assert is_heading("A Appendix")
    assert is_heading("A.1 Implementation Details")
    assert is_heading("B. More Comparison")


def test_numbered_headings_must_follow_order():
    text = "\n\n".join(["3 Method", "Body text here.", "3.1 Loss", "2 The input images are batched together",
                        "Footnote-like line above.", "4 Experiments", "5 Conclusion"])
    assert [t for _, t in find_sections(text)] == ["3 Method", "3.1 Loss", "4 Experiments", "5 Conclusion"]


def test_drop_references_keeps_appendix():
    doc = _doc(["1 Introduction", "Detectors are fast. " * 30, "2 Conclusion", "We conclude.", "References",
                "1. Al-Rfou, R., Choe, D.: Character-level language modeling",
                "12. Gu, J., Bradbury, J.: Non-autoregressive translation", "A Appendix", "A.1 Losses", "Box loss."])
    assert find_references_start(doc.text) is not None
    assert drop_reference_section(doc)
    assert "Al-Rfou" not in doc.text and "Gu, J." not in doc.text
    assert doc.text.endswith("We conclude.\n\nA Appendix\n\nA.1 Losses\n\nBox loss.")


def test_drop_references_until_end_without_appendix():
    doc = _doc(["1 Introduction", "Detectors are fast. " * 30, "References", "[1] Bochkovskiy, A.: YOLOv4"])
    assert drop_reference_section(doc)
    assert "YOLOv4" not in doc.text and doc.text.startswith("1 Introduction")


def test_reference_word_inside_sentence_is_ignored():
    doc = _doc(["1 Introduction", "See [40] for references in the context of vision. " * 20])
    assert not drop_reference_section(doc)
