"""Test upload / sửa metadata / xóa tài liệu qua API (dùng thư mục tạm, không dựng index thật)."""

import json

import pymupdf
import pytest

pytest.importorskip("fastapi")
pytest.importorskip("multipart")
from fastapi.testclient import TestClient  # noqa: E402

from api.main import create_app  # noqa: E402
from tests.test_api import FakePipe  # noqa: E402


def make_pdf(text: str = "Xin chào Hybrid RAG") -> bytes:
    doc = pymupdf.open()
    doc.new_page().insert_text((72, 72), text)
    data = doc.tobytes()
    doc.close()
    return data


@pytest.fixture
def client(tmp_path):
    pipe = FakePipe()
    pipe.cfg["paths"] = {"raw_dir": str(tmp_path / "raw"), "sources_file": str(tmp_path / "sources.yaml"),
                         "uploaded_sources_file": str(tmp_path / "uploaded.yaml"),
                         "index_dir": str(tmp_path / "indexes")}
    with TestClient(create_app(pipe)) as c:
        yield c, tmp_path


def upload(c, name="Bài báo A.pdf", content=None, **form):
    data = {"rebuild": "false", **{k: str(v).lower() if isinstance(v, bool) else v for k, v in form.items()}}
    return c.post("/api/documents", files={"file": (name, content or make_pdf(), "application/pdf")}, data=data)


def test_upload_list_update_delete(client):
    c, tmp = client
    meta = {"title": "Bài A", "year": "2024", "short_ref": "Nguyen, 2024", "unknown": "bỏ qua"}
    r = upload(c, metadata=json.dumps(meta, ensure_ascii=False))
    assert r.status_code == 200, r.text
    assert r.json()["pages"] == 1 and r.json()["warning"] is None
    assert (tmp / "raw" / "Bài báo A.pdf").exists()

    listing = c.get("/api/documents").json()
    doc = listing["documents"][0]
    assert doc["file"] == "Bài báo A.pdf" and doc["indexed"] is False
    assert doc["metadata"] == {"title": "Bài A", "year": 2024, "short_ref": "Nguyen, 2024"}
    assert listing["needs_rebuild"] is True

    r = c.put("/api/documents/Bài báo A.pdf/metadata", json={"title": "Bài A (sửa)", "short_ref": "Nguyen, 2024"})
    assert r.json()["metadata"]["title"] == "Bài A (sửa)"

    assert c.delete("/api/documents/Bài báo A.pdf", params={"rebuild": False}).status_code == 200
    assert c.get("/api/documents").json()["documents"] == []


def test_upload_rejects_bad_files(client):
    c, _ = client
    assert upload(c, name="notes.txt").status_code == 400
    assert upload(c, name="fake.pdf", content=b"hello").status_code == 400
    assert upload(c).status_code == 200
    assert upload(c).status_code == 409                       # trùng tên
    assert upload(c, overwrite=True).status_code == 200       # cho phép ghi đè


def test_upload_warns_on_scanned_pdf(client):
    c, _ = client
    doc = pymupdf.open()
    doc.new_page()                                            # trang trắng, không có chữ
    r = upload(c, name="scan.pdf", content=doc.tobytes())
    assert r.status_code == 200 and "scan" in r.json()["warning"]


def test_path_traversal_blocked(client):
    c, _ = client
    assert c.delete("/api/documents/..%2F..%2Fsecret.pdf").status_code == 404
