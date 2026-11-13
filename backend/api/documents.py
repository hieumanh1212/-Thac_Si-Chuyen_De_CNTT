"""Quản lý tài liệu qua API: upload / xóa PDF, sửa metadata, dựng lại index chạy nền.

    GET    /api/documents                  danh sách PDF + trạng thái index
    POST   /api/documents                  upload 1 PDF (multipart) + metadata tùy chọn
    PUT    /api/documents/{file}/metadata  sửa metadata (tiêu đề, tác giả, short_ref...)
    DELETE /api/documents/{file}           xóa PDF
    POST   /api/index/rebuild              dựng lại index cho mọi chiến lược (chạy nền)
    GET    /api/index/status               tiến độ dựng index
"""

from __future__ import annotations

import json
import logging
import re
import shutil
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import pymupdf
import yaml
from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from rag.chunking import STRATEGIES
from rag.config import resolve_path
from rag.indexing.build_index import META_FILE, build_strategy, index_dir
from rag.ingest import load_corpus
from rag.ingest.pdf_loader import load_all_sources, load_sources, nfc
from rag.pipeline import RAGPipeline

logger = logging.getLogger("api.documents")

MAX_UPLOAD_MB = 50
METADATA_FIELDS = ["title", "authors", "year", "venue", "document_type", "doi_url", "source_url", "course",
                   "short_ref", "number_symbol", "issue_date", "issuing_authority"]
_BAD_CHARS = re.compile(r'[<>:"/\\|?*\x00-\x1f]')


# ------------------------------ tiện ích file ------------------------------
def raw_dir(cfg: dict[str, Any]) -> Path:
    d = resolve_path(cfg["paths"]["raw_dir"])
    d.mkdir(parents=True, exist_ok=True)
    return d


def safe_pdf_name(filename: str) -> str:
    name = nfc(Path(filename or "").name).strip()
    name = _BAD_CHARS.sub("_", name)
    if not name.lower().endswith(".pdf") or len(name) <= 4:
        raise HTTPException(400, "Chỉ nhận file có đuôi .pdf")
    if len(name) > 180:
        raise HTTPException(400, "Tên file quá dài (tối đa 180 ký tự).")
    return name


def find_pdf(cfg: dict[str, Any], name: str) -> Path:
    """Tìm file trong data/raw theo tên (so khớp NFC); chặn đường dẫn kiểu ../"""
    wanted = nfc(Path(name).name)
    for p in raw_dir(cfg).glob("*.pdf"):
        if nfc(p.name) == wanted:
            return p
    raise HTTPException(404, f"Không tìm thấy tài liệu: {name}")


def clean_metadata(meta: dict[str, Any] | None) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key in METADATA_FIELDS:
        value = (meta or {}).get(key)
        if value is None:
            continue
        value = str(value).strip()
        if not value:
            continue
        out[key] = int(value) if key == "year" and value.isdigit() else value
    return out


def uploaded_sources_path(cfg: dict[str, Any]) -> Path:
    return resolve_path(cfg["paths"].get("uploaded_sources_file", "data/sources_uploaded.yaml"))


def write_uploaded_metadata(cfg: dict[str, Any], name: str, meta: dict[str, Any] | None) -> None:
    """Ghi/xóa metadata của 1 file trong sources_uploaded.yaml (meta=None => xóa)."""
    path = uploaded_sources_path(cfg)
    data = load_sources(path)
    if meta:
        data[name] = meta
    else:
        data.pop(name, None)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("# File do giao diện web tự sinh khi upload / sửa thông tin tài liệu.\n"
                "# Giá trị ở đây được ưu tiên hơn data/sources.yaml.\n")
        yaml.safe_dump({"documents": data}, f, allow_unicode=True, sort_keys=False)


def inspect_pdf(content: bytes) -> tuple[int, bool]:
    """Trả về (số trang, có chữ hay không). Lỗi => file không phải PDF hợp lệ."""
    with pymupdf.open(stream=content, filetype="pdf") as pdf:
        n = pdf.page_count
        has_text = any(pdf[i].get_text("text").strip() for i in range(min(n, 5)))
    return n, has_text


def index_meta(cfg: dict[str, Any]) -> dict[str, Any] | None:
    """meta.json của chiến lược đầu tiên đã có index (để biết index chứa những tài liệu nào)."""
    for s in STRATEGIES:
        p = index_dir(cfg, s) / META_FILE
        if p.exists():
            with open(p, encoding="utf-8") as f:
                return {**json.load(f), "_mtime": p.stat().st_mtime}
    return None


# ------------------------------ job dựng index chạy nền ------------------------------
class IndexJob:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self.state: dict[str, Any] = {"status": "idle", "progress": 0.0, "message": "", "error": None,
                                      "started_at": None, "finished_at": None}

    @property
    def running(self) -> bool:
        return self.state["status"] == "running"

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return dict(self.state)

    def _update(self, **kw: Any) -> None:
        with self._lock:
            self.state.update(kw)

    def start(self, pipe: RAGPipeline, query_lock: threading.Lock) -> bool:
        with self._lock:
            if self.state["status"] == "running":
                return False
            self.state = {"status": "running", "progress": 0.0, "message": "Chuẩn bị...", "error": None,
                          "started_at": datetime.now().isoformat(timespec="seconds"), "finished_at": None}
        self._thread = threading.Thread(target=self._run, args=(pipe, query_lock), daemon=True)
        self._thread.start()
        return True

    def _run(self, pipe: RAGPipeline, query_lock: threading.Lock) -> None:
        cfg = pipe.cfg
        strategies = cfg.get("chunking", {}).get("strategies", list(STRATEGIES))
        total = len(strategies) + 1
        t0 = time.perf_counter()
        try:
            # Nạp sẵn index cũ vào RAM để trong lúc dựng, người dùng vẫn hỏi đáp được bằng index cũ
            with query_lock:
                for s in pipe.available_strategies():
                    pipe.retriever(s)

            if not any(raw_dir(cfg).glob("*.pdf")):
                self._update(message="Không còn tài liệu - xóa index cũ")
                with query_lock:
                    for s in STRATEGIES:
                        shutil.rmtree(index_dir(cfg, s), ignore_errors=True)
                    pipe.reset()
                self._update(status="done", progress=1.0, message="Đã xóa index (không còn tài liệu nào).",
                             finished_at=datetime.now().isoformat(timespec="seconds"))
                return

            self._update(message="Đọc và làm sạch PDF...", progress=0.02)
            docs = load_corpus(cfg)
            for i, s in enumerate(strategies, start=1):
                self._update(message=f"Dựng index '{s}' ({i}/{len(strategies)})...", progress=i / total)
                build_strategy(s, docs, cfg, pipe.embedder)

            with query_lock:
                pipe.reset()          # lần hỏi tiếp theo sẽ nạp index mới
            self._update(status="done", progress=1.0,
                         message=f"Hoàn tất: {len(docs)} tài liệu, {len(strategies)} chiến lược, "
                                 f"{time.perf_counter() - t0:.0f} giây.",
                         finished_at=datetime.now().isoformat(timespec="seconds"))
        except Exception as exc:     # báo lỗi cho giao diện thay vì làm sập server
            logger.exception("Dựng index thất bại")
            self._update(status="error", error=f"{type(exc).__name__}: {exc}", message="Dựng index thất bại.",
                         finished_at=datetime.now().isoformat(timespec="seconds"))


# ------------------------------ router ------------------------------
def create_router(get_pipe: Callable[[], RAGPipeline], query_lock: threading.Lock, job: IndexJob) -> APIRouter:
    router = APIRouter(prefix="/api")

    def ensure_idle() -> None:
        if job.running:
            raise HTTPException(409, "Đang dựng index, vui lòng đợi xong rồi thao tác.")

    def start_rebuild() -> dict[str, Any]:
        job.start(get_pipe(), query_lock)
        return job.snapshot()

    @router.get("/documents")
    def list_documents() -> dict[str, Any]:
        cfg = get_pipe().cfg
        sources = load_all_sources(cfg["paths"])
        meta = index_meta(cfg)
        indexed = {nfc(d) for d in (meta or {}).get("documents", [])}
        index_time = (meta or {}).get("_mtime", 0)

        docs, changed = [], False
        for p in sorted(raw_dir(cfg).glob("*.pdf"), key=lambda x: x.name.lower()):
            name, stat = nfc(p.name), p.stat()
            docs.append({"file": name, "size_kb": round(stat.st_size / 1024, 1),
                         "uploaded_at": datetime.fromtimestamp(stat.st_mtime).isoformat(timespec="seconds"),
                         "metadata": sources.get(name, {}), "indexed": name in indexed})
            changed |= stat.st_mtime > index_time
        meta_files = [resolve_path(cfg["paths"]["sources_file"]), uploaded_sources_path(cfg)]
        changed |= any(f.exists() and f.stat().st_mtime > index_time for f in meta_files) and bool(meta)
        needs_rebuild = {d["file"] for d in docs} != indexed or (bool(docs) and changed)
        return {"documents": docs, "needs_rebuild": needs_rebuild, "job": job.snapshot(),
                "index_built_at": (meta or {}).get("built_at"), "max_upload_mb": MAX_UPLOAD_MB,
                "metadata_fields": METADATA_FIELDS}

    @router.post("/documents")
    async def upload_document(file: UploadFile = File(...), metadata: str = Form("{}"),
                              overwrite: bool = Form(False), rebuild: bool = Form(True)) -> dict[str, Any]:
        ensure_idle()
        cfg = get_pipe().cfg
        name = safe_pdf_name(file.filename or "")
        content = await file.read()
        if len(content) > MAX_UPLOAD_MB * 1024 * 1024:
            raise HTTPException(413, f"File lớn hơn {MAX_UPLOAD_MB} MB.")
        if not content.startswith(b"%PDF"):
            raise HTTPException(400, f"{name} không phải file PDF hợp lệ.")
        try:
            n_pages, has_text = inspect_pdf(content)
        except Exception as exc:
            raise HTTPException(400, f"Không đọc được PDF {name}: {exc}") from exc
        try:
            meta = clean_metadata(json.loads(metadata or "{}"))
        except json.JSONDecodeError as exc:
            raise HTTPException(400, "metadata phải là JSON hợp lệ.") from exc

        target = raw_dir(cfg) / name
        try:
            existing = find_pdf(cfg, name)
        except HTTPException:
            existing = None
        if existing and not overwrite:
            raise HTTPException(409, f"Đã có tài liệu tên '{name}'. Chọn ghi đè hoặc đổi tên file.")
        if existing and existing != target:
            existing.unlink()                    # tên cũ dạng NFD -> thay bằng tên NFC
        target.write_bytes(content)
        if meta:
            write_uploaded_metadata(cfg, name, meta)

        warning = None if has_text else ("PDF có vẻ là bản scan (không có lớp chữ) nên không trích xuất được "
                                         "nội dung. Cần OCR trước khi dùng.")
        return {"file": name, "pages": n_pages, "warning": warning,
                "job": start_rebuild() if rebuild else job.snapshot()}

    @router.put("/documents/{name}/metadata")
    def update_metadata(name: str, body: dict[str, Any]) -> dict[str, Any]:
        ensure_idle()
        cfg = get_pipe().cfg
        file = nfc(find_pdf(cfg, name).name)
        write_uploaded_metadata(cfg, file, clean_metadata(body) or None)
        return {"file": file, "metadata": load_all_sources(cfg["paths"]).get(file, {})}

    @router.delete("/documents/{name}")
    def delete_document(name: str, rebuild: bool = True) -> dict[str, Any]:
        ensure_idle()
        cfg = get_pipe().cfg
        path = find_pdf(cfg, name)
        file = nfc(path.name)
        path.unlink()
        write_uploaded_metadata(cfg, file, None)
        return {"deleted": file, "job": start_rebuild() if rebuild else job.snapshot()}

    @router.post("/index/rebuild")
    def rebuild_index() -> dict[str, Any]:
        if not job.start(get_pipe(), query_lock):
            raise HTTPException(409, "Đang dựng index.")
        return job.snapshot()

    @router.get("/index/status")
    def index_status() -> dict[str, Any]:
        return job.snapshot()

    return router
