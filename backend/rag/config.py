"""Đọc cấu hình YAML + biến môi trường (.env)."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

# rag/config.py -> parents[1] là thư mục gốc dự án
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "config.yaml"

_ENV_OVERRIDES = {
    "provider": "LLM_PROVIDER",
    "model": "LLM_MODEL",
    "base_url": "LLM_BASE_URL",
}


def setup_logging(level: int = logging.INFO) -> None:
    """Log gọn cho các script CLI; ẩn log HTTP chi tiết của thư viện tải mô hình."""
    logging.basicConfig(level=level, format="%(levelname)s | %(message)s")
    for noisy in ("httpx", "httpx2", "huggingface_hub", "sentence_transformers", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Nạp config.yaml; biến LLM_* trong .env (nếu có) ghi đè phần `llm`.

    Thứ tự chọn file config: tham số `path` -> biến môi trường RAG_CONFIG -> configs/config.yaml.
    """
    load_dotenv(PROJECT_ROOT / ".env")
    path = path or os.getenv("RAG_CONFIG")
    cfg_path = Path(path) if path else DEFAULT_CONFIG
    with open(cfg_path, encoding="utf-8") as f:
        cfg: dict[str, Any] = yaml.safe_load(f) or {}

    llm = cfg.setdefault("llm", {})
    for key, env_name in _ENV_OVERRIDES.items():
        value = os.getenv(env_name)
        if value:
            llm[key] = value
    return cfg


def resolve_path(p: str | Path) -> Path:
    """Đường dẫn tương đối trong config được tính từ thư mục gốc dự án."""
    p = Path(p)
    return p if p.is_absolute() else PROJECT_ROOT / p
