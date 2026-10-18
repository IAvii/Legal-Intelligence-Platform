"""
experiments/src/windows.py
Phase 2: Contract windowing, embedding, and top-k retrieval.

Key design choices (per EXPERIMENT_PLAN §5.3):
  - Window size: 3000 chars, overlap: 300 chars (configurable in config.yaml)
  - Embedding model: same as the app's fastembed model (MiniLM-L6-v2)
  - Embeddings are cached to disk keyed by (contract_id, window_idx)
  - Retrieval recall (for gold-present contracts) is logged separately

Public API:
    from src.windows import retrieve_windows, retrieval_recall
"""
from __future__ import annotations

import hashlib
import json
import pickle
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

EXPERIMENTS_DIR = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(EXPERIMENTS_DIR / "src"))
from config_loader import CFG, EXPERIMENTS_DIR  # noqa: E402

_ret_cfg = CFG.get("retrieval", {})
WINDOW_CHARS = _ret_cfg.get("window_chars", 3000)
OVERLAP_CHARS = _ret_cfg.get("window_overlap_chars", 300)
TOP_K = _ret_cfg.get("top_k", 3)
EMB_MODEL = _ret_cfg.get("embedding_model", "sentence-transformers/all-MiniLM-L6-v2")
CACHE_DIR = EXPERIMENTS_DIR / CFG.get("cache", {}).get("embedding_dir", "cache/embeddings")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# Embedding (lazy-loaded singleton)
# ─────────────────────────────────────────────────────────────────────────────

_embedder = None


def _get_embedder():
    global _embedder
    if _embedder is None:
        try:
            from fastembed import TextEmbedding  # type: ignore
            _embedder = TextEmbedding(model_name=EMB_MODEL)
        except ImportError:
            raise ImportError("fastembed not installed. Run: pip install fastembed")
    return _embedder


def _embed(texts: list[str]) -> np.ndarray:
    """Returns (n, dim) float32 array."""
    emb = _get_embedder()
    vecs = list(emb.embed(texts))
    return np.array(vecs, dtype=np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Windowing
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Window:
    window_id: str   # f"{contract_id}__w{idx}"
    idx: int
    text: str
    start_char: int
    end_char: int


def make_windows(contract_id: str, text: str) -> list[Window]:
    """Split contract text into overlapping character windows."""
    windows: list[Window] = []
    step = WINDOW_CHARS - OVERLAP_CHARS
    i = 0
    idx = 0
    while i < len(text):
        chunk = text[i: i + WINDOW_CHARS]
        windows.append(Window(
            window_id=f"{contract_id}__w{idx}",
            idx=idx,
            text=chunk,
            start_char=i,
            end_char=i + len(chunk),
        ))
        i += step
        idx += 1
    return windows


# ─────────────────────────────────────────────────────────────────────────────
# Cached embeddings
# ─────────────────────────────────────────────────────────────────────────────

def _cache_key(contract_id: str) -> Path:
    safe = hashlib.md5(contract_id.encode()).hexdigest()
    return CACHE_DIR / f"{safe}.pkl"


def _load_cached_windows(contract_id: str) -> tuple[list[Window], np.ndarray] | None:
    path = _cache_key(contract_id)
    if path.exists():
        with open(path, "rb") as f:
            return pickle.load(f)
    return None


def _save_cached_windows(contract_id: str, windows: list[Window], vecs: np.ndarray) -> None:
    path = _cache_key(contract_id)
    with open(path, "wb") as f:
        pickle.dump((windows, vecs), f)


def get_windows_and_embeddings(
    contract_id: str, text: str
) -> tuple[list[Window], np.ndarray]:
    """Return (windows, embeddings) with disk cache."""
    cached = _load_cached_windows(contract_id)
    if cached is not None:
        return cached
    windows = make_windows(contract_id, text)
    vecs = _embed([w.text for w in windows])
    _save_cached_windows(contract_id, windows, vecs)
    return windows, vecs


# ─────────────────────────────────────────────────────────────────────────────
# Retrieval
# ─────────────────────────────────────────────────────────────────────────────

def _cosine(q: np.ndarray, M: np.ndarray) -> np.ndarray:
    """Cosine similarity of query vector q (1-D) against matrix M (n, d)."""
    q_norm = q / (np.linalg.norm(q) + 1e-9)
    M_norm = M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-9)
    return M_norm @ q_norm


def retrieve_windows(
    contract_id: str,
    contract_text: str,
    query: str,
    top_k: int | None = None,
) -> list[Window]:
    """
    Retrieve top-k windows most relevant to `query`.
    Returns windows in descending similarity order.
    """
    if top_k is None:
        top_k = TOP_K
    windows, vecs = get_windows_and_embeddings(contract_id, contract_text)
    q_vec = _embed([query])[0]
    scores = _cosine(q_vec, vecs)
    idxs = np.argsort(scores)[::-1][:top_k]
    return [windows[i] for i in idxs]


# ─────────────────────────────────────────────────────────────────────────────
# Retrieval recall diagnostic (H4)
# ─────────────────────────────────────────────────────────────────────────────

def _normalize_text(t: str) -> str:
    """Strip and collapse whitespace for fuzzy substring matching."""
    import re
    return re.sub(r"\s+", " ", t).strip().lower()


def retrieval_recall(
    retrieved_windows: list[Window],
    gold_spans: list[str],
) -> bool:
    """
    Returns True if at least one gold span overlaps (as substring) with
    at least one retrieved window, after whitespace normalization.
    """
    retrieved_text = " ".join(_normalize_text(w.text) for w in retrieved_windows)
    for span in gold_spans:
        # Use a 10-word prefix of the span as a probe (full span may be noisy)
        probe = _normalize_text(span)
        if len(probe) > 60:
            probe = probe[:60]
        if probe and probe in retrieved_text:
            return True
    return False
