"""
memory/dense_retriever.py — cosine similarity search over dense vectors.

Requires a pre-built dense index from memory/dense_index.py.
Falls back gracefully if the index is not present.
"""

from pathlib import Path
from typing import Optional

import numpy as np

from memory.dense_index import load_dense_index


class DenseRetriever:
    """Stateful retriever over a pre-built dense vector index."""

    def __init__(self, index_dir: Path, model_name: Optional[str] = None):
        self.index_dir = Path(index_dir)
        self._state = None
        self._model = None
        self._model_name = model_name  # override; None → use stored model_name

    def _ensure_loaded(self):
        if self._state is None:
            self._state = load_dense_index(self.index_dir)
        if self._model is None:
            name = self._model_name or self._state["model_name"]
            self._model = _load_encoder(name)

    def retrieve(self, task_text: str, top_k: int = 4) -> list[dict]:
        """Return top-k records by dense cosine similarity."""
        self._ensure_loaded()
        query_vec = _encode_single(self._model, task_text)
        hits = _cosine_search(self._state, query_vec, top_k)
        return hits

    @property
    def n_records(self) -> int:
        self._ensure_loaded()
        return len(self._state["records"])


# ── Helpers ───────────────────────────────────────────────────────────────────

def _load_encoder(model_name: str):
    """Load a SentenceTransformer. Raises ImportError if not installed."""
    try:
        from sentence_transformers import SentenceTransformer  # type: ignore
    except ImportError:
        raise ImportError(
            "sentence-transformers is required for dense retrieval. "
            "Install with: pip install sentence-transformers"
        )
    import torch  # type: ignore
    device = "cuda" if torch.cuda.is_available() else "cpu"
    local = Path(model_name)
    target = str(local.resolve()) if local.exists() else model_name
    return SentenceTransformer(target, device=device)


def _encode_single(model, text: str) -> np.ndarray:
    """Encode a single query string. Returns shape (1, D) float32, L2-normalised."""
    vec = model.encode([text], normalize_embeddings=True, show_progress_bar=False)
    return vec.astype(np.float32)


def _cosine_search(state: dict, query_vec: np.ndarray, top_k: int) -> list[dict]:
    """Cosine similarity search (vectors assumed L2-normalised → inner product)."""
    vectors = state["vectors"]  # (N, D)
    records = state["records"]

    scores = (vectors @ query_vec.T).squeeze()  # (N,)
    if scores.ndim == 0:
        scores = scores.reshape(1)

    k = min(top_k, len(records))
    idx = np.argpartition(scores, -k)[-k:]
    idx = idx[np.argsort(scores[idx])[::-1]]

    results = []
    for i in idx:
        rec = dict(records[i])
        rec["score"] = float(scores[i])
        results.append(rec)
    return results


def retrieve_dense(
    task_text: str,
    index_dir: Path,
    top_k: int = 4,
    model_name: Optional[str] = None,
) -> list[dict]:
    """Convenience function: one-shot dense retrieval without a persistent object."""
    r = DenseRetriever(index_dir, model_name=model_name)
    return r.retrieve(task_text, top_k=top_k)
