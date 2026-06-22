"""
memory/hybrid_retriever.py — TF-IDF candidate selection + dense reranking.

Strategy:
  1. Use TF-IDF cosine similarity to select top-N candidates from the full index.
     This is fast and avoids embedding the entire corpus at query time.
  2. Use a dense model to rerank those N candidates by semantic similarity.
  3. Return the final top-k records.

Rationale: TF-IDF recall is high even when precision is low.
Dense reranking applies algorithm-level similarity to the shortlist,
filtering out false positives from vocabulary overlap.
"""

from pathlib import Path
from typing import Optional

import numpy as np

from memory.embed import embed_query
from memory.store import load_index, search
from memory.dense_retriever import _load_encoder, _encode_single, _cosine_search


class HybridRetriever:
    """Two-stage retriever: TF-IDF shortlist → dense rerank."""

    def __init__(
        self,
        tfidf_index_dir: Path,
        dense_index_dir: Path,
        model_name: Optional[str] = None,
        rerank_top_n: int = 20,
    ):
        self.tfidf_dir = Path(tfidf_index_dir)
        self.dense_dir = Path(dense_index_dir)
        self._model_name = model_name
        self.rerank_top_n = rerank_top_n
        self._tfidf_state = None
        self._dense_model = None
        self._dense_records: Optional[list[dict]] = None
        self._dense_id_to_vec: Optional[dict] = None

    def _ensure_loaded(self):
        if self._tfidf_state is None:
            self._tfidf_state = load_index(self.tfidf_dir)

        if self._dense_model is None:
            from memory.dense_index import load_dense_index
            dense_state = load_dense_index(self.dense_dir)
            name = self._model_name or dense_state["model_name"]
            self._dense_model = _load_encoder(name)
            # Build id → vector lookup
            self._dense_records = dense_state["records"]
            self._dense_id_to_vec = {
                rec["id"]: dense_state["vectors"][i]
                for i, rec in enumerate(dense_state["records"])
            }

    def retrieve(self, task_text: str, top_k: int = 4) -> list[dict]:
        """Return top-k records via TF-IDF shortlist + dense rerank."""
        self._ensure_loaded()

        # Stage 1: TF-IDF shortlist
        tfidf_qv = embed_query(
            task_text,
            self._tfidf_state["query_texts"],
            self._tfidf_state.get("vocab"),
        )
        candidates = search(self._tfidf_state, tfidf_qv, self.rerank_top_n)
        candidates = [c for c in candidates if c.get("verified", False)]

        if not candidates:
            return []

        # Stage 2: dense rerank over candidates only
        query_vec = _encode_single(self._dense_model, task_text)

        reranked = []
        for cand in candidates:
            cid = cand.get("id")
            if cid and cid in self._dense_id_to_vec:
                dense_vec = self._dense_id_to_vec[cid].reshape(1, -1)
                score = float((dense_vec @ query_vec.T).squeeze())
                rec = dict(cand)
                rec["score"] = score
                rec["tfidf_score"] = cand.get("score", 0.0)
                reranked.append(rec)
            else:
                # No dense vector for this record; keep TF-IDF score as fallback
                reranked.append(cand)

        reranked.sort(key=lambda r: r.get("score", 0.0), reverse=True)
        return reranked[:top_k]


def retrieve_hybrid(
    task_text: str,
    tfidf_index_dir: Path,
    dense_index_dir: Path,
    top_k: int = 4,
    rerank_top_n: int = 20,
    model_name: Optional[str] = None,
) -> list[dict]:
    """Convenience function: one-shot hybrid retrieval."""
    r = HybridRetriever(
        tfidf_index_dir=tfidf_index_dir,
        dense_index_dir=dense_index_dir,
        model_name=model_name,
        rerank_top_n=rerank_top_n,
    )
    return r.retrieve(task_text, top_k=top_k)
