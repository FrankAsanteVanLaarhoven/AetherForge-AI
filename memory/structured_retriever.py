"""
memory/structured_retriever.py — v2.19 multi-view structured retrieval + reranking.

Retrieves over the structured dense index (memory/dense_index_v219_structured), built
from the same protected memory pool with the same baseline code-aware MiniLM encoder as
the v2.18 baseline. Holding the encoder fixed isolates the effect being tested: richer
record structure + multiple query views + deterministic reranking.

Query views (derived from the task text only, since retrieval happens once before
generation):
  - instruction : the raw task text
  - family      : "family: <inferred family> ..." emphasis
  - signature   : "signature: <expected callable> ..." emphasis

NOTE: the failure-mode and tool-state views described for v2.19 require re-retrieval
*inside* the agent trajectory (after an error/observation). The current agent retrieves
once, pre-generation, from task text alone, so those two views are out of scope for this
milestone and are documented as future work rather than silently faked.

Reranking is a deterministic composite over the merged candidates:
  dense similarity (max across views)
  + family match + signature/name match + cue overlap + multi-view agreement
  - shorter-repair preference (longer verified solutions penalised)

`mode="hybrid"` first gates candidates to the baseline MiniLM dense shortlist over the
original index (the proven baseline), then applies the same structured rerank.

Records returned keep their ORIGINAL injection fields (task, corrected_tool_call, ...),
so the prompt block is identical in form to the baseline; only selection/order differ.
"""

from pathlib import Path
from typing import Optional

import numpy as np

from memory.dense_index import load_dense_index
from memory.dense_retriever import _load_encoder, _encode_single
from memory.structured_common import (
    primary_function_name, task_signature_from_prompt, infer_family, extract_cues,
    tokenize, jaccard,
)

# Deterministic rerank weights (semantic similarity dominates; structure breaks ties).
W_FAMILY = 0.15
W_SIGNATURE = 0.10
W_CUE = 0.10
W_MULTIVIEW = 0.05
W_LENGTH = 0.05


class StructuredReranker:
    """Multi-view retrieval over the structured dense index with composite reranking."""

    def __init__(
        self,
        index_dir: Path,
        model_name: Optional[str] = None,
        rerank_top_n: int = 20,
        mode: str = "dense",
        shortlist_index: Optional[Path] = None,
    ):
        self.index_dir = Path(index_dir)
        self._model_name = model_name
        self.rerank_top_n = rerank_top_n
        self.mode = mode  # "dense" | "hybrid"
        self.shortlist_index = Path(shortlist_index) if shortlist_index else None
        self._state = None
        self._model = None
        self._records = None
        self._vectors = None
        self._shortlist_state = None
        # cache of per-record max solution length for length normalisation
        self._max_sol_len = 1

    def _ensure_loaded(self):
        if self._state is not None:
            return
        self._state = load_dense_index(self.index_dir)
        self._records = self._state["records"]
        self._vectors = self._state["vectors"]
        name = self._model_name or self._state["model_name"]
        self._model = _load_encoder(name)
        self._max_sol_len = max(
            (len(r.get("verified_solution", "") or "") for r in self._records), default=1
        ) or 1
        if self.mode == "hybrid" and self.shortlist_index is not None:
            from memory.store import load_index
            self._shortlist_state = load_index(self.shortlist_index)

    # ── query views ─────────────────────────────────────────────────────────
    def _build_views(self, task_text: str) -> dict[str, str]:
        func = primary_function_name("", task_text)
        family = infer_family(task_text, func)
        signature = task_signature_from_prompt(task_text) or func
        cues = extract_cues(task_text, "", func)
        return {
            "instruction": task_text,
            "family": f"family: {family}\ncues: {', '.join(cues)}",
            "signature": f"signature: {signature}\ntask: {task_text}",
            "_meta": {"family": family, "func": func, "signature": signature, "cues": cues},
        }

    def _shortlist_ids(self, task_text: str) -> Optional[set]:
        """Baseline MiniLM dense shortlist of record ids over the original index."""
        if self._shortlist_state is None:
            return None
        from memory.embed import embed_query
        from memory.store import search
        qv = embed_query(task_text, self._shortlist_state["query_texts"],
                         self._shortlist_state.get("vocab"))
        hits = search(self._shortlist_state, qv, self.rerank_top_n)
        return {h.get("id") for h in hits if h.get("id")}

    # ── retrieval ───────────────────────────────────────────────────────────
    def retrieve(self, task_text: str, top_k: int = 4) -> list[dict]:
        self._ensure_loaded()
        views = self._build_views(task_text)
        meta = views.pop("_meta")

        view_names = ["instruction", "family", "signature"]
        view_vecs = self._model.encode(
            [views[v] for v in view_names],
            normalize_embeddings=True, show_progress_bar=False,
        ).astype(np.float32)

        # Per-view cosine over the structured index; collect candidate -> (max dense, #views).
        cand_dense: dict[int, float] = {}
        cand_views: dict[int, int] = {}
        n = len(self._records)
        for vi in range(len(view_names)):
            sims = (self._vectors @ view_vecs[vi].T).reshape(-1)
            k = min(self.rerank_top_n, n)
            top = np.argsort(-sims)[:k]
            for idx in top:
                idx = int(idx)
                s = float(sims[idx])
                if idx not in cand_dense or s > cand_dense[idx]:
                    cand_dense[idx] = s
                cand_views[idx] = cand_views.get(idx, 0) + 1

        shortlist = self._shortlist_ids(task_text) if self.mode == "hybrid" else None

        q_family = meta["family"]
        q_func = (meta["func"] or "").lower()
        q_name_tokens = set(t for t in q_func.split("_") if len(t) >= 3)
        q_cues = meta["cues"]

        scored = []
        for idx, dense in cand_dense.items():
            rec = self._records[idx]
            if shortlist is not None and rec.get("id") not in shortlist:
                continue
            family_match = 1.0 if rec.get("task_family") == q_family else 0.0
            cand_sig_tokens = set(t for t in tokenize(rec.get("task_signature", "")) if len(t) >= 3)
            sig_match = 1.0 if (q_name_tokens & cand_sig_tokens) else 0.0
            cue_overlap = jaccard(q_cues, rec.get("retrieval_cues", []) or [])
            multiview = (cand_views[idx] - 1) / max(len(view_names) - 1, 1)
            sol_len = len(rec.get("verified_solution", "") or "")
            length_pen = sol_len / self._max_sol_len
            composite = (
                dense
                + W_FAMILY * family_match
                + W_SIGNATURE * sig_match
                + W_CUE * cue_overlap
                + W_MULTIVIEW * multiview
                - W_LENGTH * length_pen
            )
            out = dict(rec)
            out["score"] = float(composite)
            out["dense_score"] = float(dense)
            scored.append(out)

        # Deterministic order: composite desc, then id for stable tie-break.
        scored.sort(key=lambda r: (-r["score"], r.get("id", "")))
        return scored[:top_k]


def retrieve_structured(
    task_text: str,
    index_dir: Path,
    top_k: int = 4,
    model_name: Optional[str] = None,
    rerank_top_n: int = 20,
    mode: str = "dense",
    shortlist_index: Optional[Path] = None,
) -> list[dict]:
    """One-shot convenience wrapper."""
    r = StructuredReranker(index_dir, model_name=model_name, rerank_top_n=rerank_top_n,
                           mode=mode, shortlist_index=shortlist_index)
    return r.retrieve(task_text, top_k=top_k)
