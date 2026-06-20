"""
memory/store.py — vector index and metadata storage.

Index backend (in order of preference):
  1. FAISS IndexFlatIP — inner product on L2-normalised vectors = cosine similarity
  2. NumPy fallback   — vectors.npy + cosine similarity at retrieval time

Metadata:
  SQLite database (metadata.sqlite) with one row per memory record.
  A parallel JSONL file (records.jsonl) is kept for human inspection and auditing.

File layout under index_dir/:
  faiss.index   — FAISS flat index  (or vectors.npy for the numpy fallback)
  vocab.json    — TF-IDF token→index (empty dict for ST embedder)
  query_texts.json — stored query texts (needed for TF-IDF query embedding)
  metadata.sqlite — SQLite DB with full record fields
  records.jsonl — human-readable copy (NOT committed)
"""

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np

INDEX_DIR = Path("memory/index")


# ── Helpers ───────────────────────────────────────────────────────────────────

def _sqlite_conn(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    conn.execute("""
        CREATE TABLE IF NOT EXISTS memories (
            id            TEXT PRIMARY KEY,
            task          TEXT,
            category      TEXT,
            failure_type  TEXT,
            query_text    TEXT,
            corrected_tool_call TEXT,
            observation   TEXT,
            final_answer  TEXT,
            source        TEXT,
            verified      INTEGER,
            created_at    TEXT,
            content_hash  TEXT,
            sensitivity   TEXT,
            vec_index     INTEGER
        )
    """)
    conn.commit()
    return conn


# ── Build ─────────────────────────────────────────────────────────────────────

def save_index(
    records: list[dict],
    vectors: np.ndarray,
    vocab: dict,
    index_dir: Path = INDEX_DIR,
) -> None:
    """Persist records + vectors to index_dir."""
    index_dir.mkdir(parents=True, exist_ok=True)

    # Assign vec_index to each record (row in vectors array)
    for i, rec in enumerate(records):
        rec["vec_index"] = i

    # ── FAISS or numpy ────────────────────────────────────────────────────
    try:
        import faiss  # type: ignore
        dim = vectors.shape[1]
        idx = faiss.IndexFlatIP(dim)
        idx.add(vectors)
        faiss.write_index(idx, str(index_dir / "faiss.index"))
        (index_dir / "backend.txt").write_text("faiss")
        print(f"[memory/store] Saved FAISS index: {idx.ntotal} vectors  dim={dim}")
    except ImportError:
        np.save(str(index_dir / "vectors.npy"), vectors)
        (index_dir / "backend.txt").write_text("numpy")
        print(
            f"[memory/store] FAISS not available — saved numpy fallback "
            f"({len(vectors)} vectors)"
        )

    # ── TF-IDF vocab + query texts ────────────────────────────────────────
    (index_dir / "vocab.json").write_text(json.dumps(vocab, ensure_ascii=False))
    query_texts = [r.get("query_text", "") for r in records]
    (index_dir / "query_texts.json").write_text(
        json.dumps(query_texts, ensure_ascii=False)
    )

    # ── SQLite ────────────────────────────────────────────────────────────
    db_path = index_dir / "metadata.sqlite"
    conn = _sqlite_conn(db_path)
    conn.execute("DELETE FROM memories")
    for rec in records:
        conn.execute(
            """INSERT OR REPLACE INTO memories VALUES
               (:id, :task, :category, :failure_type, :query_text,
                :corrected_tool_call, :observation, :final_answer,
                :source, :verified, :created_at, :content_hash,
                :sensitivity, :vec_index)""",
            {
                "id":                  rec.get("id", ""),
                "task":                rec.get("task", ""),
                "category":            rec.get("category", ""),
                "failure_type":        rec.get("failure_type", ""),
                "query_text":          rec.get("query_text", ""),
                "corrected_tool_call": rec.get("corrected_tool_call", ""),
                "observation":         rec.get("observation", ""),
                "final_answer":        rec.get("final_answer", ""),
                "source":              rec.get("source", ""),
                "verified":            int(rec.get("verified", False)),
                "created_at":          rec.get("created_at", ""),
                "content_hash":        rec.get("content_hash", ""),
                "sensitivity":         rec.get("sensitivity", "internal"),
                "vec_index":           rec.get("vec_index", -1),
            },
        )
    conn.commit()
    conn.close()

    # ── JSONL human copy (not for training) ───────────────────────────────
    jsonl_path = index_dir / "records.jsonl"
    with open(jsonl_path, "w") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"[memory/store] Saved {len(records)} records to {index_dir}")


# ── Load ──────────────────────────────────────────────────────────────────────

def load_index(index_dir: Path = INDEX_DIR) -> dict:
    """Load index state for retrieval.  Raises FileNotFoundError if missing."""
    backend_file = index_dir / "backend.txt"
    if not backend_file.exists():
        raise FileNotFoundError(
            f"No memory index found at {index_dir}. "
            "Run: python scripts/build_vector_memory.py"
        )
    backend = backend_file.read_text().strip()

    state: dict = {"backend": backend, "index_dir": index_dir}

    if backend == "faiss":
        import faiss  # type: ignore
        state["faiss_index"] = faiss.read_index(str(index_dir / "faiss.index"))
    else:
        state["vectors"] = np.load(str(index_dir / "vectors.npy"))

    state["vocab"] = json.loads((index_dir / "vocab.json").read_text())
    state["query_texts"] = json.loads((index_dir / "query_texts.json").read_text())

    db_path = index_dir / "metadata.sqlite"
    conn = _sqlite_conn(db_path)
    rows = conn.execute("SELECT * FROM memories ORDER BY vec_index").fetchall()
    conn.close()
    state["records"] = [dict(r) for r in rows]

    return state


# ── Search ────────────────────────────────────────────────────────────────────

def search(state: dict, query_vec: np.ndarray, top_k: int) -> list[dict]:
    """Return top_k records most similar to query_vec (cosine similarity)."""
    k = min(top_k, len(state["records"]))
    if k == 0:
        return []

    backend = state["backend"]

    if backend == "faiss":
        idx = state["faiss_index"]
        scores, indices = idx.search(query_vec, k)
        hits = []
        records = state["records"]
        for score, vec_idx in zip(scores[0], indices[0]):
            if vec_idx < 0:
                continue
            for rec in records:
                if rec.get("vec_index") == int(vec_idx):
                    hits.append({**rec, "score": float(score)})
                    break
        return hits
    else:
        # NumPy cosine similarity (vectors already L2-normalised)
        vectors = state["vectors"]
        sims = (vectors @ query_vec.T).squeeze()
        if sims.ndim == 0:
            sims = sims.reshape(1)
        top_indices = np.argsort(-sims)[:k]
        records = state["records"]
        return [
            {**records[i], "score": float(sims[i])}
            for i in top_indices
            if i < len(records)
        ]


# ── Append single record ──────────────────────────────────────────────────────

def append_record(
    record: dict,
    vector: np.ndarray,
    index_dir: Path = INDEX_DIR,
) -> None:
    """Add a single verified record to the index (for write-back).

    Appends to SQLite and JSONL; rebuilds the FAISS/numpy file in-place.
    This is intentionally simple (full rebuild) since write-back is rare.
    """
    state = load_index(index_dir)
    existing_records = state["records"]
    existing_vectors = (
        np.array([state["faiss_index"].reconstruct(i)
                  for i in range(state["faiss_index"].ntotal)], dtype=np.float32)
        if state["backend"] == "faiss"
        else state["vectors"]
    )

    # Check for duplicate content_hash
    ch = record.get("content_hash", "")
    for r in existing_records:
        if r.get("content_hash") == ch:
            print(f"[memory/store] Duplicate content_hash {ch!r} — skipping write-back")
            return

    record["vec_index"] = len(existing_records)
    new_vectors = np.vstack([existing_vectors, vector.reshape(1, -1)])
    all_records = existing_records + [record]
    vocab = state["vocab"]

    save_index(all_records, new_vectors, vocab, index_dir)
    print(f"[memory/store] Write-back: added record id={record.get('id')!r}")
