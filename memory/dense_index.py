"""
memory/dense_index.py — build and save a dense vector index from memory records.

File layout under output_dir/:
  dense_vectors.npy    — float32 array (N, D), L2-normalised
  dense_model.txt      — model name used for embedding
  record_ids.json      — ordered list of record IDs (maps row → id)
  query_texts.json     — query_text for each record (for hybrid retrieval)
  records.jsonl        — human-readable copy of embedded records

Usage:
  python scripts/build_dense_memory_index.py \\
      --source-index memory/index_adapted \\
      --output-dir memory/dense_index_adapted \\
      --dense-model sentence-transformers/all-MiniLM-L6-v2
"""

import json
from pathlib import Path

import numpy as np


def _load_dense_model(model_name: str, device: str = "auto"):
    """Load a SentenceTransformer model. Raises ImportError if not installed."""
    try:
        from sentence_transformers import SentenceTransformer  # type: ignore
    except ImportError:
        raise ImportError(
            "sentence-transformers is required for dense retrieval. "
            "Install with: pip install sentence-transformers"
        )

    import torch  # type: ignore
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Allow local paths or model IDs.
    local = Path(model_name)
    load_target = str(local.resolve()) if local.exists() else model_name
    model = SentenceTransformer(load_target, device=device)
    return model, device


def build_dense_index(
    records: list[dict],
    output_dir: Path,
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    batch_size: int = 32,
    device: str = "auto",
) -> dict:
    """Embed records with a dense model and save to output_dir.

    Args:
        records:    list of memory records (must have 'id' and 'query_text')
        output_dir: directory to write index files (created if absent)
        model_name: SentenceTransformer model name or local path
        batch_size: encoding batch size
        device:     "cpu", "cuda", or "auto"

    Returns:
        dict with keys: model_name, device, n_records, dim, output_dir
    """
    if not records:
        raise ValueError("No records to index.")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[dense_index] Loading model: {model_name}")
    model, resolved_device = _load_dense_model(model_name, device)
    print(f"[dense_index] Device: {resolved_device}")

    texts = [r["query_text"] for r in records]
    ids = [r["id"] for r in records]

    print(f"[dense_index] Embedding {len(texts)} records (batch={batch_size}) ...")
    vecs = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
        device=resolved_device,
    )
    vecs = vecs.astype(np.float32)

    # Save components
    np.save(str(output_dir / "dense_vectors.npy"), vecs)
    (output_dir / "dense_model.txt").write_text(model_name)
    (output_dir / "record_ids.json").write_text(json.dumps(ids))
    (output_dir / "query_texts.json").write_text(
        json.dumps([r["query_text"] for r in records])
    )
    with open(output_dir / "records.jsonl", "w") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"[dense_index] Saved {len(records)} vectors ({vecs.shape[1]}d) to {output_dir}")
    return {
        "model_name": model_name,
        "device": resolved_device,
        "n_records": len(records),
        "dim": vecs.shape[1],
        "output_dir": str(output_dir),
    }


def load_dense_index(index_dir: Path) -> dict:
    """Load a pre-built dense index from disk.

    Returns dict with keys:
        vectors      — float32 ndarray (N, D)
        record_ids   — list of str
        query_texts  — list of str
        records      — list of dicts (from records.jsonl)
        model_name   — str
    """
    index_dir = Path(index_dir)
    vectors_path = index_dir / "dense_vectors.npy"
    if not vectors_path.exists():
        raise FileNotFoundError(f"Dense index not found at {index_dir}. Build it first.")

    vectors = np.load(str(vectors_path))
    record_ids = json.loads((index_dir / "record_ids.json").read_text())
    query_texts = json.loads((index_dir / "query_texts.json").read_text())
    model_name = (index_dir / "dense_model.txt").read_text().strip()

    records = []
    rjsonl = index_dir / "records.jsonl"
    if rjsonl.exists():
        with open(rjsonl) as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))

    return {
        "vectors": vectors,
        "record_ids": record_ids,
        "query_texts": query_texts,
        "records": records,
        "model_name": model_name,
    }
