"""
memory/embed.py — fully offline text embedding.

Strategy (in order):
  1. Load a local SentenceTransformer from models/embeddings/code-memory-embedder
     if that path exists.  NEVER pass a remote HuggingFace model ID.
  2. Fall back to a deterministic word+bigram TF-IDF cosine-similarity engine
     implemented with NumPy only.  Zero network access.

The module fails closed: if the local model path is absent it prints a warning
and uses the fallback — it never attempts to download anything.
"""

import re
from pathlib import Path
from typing import Optional

import numpy as np

# ── Embedding model path ──────────────────────────────────────────────────────
# Must be a local directory.  Set to None to force the TF-IDF fallback.
EMBED_MODEL_PATH = Path("models/embeddings/code-memory-embedder")

_embedder: Optional[tuple] = None   # ("st", model) | ("tfidf", None)


def _load_embedder() -> tuple:
    global _embedder
    if _embedder is not None:
        return _embedder

    if EMBED_MODEL_PATH.exists():
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore
            # Load strictly from local path — no network call.
            local_path = str(EMBED_MODEL_PATH.resolve())
            model = SentenceTransformer(local_path)
            _embedder = ("st", model)
            print(f"[memory/embed] Loaded local embedder: {EMBED_MODEL_PATH}")
            return _embedder
        except Exception as exc:
            print(
                f"[memory/embed] Warning: failed to load local embedder "
                f"({exc}); falling back to TF-IDF."
            )

    print(
        f"[memory/embed] {EMBED_MODEL_PATH} not found — using TF-IDF fallback. "
        "Run scripts/build_vector_memory.py after placing a local model there "
        "for higher-quality retrieval."
    )
    _embedder = ("tfidf", None)
    return _embedder


def embed_texts(
    texts: list[str],
    vocab: Optional[dict] = None,
) -> tuple[np.ndarray, dict]:
    """Embed a list of texts.

    Returns:
      vectors : float32 array of shape (N, D), L2-normalised
      vocab   : token→index dict (empty for ST embedder; needed for TF-IDF retrieval)
    """
    kind, model = _load_embedder()

    if kind == "st":
        vecs = model.encode(texts, show_progress_bar=False, normalize_embeddings=True)
        return vecs.astype(np.float32), {}

    return _tfidf_embed(texts, vocab)


def embed_query(
    text: str,
    stored_texts: list[str],
    vocab: Optional[dict] = None,
) -> np.ndarray:
    """Embed a single query for retrieval.

    For ST: embed independently (vocab not needed).
    For TF-IDF: append query to stored texts so IDF is computed consistently,
    then return the last row.

    Returns shape (1, D) float32.
    """
    kind, model = _load_embedder()

    if kind == "st":
        vec = model.encode([text], show_progress_bar=False, normalize_embeddings=True)
        return vec.astype(np.float32)

    all_texts = list(stored_texts) + [text]
    vecs, _ = _tfidf_embed(all_texts, vocab)
    return vecs[-1:].astype(np.float32)


# ── TF-IDF fallback ───────────────────────────────────────────────────────────

def _tokenize(text: str) -> list[str]:
    words = re.findall(r"\w+", text.lower())
    bigrams = [f"{words[i]}_{words[i + 1]}" for i in range(len(words) - 1)]
    return words + bigrams


def _tfidf_embed(
    texts: list[str],
    vocab: Optional[dict] = None,
) -> tuple[np.ndarray, dict]:
    """Word + bigram TF-IDF, L2-normalised.  Deterministic, pure NumPy."""
    tokenized = [_tokenize(t) for t in texts]

    if vocab is None:
        vocab = {}
        for toks in tokenized:
            for tok in toks:
                if tok not in vocab:
                    vocab[tok] = len(vocab)

    V = len(vocab) or 1
    N = len(texts)

    tf = np.zeros((N, V), dtype=np.float32)
    for i, toks in enumerate(tokenized):
        for tok in toks:
            if tok in vocab:
                tf[i, vocab[tok]] += 1.0

    # Add-1 smoothed IDF
    df = (tf > 0).sum(axis=0) + 1.0
    idf = np.log((N + 1.0) / df) + 1.0
    vecs = tf * idf

    # L2 normalise
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms = np.where(norms < 1e-8, 1.0, norms)
    return (vecs / norms).astype(np.float32), vocab
