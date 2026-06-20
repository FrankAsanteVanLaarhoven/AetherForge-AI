"""
memory — offline vector memory for AetherForge code agent.

Air-gapped, fully local. No network calls, no remote APIs, no auto-downloads.
Fails closed if the local embedding model or index is missing.

Components:
  validate  — record schema and quality gates
  embed     — local embedding (sentence-transformers or TF-IDF fallback)
  store     — FAISS/numpy index + SQLite metadata
  core      — build, retrieve, write-back API
"""

from memory.core import build_index, retrieve, write_memory

__all__ = ["build_index", "retrieve", "write_memory"]
