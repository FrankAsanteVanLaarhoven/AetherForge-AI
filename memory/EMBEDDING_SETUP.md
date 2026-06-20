# Local Embedding Model Setup

AetherForge memory uses a local SentenceTransformer for embedding.
The expected path is: `models/embeddings/code-memory-embedder`

If the path is absent, TF-IDF (NumPy-only, no network) is used automatically.

---

## Air-gap transfer procedure

### Step 1 — Download on an internet-connected staging machine

```bash
pip install sentence-transformers
python - <<'EOF'
from sentence_transformers import SentenceTransformer
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
model.save("code-memory-embedder")
EOF
```

Any code-optimised ST model works. `all-MiniLM-L6-v2` is small (22 MB) and
retrieves code tasks well enough. Replace with a larger model if GPU memory
allows — the path name must stay `code-memory-embedder`.

### Step 2 — Archive + checksum

```bash
tar -czf code-memory-embedder.tar.gz code-memory-embedder/
sha256sum code-memory-embedder.tar.gz > code-memory-embedder.tar.gz.sha256
```

### Step 3 — Transfer to air-gapped machine

Copy both files via USB/approved medium to the target machine.

### Step 4 — Verify checksum and extract

```bash
sha256sum -c code-memory-embedder.tar.gz.sha256
tar -xzf code-memory-embedder.tar.gz -C /path/to/AetherForge-AI/models/embeddings/
```

The model must be at:
```
AetherForge-AI/models/embeddings/code-memory-embedder/
```

### Step 5 — Rebuild memory index

```bash
make memory-build-full
# or manually:
conda run -n ml-torch python scripts/build_vector_memory.py
```

The build log will confirm which backend was used:

- `[memory/embed] Loaded local embedder: models/embeddings/code-memory-embedder` — ST in use
- `[memory/embed] ... not found — using TF-IDF fallback` — using NumPy fallback

---

## Verifying no network access

Run the memory test suite. The network guard will fail if any socket.connect is
attempted:

```bash
conda run -n ml-torch python -m pytest tests/test_vector_memory.py -v -k network
```

---

## Notes

- `models/` is in `.gitignore` — the model is never committed to git.
- The ST loader uses `SentenceTransformer(local_path)` with a resolved
  absolute path. Sentence-transformers will NOT fall back to HuggingFace Hub
  when given an absolute path that exists.
- If the local path exists but is corrupted, `embed.py` catches the exception,
  prints a warning, and falls back to TF-IDF. No network call is attempted.
