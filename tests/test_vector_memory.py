"""
tests/test_vector_memory.py — offline vector memory test suite.

All tests run fully offline:
  - No network access (socket.connect is monkey-patched to assert no calls)
  - No remote embedding API
  - No auto-downloads
  - Index built from seed_memories.jsonl only

Run:
    conda run -n ml-torch python -m pytest tests/test_vector_memory.py -v
    # or standalone:
    conda run -n ml-torch python tests/test_vector_memory.py
"""

import json
import os
import socket
import sys
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from memory.validate import validate_record, content_hash
from memory.embed import embed_texts, embed_query, _tfidf_embed, _tokenize
from memory.store import save_index, load_index, search
from memory.core import build_index, retrieve, format_memory_block


# ── Network guard ─────────────────────────────────────────────────────────────

_original_connect = socket.socket.connect

def _no_network(self, *args, **kwargs):
    raise AssertionError(
        f"Network call attempted during memory test! Args: {args}  "
        "Memory module must be fully offline."
    )


def setUpModule():
    socket.socket.connect = _no_network


def tearDownModule():
    socket.socket.connect = _original_connect


# ── Fixtures ──────────────────────────────────────────────────────────────────

SEED_JSONL = ROOT / "memory" / "raw" / "seed_memories.jsonl"

def _load_seeds() -> list[dict]:
    records = []
    with open(SEED_JSONL) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _make_valid_record(**overrides) -> dict:
    import uuid, hashlib
    code = "def f():\n    return 1\nassert f() == 1\nprint('PASS')"
    ctc = "TOOL_CALL: execute_code(" + json.dumps({"code": code}) + ")"
    base = {
        "id": str(uuid.uuid4()),
        "task": "Write f() returning 1",
        "category": "general",
        "failure_type": "no_output",
        "query_text": "function returning integer 1",
        "corrected_tool_call": ctc,
        "observation": "PASS",
        "final_answer": "FINAL_ANSWER: f verified: PASS.",
        "source": "test",
        "verified": True,
        "created_at": "2026-06-20T00:00:00+00:00",
        "content_hash": "sha256:test",
        "sensitivity": "internal",
    }
    base.update(overrides)
    return base


# ── Validation tests ──────────────────────────────────────────────────────────

class TestValidation(unittest.TestCase):

    def test_valid_seed_records_pass(self):
        """All seed records must pass validation."""
        records = _load_seeds()
        self.assertGreater(len(records), 0, "No seed records found")
        for rec in records:
            errs = validate_record(rec)
            self.assertEqual(errs, [], f"Seed record {rec.get('id')} failed: {errs}")

    def test_missing_required_field_fails(self):
        for field in ["id", "task", "query_text", "corrected_tool_call", "observation", "verified"]:
            rec = _make_valid_record()
            del rec[field]
            errs = validate_record(rec)
            self.assertTrue(
                any(field in e for e in errs),
                f"Expected error about missing '{field}', got: {errs}"
            )

    def test_unverified_record_fails(self):
        rec = _make_valid_record(verified=False)
        errs = validate_record(rec)
        self.assertTrue(any("verified" in e for e in errs), errs)

    def test_no_pass_in_observation_fails(self):
        rec = _make_valid_record(observation="ERROR: something went wrong")
        errs = validate_record(rec)
        self.assertTrue(any("observation" in e.lower() for e in errs), errs)

    def test_markdown_fence_in_ctc_fails(self):
        bad_ctc = "TOOL_CALL: execute_code({\"code\": \"```python\ndef f(): pass\n```\"})"
        rec = _make_valid_record(corrected_tool_call=bad_ctc)
        errs = validate_record(rec)
        self.assertTrue(any("fence" in e.lower() or "```" in e for e in errs), errs)

    def test_triple_quote_in_ctc_fails(self):
        bad_ctc = 'TOOL_CALL: execute_code({"code": """def f(): pass"""})'
        rec = _make_valid_record(corrected_tool_call=bad_ctc)
        errs = validate_record(rec)
        self.assertTrue(any('"""' in e for e in errs), errs)

    def test_valid_marker_fails(self):
        rec = _make_valid_record(final_answer="VALID: this is valid")
        errs = validate_record(rec)
        self.assertTrue(any("VALID:" in e for e in errs), errs)

    def test_wrong_ctc_prefix_fails(self):
        rec = _make_valid_record(corrected_tool_call="execute_code({\"code\": \"pass\"})")
        errs = validate_record(rec)
        self.assertTrue(any("corrected_tool_call" in e for e in errs), errs)

    def test_invalid_json_ctc_fails(self):
        rec = _make_valid_record(corrected_tool_call="TOOL_CALL: execute_code({'code': 'pass'})")
        errs = validate_record(rec)
        self.assertTrue(any("JSON" in e for e in errs), errs)


# ── Embedding tests ───────────────────────────────────────────────────────────

class TestEmbedding(unittest.TestCase):

    def test_tfidf_embed_returns_normalized_vectors(self):
        texts = ["word_count frequency dict", "factorial iterative", "BFS graph"]
        vecs, vocab = _tfidf_embed(texts)
        self.assertEqual(vecs.shape[0], 3)
        import numpy as np
        norms = np.linalg.norm(vecs, axis=1)
        for n in norms:
            self.assertAlmostEqual(n, 1.0, places=5, msg="Vectors must be L2-normalised")

    def test_tfidf_embed_consistent_vocab(self):
        texts = ["the cat sat", "on the mat"]
        vecs1, vocab = _tfidf_embed(texts)
        vecs2, _ = _tfidf_embed(texts, vocab=vocab)
        import numpy as np
        np.testing.assert_array_almost_equal(vecs1, vecs2, decimal=5,
            err_msg="Same texts + same vocab must produce same vectors")

    def test_embed_query_same_shape(self):
        stored = ["word count task", "factorial function"]
        # Use embed_texts so the query dim matches whichever backend is active.
        vecs, vocab = embed_texts(stored)
        qvec = embed_query("word count", stored, vocab)
        self.assertEqual(qvec.ndim, 2)
        self.assertEqual(qvec.shape[0], 1)
        self.assertEqual(qvec.shape[1], vecs.shape[1])

    def test_embed_texts_no_network(self):
        """embed_texts must not attempt a network connection."""
        texts = ["test query for offline embedding"]
        vecs, vocab = embed_texts(texts)
        self.assertEqual(vecs.shape[0], 1)


# ── Store tests ───────────────────────────────────────────────────────────────

class TestStore(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.index_dir = Path(self.tmp) / "index"

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _build(self, records):
        texts = [r["query_text"] for r in records]
        vecs, vocab = embed_texts(texts)
        save_index(records, vecs, vocab, self.index_dir)
        return load_index(self.index_dir)

    def test_save_and_load(self):
        records = [_make_valid_record(query_text="word count function")]
        state = self._build(records)
        self.assertEqual(len(state["records"]), 1)

    def test_search_returns_results(self):
        recs = [
            _make_valid_record(id="1", query_text="word count frequency", category="word_count"),
            _make_valid_record(id="2", query_text="factorial iterative", category="factorial"),
            _make_valid_record(id="3", query_text="BFS graph breadth first", category="bfs"),
        ]
        state = self._build(recs)
        qvec = embed_query("word count", state["query_texts"], state["vocab"])
        hits = search(state, qvec, top_k=2)
        self.assertGreater(len(hits), 0)
        # Top result should be word_count
        self.assertEqual(hits[0]["category"], "word_count")

    def test_missing_index_raises(self):
        from memory.store import load_index as _li
        with self.assertRaises(FileNotFoundError):
            _li(Path(self.tmp) / "nonexistent")


# ── Build + retrieve integration tests ───────────────────────────────────────

class TestBuildAndRetrieve(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.raw_dir   = Path(self.tmp) / "raw"
        self.index_dir = Path(self.tmp) / "index"
        self.raw_dir.mkdir()
        # Copy seed records to tmp raw dir
        records = _load_seeds()
        out = self.raw_dir / "seeds.jsonl"
        with open(out, "w") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_build_memory_offline(self):
        """build_index must succeed without any network access."""
        n = build_index(raw_dir=self.raw_dir, index_dir=self.index_dir)
        self.assertGreater(n, 0)
        self.assertTrue((self.index_dir / "backend.txt").exists())
        self.assertTrue((self.index_dir / "metadata.sqlite").exists())

    def test_retrieval_returns_word_count(self):
        build_index(raw_dir=self.raw_dir, index_dir=self.index_dir)
        results = retrieve("word_count frequency dict case-insensitive",
                           index_dir=self.index_dir, top_k=3)
        self.assertGreater(len(results), 0)
        categories = [r.get("category") for r in results]
        self.assertIn("word_count", categories,
                      f"Expected word_count in top results, got {categories}")

    def test_retrieval_returns_lru_cache(self):
        build_index(raw_dir=self.raw_dir, index_dir=self.index_dir)
        results = retrieve("LRU cache OrderedDict get put eviction",
                           index_dir=self.index_dir, top_k=3)
        self.assertGreater(len(results), 0)
        categories = [r.get("category") for r in results]
        self.assertIn("lru_cache", categories,
                      f"Expected lru_cache in top results, got {categories}")

    def test_retrieval_returns_bfs(self):
        build_index(raw_dir=self.raw_dir, index_dir=self.index_dir)
        results = retrieve("BFS breadth first search graph sorted neighbors",
                           index_dir=self.index_dir, top_k=3)
        self.assertGreater(len(results), 0)
        categories = [r.get("category") for r in results]
        self.assertIn("graph_bfs", categories,
                      f"Expected graph_bfs in top results, got {categories}")

    def test_all_retrieved_are_verified(self):
        build_index(raw_dir=self.raw_dir, index_dir=self.index_dir)
        results = retrieve("any task", index_dir=self.index_dir, top_k=8)
        for r in results:
            self.assertTrue(r.get("verified"), f"Unverified record in results: {r.get('id')}")

    def test_retrieval_fails_closed_if_no_index(self):
        """retrieve() must return empty list (not raise) if index is missing."""
        results = retrieve("any task", index_dir=Path(self.tmp) / "nonexistent")
        self.assertEqual(results, [])

    def test_invalid_record_rejected_from_build(self):
        """build_index must abort if any record fails validation."""
        bad = _make_valid_record(verified=False)
        out = self.raw_dir / "bad.jsonl"
        with open(out, "w") as f:
            f.write(json.dumps(bad) + "\n")
        with self.assertRaises(SystemExit):
            build_index(raw_dir=self.raw_dir, index_dir=self.index_dir)


# ── Write-back safety tests ───────────────────────────────────────────────────

class TestWriteBackSafety(unittest.TestCase):

    def test_write_back_disabled_by_default(self):
        """write_memory() must raise RuntimeError when write_back_enabled=False."""
        from memory import core as _core
        self.assertFalse(_core.write_back_enabled,
                         "write_back_enabled must be False at module level")
        from memory.core import write_memory
        rec = _make_valid_record()
        with self.assertRaises(RuntimeError):
            write_memory(rec)


# ── Memory block format tests ─────────────────────────────────────────────────

class TestFormatMemoryBlock(unittest.TestCase):

    def test_format_is_not_observation(self):
        """The memory block must start with RETRIEVED_VERIFIED_MEMORY:, not OBSERVATION:."""
        recs = [_make_valid_record(category="word_count", score=0.9)]
        block = format_memory_block(recs)
        self.assertTrue(block.startswith("RETRIEVED_VERIFIED_MEMORY:"), block[:60])
        self.assertNotIn("OBSERVATION:", block,
                         "Memory block must not contain OBSERVATION: — that would bypass runtime")

    def test_format_empty_input(self):
        self.assertEqual(format_memory_block([]), "")

    def test_format_contains_tool_call(self):
        recs = _load_seeds()[:1]
        block = format_memory_block(recs)
        self.assertIn("TOOL_CALL:", block)

    def test_format_no_markdown_fences(self):
        recs = _load_seeds()[:3]
        block = format_memory_block(recs)
        self.assertNotIn("```", block, "Memory block must not contain markdown code fences")


# ── verified_agent scoring bypass test ───────────────────────────────────────

class TestScoringBypass(unittest.TestCase):
    """Ensure retrieved memory cannot substitute for a real OBSERVATION: PASS."""

    def test_retrieved_memory_not_counted_as_observation(self):
        """The memory block format uses RETRIEVED_VERIFIED_MEMORY:, not OBSERVATION:.

        If the agent outputs RETRIEVED_VERIFIED_MEMORY: instead of running code,
        verified_agent scoring must NOT count it as a pass.
        """
        recs = _load_seeds()[:2]
        block = format_memory_block(recs)
        # Simulate what would happen if the model echoed back the memory block
        fake_agent_output = block + "\nFINAL_ANSWER: done"
        self.assertNotIn("OBSERVATION: PASS", fake_agent_output,
                         "Memory block must never contain OBSERVATION: PASS")
        # verified_agent looks for 'PASS' in actual tool call observations
        has_real_pass = "OBSERVATION: PASS" in fake_agent_output
        self.assertFalse(has_real_pass,
                         "Memory echoed as output should not produce a verified PASS")


# ── Standalone runner ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"Python: {sys.version}")
    print(f"Root:   {ROOT}")
    print(f"Seeds:  {SEED_JSONL}")
    print()

    loader = unittest.TestLoader()
    suite  = loader.loadTestsFromModule(sys.modules[__name__])
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)
