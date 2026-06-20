"""
tests/test_extract_memory.py — unit tests for extract_memory_from_evals.py.

Tests are self-contained: they create synthetic CSV rows instead of requiring
actual outputs/ files.  All tests run fully offline (socket guard active).

Run:
    conda run -n ml-torch python -m pytest tests/test_extract_memory.py -v
    # or standalone:
    conda run -n ml-torch python tests/test_extract_memory.py
"""

import csv
import io
import json
import socket
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from scripts.extract_memory_from_evals import (
    _extract_passing_tool_call,
    _row_qualifies,
    _build_record,
    _infer_failure_type,
    _infer_category,
    _tool_call_is_clean,
    scan_csv,
    extract,
)
from memory.validate import validate_record


# ── Network guard ─────────────────────────────────────────────────────────────

_original_connect = socket.socket.connect

def _no_network(self, *args, **kwargs):
    raise AssertionError(
        f"Network call during extract_memory test! Args: {args}"
    )


def setUpModule():
    socket.socket.connect = _no_network


def tearDownModule():
    socket.socket.connect = _original_connect


# ── CSV helpers ───────────────────────────────────────────────────────────────

_CSV_COLS = [
    "id", "category", "passed", "passed_via_tool", "passed_via_fallback",
    "used_fallback_extraction", "full_transcript", "first_tool_call",
    "observations", "final_answer", "n_errors", "has_invalid_json",
    "has_indentation_error", "first_exception_type", "scoring_mode",
    "run_label", "no_output_count",
]

_CODE_OK = (
    "def word_count(text):\n"
    "    words = text.lower().split()\n"
    "    counts = {}\n"
    "    for w in words:\n"
    "        counts[w] = counts.get(w, 0) + 1\n"
    "    return counts\n"
    "r = word_count('the cat sat on the mat')\n"
    "assert r['the'] == 2\n"
    "print('PASS')"
)

def _make_transcript(code: str = _CODE_OK, obs: str = "PASS") -> str:
    ctc_json = json.dumps({"code": code})
    return (
        f"TOOL_CALL: execute_code({ctc_json})\n"
        f"OBSERVATION: {obs}\n"
        f"FINAL_ANSWER: word_count verified: PASS."
    )


def _make_row(**overrides) -> dict:
    defaults = {
        "id":                      "word_count",
        "category":                "medium",
        "passed":                  "True",
        "passed_via_tool":         "True",
        "passed_via_fallback":     "False",
        "used_fallback_extraction":"False",
        "full_transcript":         _make_transcript(),
        "first_tool_call":         f"execute_code({json.dumps({'code': _CODE_OK})})",
        "observations":            "PASS",
        "final_answer":            "FINAL_ANSWER: word_count verified: PASS.",
        "n_errors":                "0",
        "has_invalid_json":        "False",
        "has_indentation_error":   "False",
        "first_exception_type":    "",
        "scoring_mode":            "verified_agent",
        "run_label":               "single",
        "no_output_count":         "0",
    }
    defaults.update(overrides)
    return defaults


def _write_csv(rows: list[dict], path: Path):
    cols = sorted({k for row in rows for k in row})
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for row in rows:
            w.writerow(row)


# ── Transcript parsing tests ──────────────────────────────────────────────────

class TestExtractPassingToolCall(unittest.TestCase):

    def test_extracts_clean_tool_call(self):
        transcript = _make_transcript()
        ctc = _extract_passing_tool_call(transcript)
        self.assertIsNotNone(ctc)
        self.assertTrue(ctc.startswith("TOOL_CALL: execute_code("), ctc[:60])

    def test_returns_none_when_no_observation_pass(self):
        transcript = (
            'TOOL_CALL: execute_code({"code": "print(1)"})\n'
            "OBSERVATION: (no output)\n"
        )
        result = _extract_passing_tool_call(transcript)
        self.assertIsNone(result)

    def test_returns_last_tool_call_before_pass(self):
        """When there are multiple tool calls, take the one before OBSERVATION: PASS."""
        code_bad = "def f(): return 0\nprint(f())"
        code_good = "def f(): return 1\nassert f() == 1\nprint('PASS')"
        ctc_bad  = f"TOOL_CALL: execute_code({json.dumps({'code': code_bad})})"
        ctc_good = f"TOOL_CALL: execute_code({json.dumps({'code': code_good})})"
        transcript = (
            f"{ctc_bad}\n"
            "OBSERVATION: (no output)\n"
            f"{ctc_good}\n"
            "OBSERVATION: PASS\n"
        )
        ctc = _extract_passing_tool_call(transcript)
        self.assertIsNotNone(ctc)
        self.assertIn("code_good" if False else "assert f() == 1", ctc)

    def test_handles_newlines_in_code(self):
        code = "def f():\n    pass\nf()\nprint('PASS')"
        transcript = _make_transcript(code)
        ctc = _extract_passing_tool_call(transcript)
        self.assertIsNotNone(ctc)

    def test_extracted_is_valid_json(self):
        transcript = _make_transcript()
        ctc = _extract_passing_tool_call(transcript)
        self.assertIsNotNone(ctc)
        prefix = "TOOL_CALL: execute_code("
        body = ctc[len(prefix):-1]
        args = json.loads(body)
        self.assertIn("code", args)


# ── Row qualification tests ───────────────────────────────────────────────────

class TestRowQualifies(unittest.TestCase):

    def test_valid_row_qualifies(self):
        row = _make_row()
        ok, reason = _row_qualifies(row, "verified_tool")
        self.assertTrue(ok, reason)

    def test_rejects_not_passed(self):
        row = _make_row(passed="False")
        ok, reason = _row_qualifies(row, "verified_tool")
        self.assertFalse(ok)
        self.assertIn("passed", reason)

    def test_rejects_not_passed_via_tool(self):
        row = _make_row(passed_via_tool="False")
        ok, reason = _row_qualifies(row, "verified_tool")
        self.assertFalse(ok)
        self.assertIn("passed_via_tool", reason)

    def test_rejects_fallback_extraction(self):
        row = _make_row(used_fallback_extraction="True")
        ok, reason = _row_qualifies(row, "verified_tool")
        self.assertFalse(ok)
        self.assertIn("fallback", reason)

    def test_rejects_empty_transcript(self):
        row = _make_row(full_transcript="")
        ok, reason = _row_qualifies(row, "verified_tool")
        self.assertFalse(ok)
        self.assertIn("transcript", reason)


# ── Banned-marker tests ───────────────────────────────────────────────────────

class TestToolCallClean(unittest.TestCase):

    def test_clean_tool_call_passes(self):
        ctc = f"TOOL_CALL: execute_code({json.dumps({'code': 'print(1)'})})"
        reasons = _tool_call_is_clean(ctc)
        self.assertEqual(reasons, [])

    def test_rejects_valid_colon(self):
        ctc = f"TOOL_CALL: execute_code({json.dumps({'code': 'VALID: ok'})})"
        reasons = _tool_call_is_clean(ctc)
        self.assertTrue(any("VALID:" in r for r in reasons), reasons)

    def test_rejects_different_corrected_tool_call(self):
        ctc = (
            f"TOOL_CALL: execute_code({json.dumps({'code': 'x=1'})})\n"
            "DIFFERENT_CORRECTED_TOOL_CALL: ..."
        )
        reasons = _tool_call_is_clean(ctc)
        self.assertTrue(any("DIFFERENT_CORRECTED_TOOL_CALL" in r for r in reasons), reasons)

    def test_rejects_markdown_fences(self):
        bad_code = "```python\nprint(1)\n```"
        ctc = f"TOOL_CALL: execute_code({json.dumps({'code': bad_code})})"
        reasons = _tool_call_is_clean(ctc)
        self.assertTrue(any("```" in r for r in reasons), reasons)

    def test_rejects_triple_quote_json(self):
        ctc = 'TOOL_CALL: execute_code({"code": """def f(): pass"""})'
        reasons = _tool_call_is_clean(ctc)
        self.assertTrue(any('"""' in r for r in reasons), reasons)

    def test_rejects_revised_tool_call_marker(self):
        ctc = (
            f"TOOL_CALL: execute_code({json.dumps({'code': 'x=1'})})\n"
            "Revised TOOL_CALL: execute_code(...)"
        )
        reasons = _tool_call_is_clean(ctc)
        self.assertTrue(any("Revised TOOL_CALL:" in r for r in reasons), reasons)


# ── Record construction tests ─────────────────────────────────────────────────

class TestBuildRecord(unittest.TestCase):

    def _valid_ctc(self) -> str:
        return f"TOOL_CALL: execute_code({json.dumps({'code': _CODE_OK})})"

    def test_builds_valid_record(self):
        row = _make_row()
        ctc = self._valid_ctc()
        rec = _build_record(row, ctc, task_map={}, source="test/single.csv")
        self.assertIsNotNone(rec)
        errs = validate_record(rec)
        self.assertEqual(errs, [], f"Record failed validation: {errs}")

    def test_all_required_fields_present(self):
        row = _make_row()
        ctc = self._valid_ctc()
        rec = _build_record(row, ctc, task_map={}, source="test")
        for field in ["id", "task", "category", "query_text", "corrected_tool_call",
                      "observation", "verified", "created_at", "content_hash"]:
            self.assertIn(field, rec, f"Missing field: {field}")

    def test_verified_is_true(self):
        row = _make_row()
        ctc = self._valid_ctc()
        rec = _build_record(row, ctc, task_map={}, source="test")
        self.assertTrue(rec["verified"])

    def test_observation_is_pass(self):
        row = _make_row()
        ctc = self._valid_ctc()
        rec = _build_record(row, ctc, task_map={}, source="test")
        self.assertIn("PASS", rec["observation"])

    def test_content_hash_is_set(self):
        row = _make_row()
        ctc = self._valid_ctc()
        rec = _build_record(row, ctc, task_map={}, source="test")
        self.assertTrue(rec["content_hash"].startswith("sha256:"), rec["content_hash"])

    def test_uses_task_map_description(self):
        task_map = {"word_count": {"task": "FULL TASK DESCRIPTION HERE", "id": "word_count"}}
        row = _make_row()
        ctc = self._valid_ctc()
        rec = _build_record(row, ctc, task_map=task_map, source="test")
        self.assertEqual(rec["task"], "FULL TASK DESCRIPTION HERE")


# ── CSV scan tests ────────────────────────────────────────────────────────────

class TestScanCsv(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _write_rows(self, rows: list[dict], filename: str = "single.csv") -> Path:
        p = Path(self.tmp) / filename
        _write_csv(rows, p)
        return p

    def test_extracts_valid_row(self):
        path = self._write_rows([_make_row()])
        recs = scan_csv(path, {}, set(), "verified_tool")
        self.assertEqual(len(recs), 1)
        self.assertTrue(recs[0]["verified"])

    def test_rejects_unverified_row(self):
        path = self._write_rows([_make_row(passed="False")])
        recs = scan_csv(path, {}, set(), "verified_tool")
        self.assertEqual(len(recs), 0)

    def test_rejects_fallback_row(self):
        path = self._write_rows([_make_row(used_fallback_extraction="True")])
        recs = scan_csv(path, {}, set(), "verified_tool")
        self.assertEqual(len(recs), 0)

    def test_rejects_markdown_fence_in_transcript(self):
        bad_code = "```python\ndef f(): pass\n```"
        transcript = _make_transcript(bad_code)
        path = self._write_rows([_make_row(full_transcript=transcript)])
        recs = scan_csv(path, {}, set(), "verified_tool")
        self.assertEqual(len(recs), 0, "Markdown fence must be rejected")

    def test_rejects_valid_colon_in_transcript(self):
        transcript = _make_transcript("VALID: this is wrong")
        path = self._write_rows([_make_row(full_transcript=transcript)])
        recs = scan_csv(path, {}, set(), "verified_tool")
        self.assertEqual(len(recs), 0, "VALID: marker must be rejected")

    def test_rejects_different_corrected_tool_call(self):
        code = "x = 1\nprint('PASS')"
        # Inject the banned marker inside the transcript but outside the JSON
        ctc_json = json.dumps({"code": code})
        transcript = (
            f"TOOL_CALL: execute_code({ctc_json})\n"
            "OBSERVATION: PASS\n"
            "DIFFERENT_CORRECTED_TOOL_CALL: something else"
        )
        path = self._write_rows([_make_row(full_transcript=transcript)])
        # The extracted TOOL_CALL itself won't contain the banned marker
        # (it ends before OBSERVATION:), so this is actually acceptable here.
        # The banned marker only matters if it appears inside the extracted CTC.
        # So we test: banned marker INSIDE the JSON code value
        bad_code = "DIFFERENT_CORRECTED_TOOL_CALL\nprint('PASS')"
        transcript2 = _make_transcript(bad_code)
        path2 = self._write_rows([_make_row(full_transcript=transcript2)])
        recs = scan_csv(path2, {}, set(), "verified_tool")
        self.assertEqual(len(recs), 0, "DIFFERENT_CORRECTED_TOOL_CALL in code must be rejected")

    def test_rejects_triple_quote_json(self):
        # Triple-quoted string in the transcript — not valid JSON
        bad_transcript = (
            'TOOL_CALL: execute_code({"code": """def f(): pass"""})\n'
            "OBSERVATION: PASS\n"
        )
        path = self._write_rows([_make_row(full_transcript=bad_transcript)])
        recs = scan_csv(path, {}, set(), "verified_tool")
        self.assertEqual(len(recs), 0, "Triple-quoted JSON must be rejected")

    def test_deduplicates_same_tool_call(self):
        """Two rows with the same code → only one record."""
        rows = [_make_row(), _make_row()]
        path = self._write_rows(rows)
        seen: set[str] = set()
        recs = scan_csv(path, {}, seen, "verified_tool")
        self.assertEqual(len(recs), 1, "Duplicate content_hash must be deduplicated")

    def test_deduplicates_across_files(self):
        """Same record in two files → only one record total."""
        p1 = self._write_rows([_make_row()], "file1.csv")
        p2 = self._write_rows([_make_row()], "file2.csv")
        seen: set[str] = set()
        recs1 = scan_csv(p1, {}, seen, "verified_tool")
        recs2 = scan_csv(p2, {}, seen, "verified_tool")
        self.assertEqual(len(recs1) + len(recs2), 1,
                         "Same hash in two files must not be added twice")

    def test_all_extracted_records_pass_validate(self):
        rows = [
            _make_row(id="word_count"),
            _make_row(id="factorial",
                      full_transcript=_make_transcript("def f(n):\n    return n\nassert f(5)==5\nprint('PASS')")),
        ]
        path = self._write_rows(rows)
        recs = scan_csv(path, {}, set(), "verified_tool")
        for rec in recs:
            errs = validate_record(rec)
            self.assertEqual(errs, [], f"Record {rec.get('id')} failed: {errs}")


# ── Full pipeline tests ───────────────────────────────────────────────────────

class TestExtractPipeline(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.out_dir = Path(self.tmp) / "memory" / "raw"
        self.outputs_dir = Path(self.tmp) / "outputs"
        self.outputs_dir.mkdir(parents=True)

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _make_eval_dir(self, dirname: str, rows: list[dict]) -> Path:
        d = self.outputs_dir / dirname
        d.mkdir(parents=True)
        _write_csv(rows, d / "single.csv")
        return d

    def test_writes_valid_jsonl(self):
        self._make_eval_dir("run1", [_make_row()])
        out = self.out_dir / "extracted.jsonl"
        n = extract(self.outputs_dir, out, max_records=500)
        self.assertGreater(n, 0)
        self.assertTrue(out.exists())
        with open(out) as f:
            lines = [l.strip() for l in f if l.strip()]
        self.assertGreater(len(lines), 0)
        for line in lines:
            rec = json.loads(line)
            errs = validate_record(rec)
            self.assertEqual(errs, [], f"Invalid JSONL record: {errs}")

    def test_returns_zero_when_no_passing_rows(self):
        self._make_eval_dir("run1", [_make_row(passed="False")])
        out = self.out_dir / "extracted.jsonl"
        n = extract(self.outputs_dir, out)
        self.assertEqual(n, 0)

    def test_respects_max_records(self):
        rows = [_make_row(id=f"task_{i}", full_transcript=_make_transcript(
            f"def f_{i}(): return {i}\nassert f_{i}() == {i}\nprint('PASS')"
        )) for i in range(20)]
        self._make_eval_dir("run1", rows)
        out = self.out_dir / "extracted.jsonl"
        n = extract(self.outputs_dir, out, max_records=5)
        self.assertLessEqual(n, 5)

    def test_extracted_records_pass_audit_validate(self):
        """Records extracted from CSVs must all pass memory.validate.validate_record."""
        rows = [
            _make_row(id="word_count"),
            _make_row(id="lru_cache",
                      full_transcript=_make_transcript("class C:\n    pass\nprint('PASS')")),
        ]
        self._make_eval_dir("run1", rows)
        out = self.out_dir / "extracted.jsonl"
        extract(self.outputs_dir, out)
        with open(out) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                errs = validate_record(rec)
                self.assertEqual(errs, [], f"Extracted record failed audit: {errs}\n{rec}")

    def test_no_network_during_extraction(self):
        """Extraction must work fully offline (network guard is active in setUpModule)."""
        self._make_eval_dir("run1", [_make_row()])
        out = self.out_dir / "extracted.jsonl"
        # If network is attempted, socket guard raises AssertionError
        extract(self.outputs_dir, out)


# ── Standalone runner ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"Python: {sys.version}")
    print(f"Root:   {ROOT}")
    print()
    loader = unittest.TestLoader()
    suite  = loader.loadTestsFromModule(sys.modules[__name__])
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)
