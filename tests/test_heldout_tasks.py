"""
Tests for the held-out and recovery-stress task suites, and the --tasks-file
extension to evaluate_code_agent.py.
"""
import json
import re
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.evaluate_code_agent import (
    BUILTIN_TASK_IDS,
    TASKS as BUILTIN_TASKS,
    load_tasks_from_jsonl,
)

HELDOUT_PATH  = ROOT / "data" / "heldout_code_agent_tasks.jsonl"
RECOVERY_PATH = ROOT / "data" / "recovery_stress_tasks.jsonl"


def _load_jsonl(path: Path) -> list[dict]:
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


# ---------------------------------------------------------------------------
# Held-out task file integrity
# ---------------------------------------------------------------------------

class TestHeldoutTaskFile(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.records = _load_jsonl(HELDOUT_PATH)

    def test_file_exists(self):
        self.assertTrue(HELDOUT_PATH.exists(), f"Missing: {HELDOUT_PATH}")

    def test_minimum_task_count(self):
        self.assertGreaterEqual(len(self.records), 24,
                                "Held-out set needs at least 24 tasks")

    def test_maximum_task_count(self):
        self.assertLessEqual(len(self.records), 32,
                             "Held-out set should not exceed 32 tasks (keep focused)")

    def test_ids_are_unique(self):
        ids = [r["id"] for r in self.records]
        self.assertEqual(len(ids), len(set(ids)),
                         f"Duplicate held-out IDs: {[x for x in ids if ids.count(x)>1]}")

    def test_ids_do_not_overlap_with_builtin(self):
        overlap = {r["id"] for r in self.records} & BUILTIN_TASK_IDS
        self.assertEqual(overlap, set(),
                         f"Held-out IDs overlap with built-in benchmark: {overlap}")

    def test_required_fields_present(self):
        for rec in self.records:
            self.assertIn("id",       rec, f"Missing 'id' in {rec}")
            self.assertIn("category", rec, f"Missing 'category' in {rec}")
            self.assertTrue(
                rec.get("prompt") or rec.get("task"),
                f"Record {rec['id']!r} has no 'prompt'/'task'"
            )

    def test_prompts_are_non_empty(self):
        for rec in self.records:
            text = rec.get("prompt", rec.get("task", "")).strip()
            self.assertGreater(len(text), 20,
                               f"Prompt for {rec['id']!r} is suspiciously short")

    def test_categories_are_strings(self):
        for rec in self.records:
            self.assertIsInstance(rec["category"], str)
            self.assertGreater(len(rec["category"]), 0)

    def test_no_exact_builtin_task_names_in_prompts(self):
        """Held-out prompts should not start with or copy exact known benchmark prompts."""
        builtin_starts = {t["task"][:40] for t in BUILTIN_TASKS}
        for rec in self.records:
            prompt_start = (rec.get("prompt") or rec.get("task", ""))[:40]
            self.assertNotIn(
                prompt_start, builtin_starts,
                f"Held-out task {rec['id']!r} appears to duplicate a built-in prompt"
            )

    def test_ids_are_snake_case(self):
        bad = [r["id"] for r in self.records if not re.match(r'^[a-z][a-z0-9_]*$', r["id"])]
        self.assertEqual(bad, [], f"Non-snake_case IDs: {bad}")


# ---------------------------------------------------------------------------
# Recovery-stress task file integrity
# ---------------------------------------------------------------------------

class TestRecoveryStressTaskFile(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.records = _load_jsonl(RECOVERY_PATH)

    def test_file_exists(self):
        self.assertTrue(RECOVERY_PATH.exists(), f"Missing: {RECOVERY_PATH}")

    def test_minimum_task_count(self):
        self.assertGreaterEqual(len(self.records), 8,
                                "Recovery-stress set needs at least 8 tasks")

    def test_ids_are_unique(self):
        ids = [r["id"] for r in self.records]
        self.assertEqual(len(ids), len(set(ids)),
                         f"Duplicate recovery IDs: {[x for x in ids if ids.count(x)>1]}")

    def test_ids_do_not_overlap_with_heldout(self):
        heldout_ids = {r["id"] for r in _load_jsonl(HELDOUT_PATH)}
        overlap = {r["id"] for r in self.records} & heldout_ids
        self.assertEqual(overlap, set(),
                         f"Recovery IDs overlap with held-out IDs: {overlap}")

    def test_prompts_contain_failed_observation(self):
        """Every recovery-stress prompt must include a failed OBSERVATION."""
        obs_pattern = re.compile(r"OBSERVATION:\s*(ERROR|PASS|\(no output\))")
        for rec in self.records:
            prompt = rec.get("prompt") or rec.get("task", "")
            self.assertRegex(
                prompt, obs_pattern,
                f"Recovery task {rec['id']!r}: prompt must contain OBSERVATION: ERROR or (no output)"
            )

    def test_prompts_contain_tool_call(self):
        """Every recovery-stress prompt must show a bad TOOL_CALL."""
        for rec in self.records:
            prompt = rec.get("prompt") or rec.get("task", "")
            self.assertIn(
                "TOOL_CALL:", prompt,
                f"Recovery task {rec['id']!r}: prompt must contain a TOOL_CALL"
            )

    def test_prompts_do_not_contain_corrected_pass(self):
        """Prompts must not leak the corrected solution with OBSERVATION: PASS."""
        for rec in self.records:
            prompt = rec.get("prompt") or rec.get("task", "")
            # The prompt may contain OBSERVATION: ERROR or (no output), but not PASS
            # (that would hand the agent the answer)
            self.assertNotIn(
                "OBSERVATION: PASS", prompt,
                f"Recovery task {rec['id']!r}: prompt leaks OBSERVATION: PASS (do not include corrected result)"
            )

    def test_prompts_do_not_include_print_pass_in_corrected_code(self):
        """Prompts should not contain print('PASS') as part of a corrected snippet."""
        # A failed attempt might show the broken code calling print('PASS') but
        # that PASS never appeared in the observation (the OBSERVATION is ERROR).
        # Guard: the corrected final TOOL_CALL with print('PASS') must not exist
        # in the prompt — the agent must write it.
        for rec in self.records:
            prompt = rec.get("prompt") or rec.get("task", "")
            lines = prompt.splitlines()
            # Allow print('PASS') only if immediately followed by OBSERVATION: ERROR
            # (i.e., the bad attempt included it but failed anyway).
            # Flag if there is a print('PASS') that appears AFTER the observation section.
            obs_idx = next(
                (i for i, ln in enumerate(lines) if "OBSERVATION:" in ln), len(lines)
            )
            after_obs = "\n".join(lines[obs_idx + 1:])
            self.assertNotIn(
                "print('PASS')", after_obs,
                f"Recovery task {rec['id']!r}: corrected code with print('PASS') found after OBSERVATION"
            )

    def test_categories_are_recovery_stress(self):
        for rec in self.records:
            self.assertEqual(
                rec.get("category"), "recovery-stress",
                f"Recovery task {rec['id']!r} should have category 'recovery-stress'"
            )


# ---------------------------------------------------------------------------
# load_tasks_from_jsonl function
# ---------------------------------------------------------------------------

class TestLoadTasksFromJsonl(unittest.TestCase):

    def _write_tmp(self, records: list) -> Path:
        tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False)
        for r in records:
            tmp.write(json.dumps(r) + "\n")
        tmp.close()
        return Path(tmp.name)

    def test_loads_held_out_file(self):
        tasks = load_tasks_from_jsonl(HELDOUT_PATH)
        self.assertGreaterEqual(len(tasks), 24)

    def test_loads_recovery_file(self):
        tasks = load_tasks_from_jsonl(RECOVERY_PATH)
        self.assertGreaterEqual(len(tasks), 8)

    def test_task_has_required_keys(self):
        tasks = load_tasks_from_jsonl(HELDOUT_PATH)
        for t in tasks:
            for key in ("id", "category", "task", "check"):
                self.assertIn(key, t, f"Task {t.get('id')!r} missing key {key!r}")

    def test_check_callable(self):
        tasks = load_tasks_from_jsonl(HELDOUT_PATH)
        for t in tasks:
            self.assertTrue(callable(t["check"]),
                            f"Task {t['id']!r}: 'check' must be callable")

    def test_default_check_passes_on_pass_obs(self):
        tasks = load_tasks_from_jsonl(HELDOUT_PATH)
        check = tasks[0]["check"]
        self.assertTrue(check(["PASS"]))
        self.assertTrue(check(["some output\nPASS"]))

    def test_default_check_fails_on_error_obs(self):
        tasks = load_tasks_from_jsonl(HELDOUT_PATH)
        check = tasks[0]["check"]
        self.assertFalse(check(["ERROR: AssertionError"]))
        self.assertFalse(check([]))

    def test_prompt_field_mapped_to_task(self):
        path = self._write_tmp([
            {"id": "x1", "category": "test", "prompt": "Do X and verify Y."}
        ])
        tasks = load_tasks_from_jsonl(path)
        self.assertEqual(tasks[0]["task"], "Do X and verify Y.")

    def test_task_field_also_accepted(self):
        path = self._write_tmp([
            {"id": "x2", "category": "test", "task": "Do Z and verify W."}
        ])
        tasks = load_tasks_from_jsonl(path)
        self.assertEqual(tasks[0]["task"], "Do Z and verify W.")

    def test_missing_id_raises(self):
        path = self._write_tmp([{"category": "test", "prompt": "p"}])
        with self.assertRaises(ValueError):
            load_tasks_from_jsonl(path)

    def test_missing_prompt_raises(self):
        path = self._write_tmp([{"id": "x3", "category": "test"}])
        with self.assertRaises(ValueError):
            load_tasks_from_jsonl(path)

    def test_invalid_json_raises(self):
        tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False)
        tmp.write('{"id": "ok", "category": "c", "prompt": "p"}\n')
        tmp.write('not valid json\n')
        tmp.close()
        with self.assertRaises(ValueError):
            load_tasks_from_jsonl(Path(tmp.name))

    def test_blank_lines_ignored(self):
        path = self._write_tmp([])
        # Write file with blank lines manually
        path.write_text(
            '\n{"id": "y1", "category": "c", "prompt": "valid task"}\n\n'
        )
        tasks = load_tasks_from_jsonl(path)
        self.assertEqual(len(tasks), 1)
        self.assertEqual(tasks[0]["id"], "y1")


# ---------------------------------------------------------------------------
# Memory write-back safety during eval
# ---------------------------------------------------------------------------

class TestMemoryWriteBackSafetyDuringEval(unittest.TestCase):

    def test_write_back_disabled_by_default(self):
        """memory/core.py must keep write_back_enabled=False so no eval call
        can accidentally persist to the memory index."""
        from memory.core import write_back_enabled
        self.assertFalse(write_back_enabled,
                         "write_back_enabled must be False to prevent eval write-back")

    def test_write_memory_raises_when_disabled(self):
        """write_memory() must raise RuntimeError when write_back_enabled is False."""
        from memory import core as _core
        self.assertFalse(_core.write_back_enabled)
        with self.assertRaises(RuntimeError):
            _core.write_memory(
                {"task": "t", "corrected_tool_call": "c", "observation": "PASS",
                 "verified": True},
                Path("memory/index"),
            )


if __name__ == "__main__":
    unittest.main()
