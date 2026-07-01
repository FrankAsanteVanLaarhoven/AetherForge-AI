"""
Tests for the v2.34 tool-call format controller and decision gate. All deterministic; no GPU, model,
or network. Covers detection (valid / no_tool_call / invalid JSON), repair (unambiguous wrap vs
ambiguous/unsafe rejection), the decision gate, and local-only artifact paths.
"""
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.tool_call_format_control import (  # noqa: E402
    detect_tool_call, has_invalid_tool_json, is_no_tool_call, repair_to_execute_code,
)
from scripts.summarise_v234_tool_call_format_control import decide  # noqa: E402

VALID = 'execute_code({"code": "def f():\\n    return 1\\nassert f()==1\\nprint(\'PASS\')"})'
INVALID_JSON = 'execute_code({ "code": """def f():\n    return 1""" })'
NO_TOOL = "PLAN: solve it\nTOOL_CALL: \nassert f()==1\nprint('PASS')\n"
NEAR_MISS = "PLAN: x\nTOOL_CALL: \ndef f():\n    return 1\nassert f()==1\nprint('PASS')\n"
AMBIGUOUS = ("TOOL_CALL: \ndef f():\n    return 1\nassert f()==1\n"
             "\n```python\ndef g():\n    return 2\nassert g()==2\n```")
NOT_CODE = "PLAN: I will think about it.\nTOOL_CALL: \njust some prose, no code here.\n"


class TestDetection(unittest.TestCase):
    def test_valid_call_detected(self):
        d = detect_tool_call(VALID)
        self.assertTrue(d["has_tool_call"] and d["valid_json"])
        self.assertFalse(is_no_tool_call(VALID))

    def test_no_tool_call_detected(self):
        self.assertTrue(is_no_tool_call(NO_TOOL))

    def test_invalid_json_detected(self):
        self.assertTrue(has_invalid_tool_json(INVALID_JSON))
        self.assertFalse(has_invalid_tool_json(VALID))


class TestRepair(unittest.TestCase):
    def test_valid_call_passes_unchanged(self):
        r = repair_to_execute_code(VALID)
        self.assertEqual(r["status"], "ok")
        self.assertEqual(r["action"], "passthrough")
        self.assertIn("def f():", r["code"])

    def test_near_miss_wrapped_when_unambiguous(self):
        r = repair_to_execute_code(NEAR_MISS)
        self.assertEqual(r["status"], "ok")
        self.assertEqual(r["action"], "wrapped")
        self.assertTrue(r["call"].startswith("execute_code("))
        self.assertIn("def f()", r["code"])

    def test_invalid_json_recovered(self):
        r = repair_to_execute_code(INVALID_JSON)
        self.assertEqual(r["status"], "ok")
        self.assertIn("def f()", r["code"])

    def test_ambiguous_rejected(self):
        r = repair_to_execute_code(AMBIGUOUS)
        self.assertEqual(r["status"], "rejected")
        self.assertEqual(r["reason"], "ambiguous")

    def test_no_code_rejected(self):
        self.assertEqual(repair_to_execute_code("nothing here")["reason"], "no_code")

    def test_non_code_rejected(self):
        r = repair_to_execute_code(NOT_CODE)
        self.assertEqual(r["status"], "rejected")
        self.assertIn(r["reason"], ("not_code", "no_code"))


_B = {
    "no_tool_call_dominant_controlled": False,
    "baseline": {"tool_call_rate": 0.19, "no_tool_call": 26, "pass": 5},
    "controlled": {"tool_call_rate": 0.91, "no_tool_call": 3, "pass": 8},
    "tree_serialize_preserved": True, "recovered_passes": 3, "recovered_tool_calls": 23,
}


def _with(**over):
    b = {k: (dict(v) if isinstance(v, dict) else v) for k, v in _B.items()}
    for k, v in over.items():
        b[k] = v
    return b


class TestDecisionGate(unittest.TestCase):
    def test_missing_benchmark_holds(self):
        self.assertEqual(decide(None, 0)[1], "HOLD")

    def test_no_tool_call_dominant_rejects(self):
        self.assertEqual(decide(_with(no_tool_call_dominant_controlled=True), 0)[1], "HOLD/REJECT")

    def test_tool_call_rate_not_improving_rejects(self):
        b = _with(); b["controlled"]["tool_call_rate"] = 0.20  # no material gain over 0.19
        self.assertEqual(decide(b, 0)[1], "HOLD/REJECT")

    def test_score_not_improving_over_5_rejects(self):
        b = _with(); b["controlled"]["pass"] = 5
        self.assertEqual(decide(b, 0)[1], "HOLD/REJECT")

    def test_tree_serialize_regression_rejects(self):
        self.assertEqual(decide(_with(tree_serialize_preserved=False), 0)[1], "HOLD/REJECT")

    def test_contamination_rejects(self):
        self.assertEqual(decide(_with(), 1)[1], "HOLD/REJECT")

    def test_promotes_only_when_all_gates_pass(self):
        self.assertEqual(decide(_with(), 0)[1], "PROMOTE")


class TestArtifactPathsGitignored(unittest.TestCase):
    def test_outputs_local_only(self):
        r = subprocess.run(["git", "check-ignore", "outputs/v234_tool_call_format_control/x"],
                           cwd=str(ROOT), capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, msg="v2.34 outputs must be gitignored (local-only)")


if __name__ == "__main__":
    unittest.main()
