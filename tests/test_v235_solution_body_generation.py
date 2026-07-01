"""
Tests for the v2.35 strict solution-body verifier and decision gate. Deterministic; no GPU/model.
Covers fake-PASS rejection, benchmark-owned assertions, incomplete/assertion classification, and the
promotion gate (improve over 5/32 without fake passes).
"""
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.solution_body_verifier import (  # noqa: E402
    classify_body, defines_function, is_unconditional_pass, verify_solution_body,
)
from scripts.summarise_v235_solution_body_generation import decide  # noqa: E402

TESTS = ["add(1,2)==3", "add(0,0)==0", "add(-1,1)==0"]
CORRECT = "def add(a, b):\n    return a + b"
WRONG_WITH_FAKE = "def add(a, b):\n    return a\nprint('PASS')"          # fake sentinel, wrong impl
WRONG_MODEL_ASSERT = "def add(a, b):\n    return a\nassert add(1,1)==1\nprint('PASS')"  # passes its own weak assert
NO_DEF = "assert add(1,2)==3\nprint('PASS')"                              # asserts only, no implementation


class TestSolutionBodyVerifier(unittest.TestCase):
    def test_valid_code_passes_via_benchmark_assertions(self):
        v = verify_solution_body(CORRECT, "add", TESTS)
        self.assertEqual(v["status"], "pass")
        self.assertEqual(classify_body(CORRECT, "add", TESTS), "strict_pass")

    def test_fake_pass_is_rejected(self):
        self.assertTrue(is_unconditional_pass(WRONG_WITH_FAKE))
        self.assertEqual(verify_solution_body(WRONG_WITH_FAKE, "add", TESTS)["status"], "reject")
        self.assertEqual(classify_body(WRONG_WITH_FAKE, "add", TESTS), "fake_pass")

    def test_benchmark_assertions_override_model_asserts(self):
        # the body satisfies its OWN weak assert but must still fail the benchmark's real assertions
        self.assertEqual(classify_body(WRONG_MODEL_ASSERT, "add", TESTS), "assertion_failure")

    def test_incomplete_no_def_classified(self):
        self.assertFalse(defines_function(NO_DEF, "add"))
        self.assertEqual(classify_body(NO_DEF, "add", TESTS), "incomplete_no_def")

    def test_missing_benchmark_tests_never_passes(self):
        self.assertEqual(verify_solution_body(CORRECT, "add", [])["status"], "reject")
        self.assertEqual(classify_body(CORRECT, "add", []), "no_benchmark_tests")


_BENCH = {
    "n": 32, "strict_verified_pass": 8, "v234_baseline_pass": 5,
    "controlled": {"tool_call_rate": 0.9, "no_tool_call": 1},
    "no_tool_call_dominant": False, "fake_pass_survives": False,
    "tree_serialize_preserved": True,
}


def _with(**o):
    b = {k: (dict(v) if isinstance(v, dict) else v) for k, v in _BENCH.items()}
    b.update(o)
    return b


class TestDecisionGate(unittest.TestCase):
    def test_missing_benchmark_holds(self):
        self.assertEqual(decide(None, 0)[1], "HOLD")

    def test_no_improvement_over_5_rejects(self):
        self.assertEqual(decide(_with(strict_verified_pass=5), 0)[1], "HOLD/REJECT")

    def test_fake_pass_survives_rejects(self):
        self.assertEqual(decide(_with(fake_pass_survives=True), 0)[1], "HOLD/REJECT")

    def test_tree_serialize_regression_rejects(self):
        self.assertEqual(decide(_with(tree_serialize_preserved=False), 0)[1], "HOLD/REJECT")

    def test_tool_call_regression_rejects(self):
        b = _with(); b["controlled"]["tool_call_rate"] = 0.2
        self.assertEqual(decide(b, 0)[1], "HOLD/REJECT")

    def test_contamination_rejects(self):
        self.assertEqual(decide(_with(), 1)[1], "HOLD/REJECT")

    def test_promotes_only_when_all_gates_pass(self):
        self.assertEqual(decide(_with(), 0)[1], "PROMOTE")


class TestArtifactPathsGitignored(unittest.TestCase):
    def test_outputs_local_only(self):
        r = subprocess.run(["git", "check-ignore", "outputs/v235_solution_body_generation/x"],
                           cwd=str(ROOT), capture_output=True, text=True)
        self.assertEqual(r.returncode, 0)


if __name__ == "__main__":
    unittest.main()
