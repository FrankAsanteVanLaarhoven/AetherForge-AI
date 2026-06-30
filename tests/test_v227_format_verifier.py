"""
Tests for the v2.27 format-control layer: the deterministic tree-serialization format verifier
(scripts/v227_format_verifier.py) and the format-control evaluation invariants
(scripts/v227_format_control_eval.py). All deterministic; no model, no network.
"""
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.v227_format_verifier import (  # noqa: E402
    FAILURE_TYPES, LOGICAL_TASKS, REPRESENTATIONS,
    classify_failure, format_verify, render,
)
from scripts.v227_format_control_eval import _battery, _inject, canonical_control  # noqa: E402


class TestRenderers(unittest.TestCase):
    def test_all_logical_x_representation_renderers_exist(self):
        for lt in LOGICAL_TASKS:
            for rep in REPRESENTATIONS:
                self.assertIsNotNone(render((1, (2, 3)), lt, rep))

    def test_reference_serializations(self):
        t = (1, (2, 3))
        self.assertEqual(render(t, "full_structure", "exact_string"), "(1 (2 3))")
        self.assertEqual(render(t, "full_structure", "token_list"),
                         ["(", "1", "(", "2", "3", ")", ")"])
        self.assertEqual(render(t, "full_structure", "nested_list"), [1, [2, 3]])
        self.assertEqual(render(t, "full_structure", "json"),
                         {"branch": [{"leaf": 1}, {"branch": [{"leaf": 2}, {"leaf": 3}]}]})
        self.assertEqual(render(t, "leaf_values", "exact_string"), "1,2,3")
        self.assertEqual(render(t, "leaf_depth", "exact_string"), "1:1,2:2,3:2")

    def test_unknown_renderer_raises(self):
        with self.assertRaises(ValueError):
            render((1, 2), "full_structure", "nonexistent_format")


class TestClassifier(unittest.TestCase):
    def test_pass_when_equal(self):
        self.assertIsNone(classify_failure("(1 (2 3))", "(1 (2 3))", "exact_string"))
        block = format_verify("(1 (2 3))", "(1 (2 3))", "exact_string", "full_structure")
        self.assertEqual(block["status"], "pass")
        self.assertIsNone(block["failure_type"])

    def test_failure_type_labels(self):
        cases = {
            "missing_null_marker": ("(1 2 3)", "(1 (2 3))"),
            "extra_null_marker": ("(1 (2 3)))", "(1 (2 3))"),
            "separator_error": ("(1 (2,3))", "(1 (2 3))"),
            "type_error": ([1, [2, 3]], "(1 (2 3))"),
        }
        for expected_label, (obs, exp) in cases.items():
            self.assertEqual(classify_failure(obs, exp, "exact_string"), expected_label,
                             msg=f"{obs!r} vs {exp!r}")

    def test_ordering_vs_algorithmic_on_lists(self):
        self.assertEqual(classify_failure([3, 2, 1], [1, 2, 3], "nested_list"), "ordering_error")
        self.assertEqual(classify_failure([1, 2, 9], [1, 2, 3], "nested_list"), "algorithmic_error")

    def test_all_labels_are_known(self):
        block = format_verify("(1 2 3)", "(1 (2 3))", "exact_string", "full_structure")
        self.assertIn(block["failure_type"], FAILURE_TYPES)
        self.assertTrue(block["repair_hint"])
        # structured block keys are stable and stderr-independent
        self.assertEqual(set(block), {"status", "failure_type", "expected_format",
                                      "observed_format", "diagnosis", "repair_hint"})


class TestFormatControlInvariants(unittest.TestCase):
    def test_canonical_control_is_total(self):
        rates = canonical_control(_battery(12))
        for (lt, rep), (ok, n) in rates.items():
            self.assertEqual(ok, n, msg=f"canonical control not 100% for {lt}/{rep}")

    def test_injected_faults_are_correctly_classified(self):
        battery = _battery(20)
        for fault in FAILURE_TYPES:
            matched = 0
            seen = 0
            for tree in battery:
                lt, rep, broken = _inject(fault, tree)
                expected = render(tree, lt, rep)
                if broken == expected:
                    continue
                seen += 1
                if classify_failure(broken, expected, rep) == fault:
                    matched += 1
            self.assertGreater(seen, 0, msg=f"no injectable case for {fault}")
            self.assertEqual(matched, seen, msg=f"misclassified {fault}: {matched}/{seen}")


if __name__ == "__main__":
    unittest.main()
