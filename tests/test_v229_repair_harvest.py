"""
Tests for the v2.29 genuine repair trace harvest (scripts/build_v229_repair_harvest.py).
Pure mutator tests run standalone; record/aggregate-invariant tests run only when the local harvest
has been built (gitignored output).
"""
import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.build_v229_repair_harvest import (  # noqa: E402
    _mut_bracket, _mut_extra_marker, _mut_missing_marker, _mut_separator,
)

FORMAT_TYPES = {"missing_null_marker", "extra_null_marker", "separator_error",
                "ordering_error", "type_error", "format_error"}
REF = ('def f(node):\n    if isinstance(node, tuple):\n'
       '        return "(" + f(node[0]) + " " + f(node[1]) + ")"\n    return str(node)')

TRACES = ROOT / "data" / "generated" / "v229" / "repair_traces.jsonl"
AGG = ROOT / "data" / "generated" / "v229" / "harvest_aggregate.json"


class TestMutators(unittest.TestCase):
    def test_format_mutations_change_code(self):
        # Each mutator returns (perturbation_name, mutated_code) and changes only the format.
        for mut in (_mut_separator, _mut_bracket, _mut_missing_marker, _mut_extra_marker):
            res = mut(REF)
            self.assertIsNotNone(res, msg=f"{mut.__name__} not applicable to reference")
            fault, mutated = res
            self.assertIsInstance(fault, str)
            self.assertNotEqual(mutated.strip(), REF.strip(), msg=f"{mut.__name__} did not change code")

    def test_bracket_mutation_swaps_delimiters(self):
        _, mutated = _mut_bracket(REF)
        self.assertIn('"["', mutated)
        self.assertNotIn('"("', mutated)


@unittest.skipUnless(TRACES.exists(), "v2.29 harvest not built (gitignored output absent)")
class TestHarvestedRecords(unittest.TestCase):
    def setUp(self):
        self.recs = [json.loads(l) for l in open(TRACES)]

    def test_nonempty(self):
        self.assertGreater(len(self.recs), 0)

    def test_all_genuine_transitions(self):
        for r in self.recs:
            self.assertNotEqual(r["candidate_solution"].strip(), r["final_solution"].strip())
            self.assertTrue(r["candidate_differs_from_final"])
            self.assertTrue(r["repair_successful"])

    def test_candidate_fail_final_pass(self):
        for r in self.recs:
            self.assertEqual(r["candidate_status"], "fail")
            self.assertEqual(r["final_status"], "pass")

    def test_verifier_failure_type_is_format_family(self):
        for r in self.recs:
            self.assertIn(r["verifier_signal"]["failure_type"], FORMAT_TYPES)
            self.assertTrue(r["repair_plan"])

    def test_contamination_guard_zero(self):
        for r in self.recs:
            self.assertEqual(sum(r["contamination_guard"].values()), 0)


@unittest.skipUnless(AGG.exists(), "v2.29 harvest aggregate absent")
class TestHarvestAggregate(unittest.TestCase):
    def test_aggregate_invariants(self):
        a = json.loads(AGG.read_text())
        self.assertGreater(a["successful_repairs"], 0)
        self.assertTrue(a["all_genuine_transitions"])
        self.assertEqual(a["contamination_guard_violations"], 0)
        self.assertTrue(set(a["failure_type_distribution"]).issubset(FORMAT_TYPES))


if __name__ == "__main__":
    unittest.main()
