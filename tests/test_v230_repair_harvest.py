"""
Tests for the v2.30 broadened repair harvest (scripts/build_v230_broadened_repair_harvest.py) and
its v2.28-builder integration. Pure structural tests run standalone; record/aggregate-invariant
tests run only when the local harvest/dataset has been built (gitignored output).
"""
import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.build_v230_broadened_repair_harvest import (  # noqa: E402
    ALGO_MUTATORS, ALGO_TASKS, FORMAT_MUTATORS, FORMAT_TASKS, FORMAT_TYPES,
)

TRACES = ROOT / "data" / "generated" / "v230" / "repair_traces.jsonl"
HARVEST = ROOT / "data" / "generated" / "v230" / "harvest_aggregate.json"
DATASET = ROOT / "data" / "generated" / "v228" / "dataset_aggregate.json"


class TestTaskLibrary(unittest.TestCase):
    def test_task_names_disjoint_from_benchmark(self):
        from scripts.build_v227_trace_factory import _load_overlap_corpus
        bench = _load_overlap_corpus()[0]
        for func, *_ in FORMAT_TASKS + ALGO_TASKS:
            self.assertNotIn(func, bench, msg=f"{func} collides with a benchmark callable")

    def test_algo_mutators_are_algorithmic(self):
        for _, _, ft in ALGO_MUTATORS:
            self.assertEqual(ft, "algorithmic_error")

    def test_format_mutators_are_format_family(self):
        for _, _, ft in FORMAT_MUTATORS:
            self.assertIn(ft, FORMAT_TYPES)


@unittest.skipUnless(TRACES.exists(), "v2.30 harvest not built (gitignored output absent)")
class TestHarvestedRecords(unittest.TestCase):
    def setUp(self):
        self.recs = [json.loads(l) for l in open(TRACES)]

    def test_all_genuine_and_verified(self):
        for r in self.recs:
            self.assertNotEqual(r["candidate_solution"].strip(), r["final_solution"].strip())
            self.assertEqual(r["candidate_status"], "fail")
            self.assertEqual(r["final_status"], "pass")
            self.assertTrue(r["repair_successful"])

    def test_categories_and_guard(self):
        for r in self.recs:
            self.assertIn(r["repair_category"],
                          {"format_repair", "algorithmic_repair", "mixed_repair"})
            self.assertEqual(sum(r["contamination_guard"].values()), 0)
            self.assertTrue(r["repair_plan"])

    def test_both_categories_present(self):
        cats = {r["repair_category"] for r in self.recs}
        self.assertIn("format_repair", cats)
        self.assertIn("algorithmic_repair", cats)


@unittest.skipUnless(HARVEST.exists(), "v2.30 harvest aggregate absent")
class TestHarvestAggregate(unittest.TestCase):
    def test_invariants(self):
        a = json.loads(HARVEST.read_text())
        self.assertGreater(a["successful_repairs"], 0)
        self.assertTrue(a["all_genuine_transitions"])
        self.assertEqual(a["contamination_guard_violations"], 0)
        self.assertGreater(a["algorithmic_repair"], 0)
        self.assertGreater(a["format_repair"], 0)


@unittest.skipUnless(DATASET.exists(), "v2.28 dataset aggregate absent")
class TestDatasetIntegration(unittest.TestCase):
    def test_dataset_has_both_repair_categories(self):
        a = json.loads(DATASET.read_text())
        uc = a["use_tag_counts"]
        self.assertIn("algorithmic_repair_candidate", uc)
        self.assertGreater(uc["format_repair_candidate"], 0)
        self.assertGreater(uc["algorithmic_repair_candidate"], 0)
        self.assertEqual(a["contamination_guard_violations"], 0)


if __name__ == "__main__":
    unittest.main()
