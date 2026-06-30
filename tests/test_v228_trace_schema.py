"""
Tests for the v2.28 self-improving trace dataset schema + filters
(scripts/build_v228_self_improving_dataset.py). Pure-function tests run standalone; the
aggregate-invariant tests run only when the local dataset has been built (gitignored output).
"""
import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.build_v228_self_improving_dataset import (  # noqa: E402
    CANONICAL_FIELDS, FORMAT_FAILURE_TYPES, _capability_tag, _quality, _short,
)

REQUIRED_SCHEMA_KEYS = {
    "record_id", "task_id", "task_family", "capability_tag", "representation", "model_config",
    "prompt_mode", "retrieved_memory", "plan", "candidate_solution", "verifier_signal",
    "repair_plan", "repaired_solution", "final_solution", "final_status", "quality",
    "contamination_guard", "split", "rejection_reason",
}
AGG = ROOT / "data" / "generated" / "v228" / "dataset_aggregate.json"


class TestSchema(unittest.TestCase):
    def test_canonical_fields_cover_required_schema(self):
        missing = REQUIRED_SCHEMA_KEYS - set(CANONICAL_FIELDS)
        self.assertEqual(missing, set(), msg=f"schema missing {missing}")

    def test_capability_tag_maps_tree_serialize_to_format_control(self):
        self.assertEqual(_capability_tag("tree_serialize_repr", "exact_string"), "format_control")
        self.assertEqual(_capability_tag("", ""), "unknown")

    def test_short_truncates(self):
        self.assertTrue(len(_short("x" * 500)) <= 120)
        self.assertEqual(_short("(1 (2 3))"), "(1 (2 3))")
        self.assertEqual(_short([1, [2, 3]]), "[1, [2, 3]]")


class TestQualityScoring(unittest.TestCase):
    def test_quality_score_bounds_and_flags(self):
        q = _quality(True, True, True, True, True, True, True)
        self.assertEqual(q["quality_score"], 1.0)
        for flag in ("has_plan", "has_verifier_signal", "has_repair",
                     "candidate_differs_from_final", "repair_successful", "format_control_used"):
            self.assertIn(flag, q)
        z = _quality(False, False, False, False, False, False, False)
        self.assertEqual(z["quality_score"], 0.0)
        self.assertTrue(0.0 <= _quality(True, True, False, False, False, True, True)["quality_score"] <= 1.0)

    def test_format_failure_types_are_a_subset_of_verifier_types(self):
        self.assertTrue(FORMAT_FAILURE_TYPES.issubset({
            "algorithmic_error", "format_error", "missing_null_marker", "extra_null_marker",
            "separator_error", "ordering_error", "type_error"}))


@unittest.skipUnless(AGG.exists(), "v2.28 dataset not built (gitignored output absent)")
class TestAggregateInvariants(unittest.TestCase):
    def setUp(self):
        self.a = json.loads(AGG.read_text())

    def test_accepted_plus_rejected_equals_scanned(self):
        self.assertEqual(self.a["accepted"] + self.a["rejected"], self.a["total_scanned"])

    def test_no_contamination_violations(self):
        self.assertEqual(self.a["contamination_guard_violations"], 0)

    def test_representation_distribution_sums_to_accepted(self):
        self.assertEqual(sum(self.a["representation_distribution"].values()), self.a["accepted"])

    def test_use_tag_keys_present(self):
        for k in ("sft_candidate", "preference_pair_candidate",
                  "format_repair_candidate", "verifier_format_candidate"):
            self.assertIn(k, self.a["use_tag_counts"])

    def test_schema_fields_recorded(self):
        self.assertEqual(self.a["schema_fields"], CANONICAL_FIELDS)


if __name__ == "__main__":
    unittest.main()
