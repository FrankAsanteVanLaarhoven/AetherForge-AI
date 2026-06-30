"""
Tests for the v2.31 tiny repair-trace SFT pilot: dataset export format + contamination safety, and
import-safety / GPU-gating of the trainer and eval harness. No GPU, model, or network required.
"""
import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.build_v231_repair_sft_dataset import _format_example  # noqa: E402

AGG = ROOT / "data" / "generated" / "v231" / "sft_aggregate.json"
TRAIN_JSONL = ROOT / "data" / "generated" / "v231" / "sft_train.jsonl"


class TestFormatExample(unittest.TestCase):
    def test_input_output_contract(self):
        rec = {
            "logical_task": "csv_join", "representation": "list_format",
            "candidate_solution": "def csv_join(xs):\n    return \";\".join(map(str, xs))",
            "verifier_signal": {"failure_type": "separator_error", "expected": "1,2",
                                "observed": "1;2", "diagnosis": "wrong separator",
                                "repair_hint": "use comma"},
            "repair_plan": "switch separator to comma",
            "final_solution": "def csv_join(xs):\n    return \",\".join(map(str, xs))",
        }
        inp, out = _format_example(rec)
        self.assertTrue(inp.startswith("INPUT:"))
        for token in ("### Task", "### Failed candidate", "### Verifier signal",
                      "failure_type: separator_error", "repair_hint: use comma"):
            self.assertIn(token, inp)
        self.assertTrue(out.startswith("OUTPUT:"))
        self.assertIn("### Repair plan", out)
        self.assertIn("### Corrected solution", out)


class TestGatedScriptsImportSafe(unittest.TestCase):
    def test_trainer_and_eval_import_and_have_gates(self):
        import scripts.train_v231_repair_sft as tr
        import scripts.eval_v231_repair_sft as ev
        self.assertTrue(hasattr(tr, "_precheck"))
        self.assertTrue(hasattr(ev, "_gpu_ok"))
        # the eval slice helper must reject solutions lacking the PASS sentinel
        self.assertFalse(ev._passes("x = 1"))


@unittest.skipUnless(AGG.exists(), "v2.31 SFT export not built (gitignored output absent)")
class TestExportAggregate(unittest.TestCase):
    def setUp(self):
        self.a = json.loads(AGG.read_text())

    def test_split_sums(self):
        self.assertEqual(self.a["train_records"] + self.a["val_records"], self.a["total_available"])

    def test_no_contamination(self):
        self.assertEqual(self.a["contamination_guard_violations"], 0)

    def test_both_categories_present(self):
        self.assertGreater(self.a["format_repair"], 0)
        self.assertGreater(self.a["algorithmic_repair"], 0)


@unittest.skipUnless(TRAIN_JSONL.exists(), "v2.31 train split absent")
class TestTrainRecords(unittest.TestCase):
    def test_records_have_input_output_and_no_benchmark_names(self):
        from scripts.build_v227_trace_factory import _load_overlap_corpus
        bench = _load_overlap_corpus()[0]
        rows = [json.loads(l) for l in open(TRAIN_JSONL)]
        self.assertGreater(len(rows), 0)
        for r in rows:
            self.assertIn("### Failed candidate", r["input"])
            self.assertIn("### Corrected solution", r["output"])
            self.assertNotIn(r["task_id"], bench)


if __name__ == "__main__":
    unittest.main()
