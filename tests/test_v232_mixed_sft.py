"""
Tests for the v2.32 mixed repair + tool-use-preservation pilot: dataset build (preservation scaffold
+ contamination), and the promotion gate (repair must improve without dropping the 32-task benchmark;
tool-use preserved). No GPU/model/network required.
"""
import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.build_v232_mixed_dataset import _preservation_example  # noqa: E402
from scripts.summarise_v232_mixed_sft import decide  # noqa: E402

_TRAIN = {"loss_trend": [1.7, 0.5]}
_EV_OK = {"base_repair_pass": 1, "adapter_repair_pass": 4, "val_preservation": 8,
          "adapter_preservation_pass": 8}
_EV_NO_IMPROVE = {"base_repair_pass": 4, "adapter_repair_pass": 4, "val_preservation": 8,
                  "adapter_preservation_pass": 8}
_EV_LOW_PRES = {"base_repair_pass": 1, "adapter_repair_pass": 4, "val_preservation": 8,
                "adapter_preservation_pass": 2}
_BENCH_OK = {"adapter_32_pass": 23, "tree_serialize_3of3_preserved": True}
_BENCH_REGRESS = {"adapter_32_pass": 19, "tree_serialize_3of3_preserved": True}


class TestPreservationExample(unittest.TestCase):
    def test_scaffold_has_tool_call_and_pass(self):
        inp, out = _preservation_example(
            "list_total", "arithmetic",
            "def list_total(xs):\n    return sum(xs)",
            ["list_total([1,2]) == 3", "list_total([]) == 0"])
        self.assertTrue(inp.startswith("INPUT:"))
        for tok in ("PLAN:", "TOOL_CALL: execute_code(", "VERIFIER:", "status: PASS", "FINAL_ANSWER:"):
            self.assertIn(tok, out)


class TestPromotionGate(unittest.TestCase):
    def test_not_run_holds(self):
        self.assertEqual(decide(None, None, None, 0)[1], "HOLD")

    def test_repair_improved_no_bench_is_partial(self):
        self.assertEqual(decide(_TRAIN, _EV_OK, None, 0)[1], "PARTIAL/HOLD-PENDING-BENCHMARKS")

    def test_all_gates_promote(self):
        self.assertEqual(decide(_TRAIN, _EV_OK, _BENCH_OK, 0)[1], "PROMOTE")

    def test_benchmark_regression_holds(self):
        self.assertEqual(decide(_TRAIN, _EV_OK, _BENCH_REGRESS, 0)[1], "HOLD")

    def test_no_repair_improvement_holds(self):
        # repair must IMPROVE; equal pass rate does not promote even with a clean benchmark
        self.assertEqual(decide(_TRAIN, _EV_NO_IMPROVE, _BENCH_OK, 0)[1], "HOLD")

    def test_low_preservation_holds(self):
        self.assertEqual(decide(_TRAIN, _EV_LOW_PRES, _BENCH_OK, 0)[1], "HOLD")

    def test_contamination_blocks(self):
        self.assertEqual(decide(_TRAIN, _EV_OK, _BENCH_OK, 1)[1], "HOLD")


@unittest.skipUnless((ROOT / "data/generated/v232/mixed_aggregate.json").exists(),
                     "v2.32 mixed dataset not built")
class TestMixedAggregate(unittest.TestCase):
    def test_invariants(self):
        a = json.loads((ROOT / "data/generated/v232/mixed_aggregate.json").read_text())
        self.assertEqual(a["repair_examples"] + a["tool_use_preservation_examples"], a["total"])
        self.assertGreater(a["tool_use_preservation_examples"], 0)
        self.assertEqual(a["contamination_guard_violations"], 0)


if __name__ == "__main__":
    unittest.main()
