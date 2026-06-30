"""
Tests for the v2.33 scaffold-first tool-call preservation pilot: scaffold-only dataset (no repair),
contamination guard, decision gate (preservation + benchmark; repair is not a gate; no_tool_call
rejection), CPU/no-CUDA skip, and local-only artifact paths.
"""
import json
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.build_v227_trace_factory import _contamination_guard, _load_overlap_corpus  # noqa: E402
from scripts.summarise_v233_scaffold_sft import decide  # noqa: E402
import scripts.train_v233_scaffold_sft as tr  # noqa: E402
import scripts.eval_v233_scaffold_sft as ev_mod  # noqa: E402

_TRAIN = {"loss_trend": [1.6, 0.5]}
_EV_OK = {"val_records": 5, "preservation_pass": 5}
_EV_LOW = {"val_records": 5, "preservation_pass": 1}
_BENCH_OK = {"adapter_32_pass": 23, "tree_serialize_3of3_preserved": True, "no_tool_call_dominant": False}
_BENCH_REGRESS = {"adapter_32_pass": 19, "tree_serialize_3of3_preserved": True, "no_tool_call_dominant": False}
_BENCH_NOTOOL = {"adapter_32_pass": 23, "tree_serialize_3of3_preserved": True, "no_tool_call_dominant": True}

AGG = ROOT / "data" / "generated" / "v233" / "scaffold_aggregate.json"
TRAIN_JSONL = ROOT / "data" / "generated" / "v233" / "scaffold_train.jsonl"


class TestContaminationGuard(unittest.TestCase):
    def test_guard_rejects_heldout_function_name(self):
        corpus = _load_overlap_corpus()
        bench = corpus[0]
        heldout = next(iter(bench))  # a real 32-task benchmark callable name
        cg = _contamination_guard(heldout, heldout, "def x(): pass", "def x(): pass", {}, corpus)
        self.assertTrue(any(cg.values()), msg=f"guard should flag held-out name {heldout}")


class TestDecisionGate(unittest.TestCase):
    def test_not_run_holds(self):
        self.assertEqual(decide(None, None, None, 0)[1], "HOLD")

    def test_preservation_no_bench_is_partial(self):
        self.assertEqual(decide(_TRAIN, _EV_OK, None, 0)[1], "PARTIAL/HOLD-PENDING-BENCHMARKS")

    def test_promotes_only_when_all_gates_pass(self):
        self.assertEqual(decide(_TRAIN, _EV_OK, _BENCH_OK, 0)[1], "PROMOTE")

    def test_low_preservation_holds(self):
        self.assertEqual(decide(_TRAIN, _EV_LOW, _BENCH_OK, 0)[1], "HOLD/REJECT")

    def test_benchmark_regression_holds(self):
        self.assertEqual(decide(_TRAIN, _EV_OK, _BENCH_REGRESS, 0)[1], "HOLD/REJECT")

    def test_no_tool_call_dominant_rejects(self):
        self.assertEqual(decide(_TRAIN, _EV_OK, _BENCH_NOTOOL, 0)[1], "HOLD/REJECT")

    def test_contamination_blocks(self):
        self.assertEqual(decide(_TRAIN, _EV_OK, _BENCH_OK, 1)[1], "HOLD/REJECT")


class TestCpuSkip(unittest.TestCase):
    def test_precheck_skips_without_cuda(self):
        # without CUDA the trainer must refuse cleanly (no fabricated metrics); with CUDA, skip.
        import torch
        if torch.cuda.is_available():
            self.skipTest("CUDA present; CPU-skip path not exercised here")
        self.assertIsNone(tr._precheck())
        self.assertFalse(ev_mod._gpu_ok())


class TestArtifactPathsGitignored(unittest.TestCase):
    def test_generated_paths_local_only(self):
        for p in ("data/generated/v233/scaffold_train.jsonl", "outputs/v233_scaffold_first_sft/adapter"):
            r = subprocess.run(["git", "check-ignore", p], cwd=str(ROOT),
                               capture_output=True, text=True)
            self.assertEqual(r.returncode, 0, msg=f"{p} must be gitignored (local-only)")


@unittest.skipUnless(AGG.exists(), "v2.33 scaffold dataset not built")
class TestScaffoldDataset(unittest.TestCase):
    def test_no_repair_objective(self):
        a = json.loads(AGG.read_text())
        self.assertEqual(a["repair_examples"], 0)
        self.assertEqual(set(a["objective_distribution"]), {"tool_use_preservation"})
        self.assertEqual(a["contamination_guard_violations"], 0)

    def test_train_records_have_scaffold_and_no_benchmark_names(self):
        bench = _load_overlap_corpus()[0]
        rows = [json.loads(l) for l in open(TRAIN_JSONL)]
        self.assertGreater(len(rows), 0)
        for r in rows:
            self.assertEqual(r["objective"], "tool_use_preservation")
            self.assertIn("execute_code", r["output"])
            self.assertNotIn(r.get("func", ""), bench)


if __name__ == "__main__":
    unittest.main()
