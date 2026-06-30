"""
scripts/build_v228_self_improving_dataset.py — v2.28 self-improving trace dataset (source-only).

Normalises the v2.26 + v2.27 local trace reconstructions into ONE canonical, contamination-guarded,
quality-scored self-improvement dataset schema suitable for LATER SFT / preference / lightweight
scaffold training. This is NOT a training milestone: no model is trained, no weights are touched.

What it adds on top of the raw v2.26/v2.27 traces:
  - one canonical record schema (see CANONICAL_FIELDS / docs/v2.28_self_improving_trace_dataset.md);
  - a UNIFORMLY RECOMPUTED contamination guard (v2.26's prompt/solution overlap was declared-only);
  - verifier-signal ENRICHMENT via the v2.27 format verifier (expected/observed/diagnosis/repair_hint);
  - re-verification of the final solution by standalone execution;
  - strict quality filters + split / training-use classification.

v2.27 is the authoritative source; v2.26 records reconstruct the SAME eval runs, so they are deduped
to v2.27 by (task_id, run) and otherwise routed as superseded.

Output (LOCAL-ONLY, gitignored):
    data/generated/v228/dataset.jsonl
    data/generated/v228/dataset_aggregate.json   (small; consumed by the summariser)

Usage:
    python scripts/build_v228_self_improving_dataset.py
"""

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.build_v227_trace_factory import (  # noqa: E402
    PROBE_INPUTS, _contamination_guard, _executes_pass, _load_overlap_corpus, _observed_output,
)
from scripts.v227_format_verifier import format_verify, render  # noqa: E402

META_PATH = ROOT / "data" / "v226_representation_tasks.jsonl"
SOURCES = [("v229", ROOT / "data/generated/v229/repair_traces.jsonl"),   # genuine format repairs
           ("v230", ROOT / "data/generated/v230/repair_traces.jsonl"),   # broadened format + algo repairs
           ("v227", ROOT / "data/generated/v227/traces.jsonl"),
           ("v226", ROOT / "data/generated/v226/traces.jsonl")]   # priority order; missing = skipped
OUT_DIR = ROOT / "data" / "generated" / "v228"

CANONICAL_FIELDS = [
    "record_id", "source", "task_id", "task_family", "capability_tag", "representation",
    "model_config", "prompt_mode", "retrieved_memory", "plan", "candidate_solution",
    "verifier_signal", "repair_plan", "repaired_solution", "final_solution", "final_status",
    "quality", "contamination_guard", "split", "use_tags", "rejection_reason",
]
FORMAT_FAILURE_TYPES = {"missing_null_marker", "extra_null_marker", "separator_error",
                        "ordering_error", "type_error", "format_error"}


def _capability_tag(family, representation):
    if "tree_serialize" in (family or "") or representation:
        return "format_control"
    return "unknown"


def _enrich_verifier(candidate, func, logical, representation, stored_status):
    """Re-run the v2.27 format verifier on the candidate's output to fill the structured block."""
    base = {"status": stored_status or "fail", "failure_type": None, "expected": None,
            "observed": None, "diagnosis": "", "repair_hint": None}
    if not (func and logical and representation):
        return base
    ok, observed = _observed_output(candidate, func, PROBE_INPUTS[0])
    try:
        expected = render(PROBE_INPUTS[0], logical, representation)
    except ValueError:
        return base
    if not ok:
        base.update({"failure_type": "algorithmic_error", "expected": _short(expected),
                     "observed": f"ERR::{observed}", "diagnosis": "candidate raised before output.",
                     "repair_hint": "make the function return without raising, then format."})
        return base
    block = format_verify(observed, expected, representation, logical)
    base.update({"status": block["status"], "failure_type": block["failure_type"],
                 "expected": _short(expected), "observed": _short(observed),
                 "diagnosis": block["diagnosis"], "repair_hint": block["repair_hint"]})
    return base


def _short(v):
    s = v if isinstance(v, str) else json.dumps(v, ensure_ascii=False)
    return s if len(s) <= 120 else s[:117] + "..."


def _quality(has_plan, has_vsig, has_repair, differs, repair_ok, fmt_ctrl, final_verified):
    flags = {
        "has_plan": bool(has_plan), "has_verifier_signal": bool(has_vsig),
        "has_repair": bool(has_repair), "candidate_differs_from_final": bool(differs),
        "repair_successful": bool(repair_ok), "format_control_used": bool(fmt_ctrl),
    }
    score = round((sum(flags.values()) + int(bool(final_verified))) / (len(flags) + 1), 3)
    flags["quality_score"] = score
    return flags


def main():
    if not META_PATH.exists():
        print("Run make build-v226-representation-tasks first.")
        sys.exit(1)
    meta = {json.loads(l)["id"]: json.loads(l) for l in open(META_PATH)}
    corpus = _load_overlap_corpus()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    records, seen_keys = [], set()
    found = False

    for source, path in SOURCES:
        if not path.exists():
            continue
        found = True
        for line in open(path):
            t = json.loads(line)
            tid = t.get("task_id")
            run = t.get("run")
            key = (tid, run)
            superseded = key in seen_keys
            seen_keys.add(key)
            m = meta.get(tid, {})
            func = m.get("func_name", "")
            logical = t.get("logical_task") or m.get("logical_task", "")
            representation = t.get("representation") or m.get("representation", "")
            family = t.get("task_family", "tree_serialize_repr")
            candidate = t.get("candidate_solution", "") or ""
            final = t.get("final_solution", "") or t.get("repaired_solution", "") or candidate
            repaired = t.get("repaired_solution", final)
            stored_status = (t.get("verifier_signal", {}) or {}).get("status", "")
            repair_plan = t.get("repair_plan", "")

            # recomputed ground truth
            cand_verified = _executes_pass(candidate)
            final_verified = _executes_pass(final)
            differs = candidate.strip() != final.strip()
            repair_outcome = t.get("repair_outcome", "")
            repair_claimed = bool(repair_plan) or repair_outcome.startswith("repair_attempted") or differs
            repair_successful = differs and (not cand_verified) and final_verified

            vsig = _enrich_verifier(candidate, func, logical, representation, stored_status)
            # prefer the source-stored failure_type when tree-only enrichment cannot classify
            # (non-tree-serialize tasks, e.g. the v2.30 broadened/algorithmic harvest).
            stored_v = t.get("verifier_signal") or {}
            if not vsig.get("failure_type") and stored_v.get("failure_type"):
                vsig["failure_type"] = stored_v["failure_type"]
                vsig["diagnosis"] = vsig.get("diagnosis") or stored_v.get("diagnosis", "")
                vsig["repair_hint"] = vsig.get("repair_hint") or stored_v.get("repair_hint")
            cg = _contamination_guard(tid, func, candidate, final, m, corpus)
            quality = _quality(
                has_plan=t.get("trace_quality", {}).get("plan_present", bool(repair_plan)),
                has_vsig=bool(vsig.get("status")),
                has_repair=repair_claimed,
                differs=differs, repair_ok=repair_successful,
                fmt_ctrl=("format" in (vsig.get("failure_type") or "") or representation != ""),
                final_verified=final_verified,
            )

            rec = {
                "record_id": f"{source}_{tid}_run{run}",
                "source": source, "task_id": tid, "task_family": family,
                "capability_tag": _capability_tag(family, representation),
                "representation": representation, "model_config": t.get("model_config", ""),
                "prompt_mode": t.get("prompt_mode", "structured_verifier_repair"),
                "retrieved_memory": [], "plan": [repair_plan] if repair_plan else [],
                "candidate_solution": candidate, "verifier_signal": vsig,
                "repair_plan": repair_plan, "repaired_solution": repaired,
                "final_solution": final,
                "final_status": "pass" if final_verified else "fail",
                "quality": quality, "contamination_guard": cg,
                "split": "", "use_tags": [], "rejection_reason": "",
            }

            # ── strict reject filters ──
            reasons = []
            if not vsig.get("status"):
                reasons.append("missing_verifier_signal")
            if rec["final_status"] not in ("pass", "fail"):
                reasons.append("missing_final_status")
            if repair_claimed and not differs and repair_outcome != "no_repair_needed":
                reasons.append("claimed_repair_without_change")
            if any(cg.values()):
                reasons.append("contamination_overlap")
            if not quality["has_plan"]:
                reasons.append("no_plan")
            if not family:
                reasons.append("no_task_family")
            if not representation:
                reasons.append("no_representation")
            if rec["final_status"] == "pass" and not final_verified:
                reasons.append("unverified_final")
            if superseded:
                reasons.append("superseded_by_higher_priority_source")

            if reasons:
                rec["split"] = "rejected"
                rec["rejection_reason"] = ",".join(reasons)
                records.append(rec)
                continue

            # ── training-use classification ──
            tags = []
            if final_verified and quality["has_plan"]:
                tags.append("sft")
            if repair_successful:
                # split by the (preferred) verifier failure_type: format-family vs algorithmic
                tags.append("format_repair" if vsig.get("failure_type") in FORMAT_FAILURE_TYPES
                            else "algorithmic_repair")
            if vsig.get("failure_type") in FORMAT_FAILURE_TYPES:
                tags.append("verifier_format")
            rec["use_tags"] = tags
            records.append(rec)

    # preference pairs: (task_id, representation) groups with BOTH a verified pass and fail
    groups = defaultdict(lambda: {"pass": 0, "fail": 0})
    for r in records:
        if r["split"] == "rejected":
            continue
        groups[(r["task_id"], r["representation"])][r["final_status"]] += 1
    for r in records:
        if r["split"] == "rejected":
            continue
        g = groups[(r["task_id"], r["representation"])]
        if g["pass"] > 0 and g["fail"] > 0:
            r["use_tags"].append("preference")

    # coarse split routing
    for r in records:
        if r["split"] == "rejected":
            continue
        tags = r["use_tags"]
        if any(t in tags for t in ("sft", "format_repair", "algorithmic_repair", "verifier_format")):
            r["split"] = "train_candidate"
        elif "preference" in tags:
            r["split"] = "preference_candidate"
        else:
            r["split"] = "eval_only"

    if not found and not records:
        print("No v2.26/v2.27 traces found. Run make build-v227-traces (and v226) first.")
        sys.exit(1)

    out_path = OUT_DIR / "dataset.jsonl"
    with open(out_path, "w") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    accepted = [r for r in records if r["split"] != "rejected"]
    rejected = [r for r in records if r["split"] == "rejected"]
    use_counts = Counter(t for r in accepted for t in r["use_tags"])
    agg = {
        "total_scanned": len(records),
        "accepted": len(accepted), "rejected": len(rejected),
        "rejection_reasons": dict(Counter(reason for r in rejected
                                           for reason in r["rejection_reason"].split(","))),
        "split_distribution": dict(Counter(r["split"] for r in records)),
        "use_tag_counts": {
            "sft_candidate": use_counts.get("sft", 0),
            "preference_pair_candidate": use_counts.get("preference", 0),
            "format_repair_candidate": use_counts.get("format_repair", 0),
            "algorithmic_repair_candidate": use_counts.get("algorithmic_repair", 0),
            "verifier_format_candidate": use_counts.get("verifier_format", 0),
        },
        "representation_distribution": dict(Counter(r["representation"] for r in accepted)),
        "task_family_distribution": dict(Counter(r["task_family"] for r in accepted)),
        "capability_tag_distribution": dict(Counter(r["capability_tag"] for r in accepted)),
        "contamination_guard_violations": sum(1 for r in records if any(r["contamination_guard"].values())),
        "quality_score_buckets": dict(Counter(
            ("0.8-1.0" if r["quality"]["quality_score"] >= 0.8 else
             "0.6-0.8" if r["quality"]["quality_score"] >= 0.6 else
             "0.4-0.6" if r["quality"]["quality_score"] >= 0.4 else "<0.4")
            for r in accepted)),
        "schema_fields": CANONICAL_FIELDS,
    }
    (OUT_DIR / "dataset_aggregate.json").write_text(json.dumps(agg, indent=2) + "\n")

    print(f"[v228] wrote {len(records)} records -> {out_path}  (LOCAL-ONLY, gitignored)")
    print(f"[v228] accepted {len(accepted)} / rejected {len(rejected)}")
    print(f"[v228] use tags: {agg['use_tag_counts']}")
    print(f"[v228] rejection reasons: {agg['rejection_reasons']}")
    print(f"[v228] contamination_guard_violations: {agg['contamination_guard_violations']}")


if __name__ == "__main__":
    main()
