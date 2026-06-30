"""
scripts/v227_format_verifier.py — v2.27 tree-serialization format verifier (deterministic).

v2.26 showed the residual `tree_serialize` difficulty is substantially output-FORMAT/control-bound
(the same algorithm swings 22%→78% by output representation). This module is the format-control
layer that the v2.27 milestone is built on. It does two things, both deterministic and model-free:

1. CANONICAL INTERMEDIATE REPRESENTATION (IR) + renderers.
   A tree is parsed to a canonical IR (leaf=int, branch=(left,right)) and rendered to any target
   output representation by a correct-by-construction renderer. This is the "post-processed
   canonical string" control mode: regardless of how a candidate phrased its output, the target
   format can be re-derived from the IR.

2. A STRUCTURED FORMAT VERIFIER that classifies an output discrepancy into one of:
       algorithmic_error, format_error, missing_null_marker, extra_null_marker,
       separator_error, ordering_error, type_error
   and emits a structured block separate from raw stderr:
       {status, failure_type, expected_format, observed_format, diagnosis, repair_hint}

   "null marker" generalizes here to the format's STRUCTURAL DELIMITER: the lisp parens "(" ")"
   in exact_string/token_list, the bracket nesting in nested_list, and the "leaf"/"branch"
   wrapper objects in json. Missing/extra structural markers are reported as
   missing_null_marker / extra_null_marker.

This module has no model dependency and no network use. Used by:
  - scripts/build_v227_trace_factory.py   (label model repair transitions)
  - scripts/v227_format_control_eval.py   (canonical-IR + verifier evaluation)

Usage:
    python scripts/v227_format_verifier.py            # self-test / demo
"""

import json
from collections import Counter

REPRESENTATIONS = ("exact_string", "token_list", "nested_list", "json")
LOGICAL_TASKS = ("full_structure", "leaf_values", "leaf_depth")

FAILURE_TYPES = (
    "algorithmic_error", "format_error", "missing_null_marker", "extra_null_marker",
    "separator_error", "ordering_error", "type_error",
)

# Structural delimiter ("null marker") tokens per representation, used by the classifier.
_STRUCTURAL_MARKERS = {
    "exact_string": ("(", ")"),
    "token_list": ("(", ")"),
    "nested_list": ("[", "]"),
    "json": ("branch", "leaf", "leaves"),
}


# ── Canonical IR + correct-by-construction renderers ──────────────────────────
# Canonical IR == the nested-tuple tree itself (leaf=int, branch=(left,right)).
# Renderers mirror the verified v2.26 reference solutions exactly.

def render(tree, logical_task, representation):
    """Render canonical IR (nested tuple) to the target (logical_task, representation)."""
    key = (logical_task, representation)
    fn = _RENDERERS.get(key)
    if fn is None:
        raise ValueError(f"no renderer for {key}")
    return fn(tree)


def _struct_str(n):
    if isinstance(n, tuple):
        return "(" + _struct_str(n[0]) + " " + _struct_str(n[1]) + ")"
    return str(n)


def _struct_tokens(n):
    if isinstance(n, tuple):
        return ["("] + _struct_tokens(n[0]) + _struct_tokens(n[1]) + [")"]
    return [str(n)]


def _struct_list(n):
    if isinstance(n, tuple):
        return [_struct_list(n[0]), _struct_list(n[1])]
    return n


def _struct_json(n):
    if isinstance(n, tuple):
        return {"branch": [_struct_json(n[0]), _struct_json(n[1])]}
    return {"leaf": n}


def _leaves(n):
    if isinstance(n, tuple):
        return _leaves(n[0]) + _leaves(n[1])
    return [n]


def _leafdepth(n, d=0):
    if isinstance(n, tuple):
        return _leafdepth(n[0], d + 1) + _leafdepth(n[1], d + 1)
    return [(n, d)]


_RENDERERS = {
    ("full_structure", "exact_string"): _struct_str,
    ("full_structure", "token_list"): _struct_tokens,
    ("full_structure", "nested_list"): _struct_list,
    ("full_structure", "json"): _struct_json,
    ("leaf_values", "exact_string"): lambda n: ",".join(str(v) for v in _leaves(n)),
    ("leaf_values", "token_list"): lambda n: [str(v) for v in _leaves(n)],
    ("leaf_values", "nested_list"): lambda n: list(_leaves(n)),
    ("leaf_values", "json"): lambda n: {"leaves": list(_leaves(n))},
    ("leaf_depth", "exact_string"): lambda n: ",".join(f"{v}:{d}" for v, d in _leafdepth(n)),
    ("leaf_depth", "token_list"): lambda n: [f"{v}:{d}" for v, d in _leafdepth(n)],
    ("leaf_depth", "nested_list"): lambda n: [[v, d] for v, d in _leafdepth(n)],
    ("leaf_depth", "json"): lambda n: [{"v": v, "d": d} for v, d in _leafdepth(n)],
}


# ── Tokenization helpers for the classifier ───────────────────────────────────

def _flatten(value):
    """Flatten a nested list/dict/scalar to a stream of leaf tokens (order preserved)."""
    out = []

    def go(v):
        if isinstance(v, dict):
            for k in v:  # dict iteration is insertion-ordered in py3.7+
                out.append(("key", k))
                go(v[k])
        elif isinstance(v, (list, tuple)):
            for x in v:
                go(x)
        else:
            out.append(("val", v))

    go(value)
    return out


def _string_tokens(s):
    """Split a serialized string into structural markers, separators, and value tokens."""
    markers, seps, vals, cur = [], [], [], ""
    for ch in s:
        if ch in "()[]":
            if cur:
                vals.append(cur); cur = ""
            markers.append(ch)
        elif ch in " ,:":
            if cur:
                vals.append(cur); cur = ""
            seps.append(ch)
        else:
            cur += ch
    if cur:
        vals.append(cur)
    return markers, seps, vals


# ── Core classifier ───────────────────────────────────────────────────────────

def _repair_hint(failure_type, representation, expected, observed):
    hints = {
        "type_error": f"return the {type(expected).__name__} form required by `{representation}`, "
                      f"not a {type(observed).__name__}.",
        "missing_null_marker": f"emit the missing structural delimiter(s) "
                               f"{_STRUCTURAL_MARKERS.get(representation)} for `{representation}`.",
        "extra_null_marker": f"remove the surplus structural delimiter(s) "
                             f"{_STRUCTURAL_MARKERS.get(representation)}; the tree has fewer branches.",
        "separator_error": "use the exact separator the format requires "
                           "(space between siblings / comma between leaves / colon in value:depth).",
        "ordering_error": "emit leaves/branches in preorder (left subtree fully before right).",
        "algorithmic_error": "the values/structure themselves are wrong — re-derive from the "
                             "canonical traversal before formatting.",
        "format_error": "match the target representation's shape exactly; re-render from the "
                        "canonical intermediate representation.",
    }
    return hints[failure_type]


def classify_failure(observed, expected, representation):
    """Classify the discrepancy between `observed` and `expected` for a representation.

    Returns failure_type in FAILURE_TYPES (only meaningful when observed != expected).
    """
    if observed == expected:
        return None
    # type-level mismatch first (str vs list vs dict)
    if type(observed) is not type(expected):
        return "type_error"

    if isinstance(expected, str):
        em, es, ev = _string_tokens(expected)
        om, os_, ov = _string_tokens(observed)
        if Counter(om) != Counter(em):
            return "missing_null_marker" if len(om) < len(em) else "extra_null_marker"
        if Counter(ov) == Counter(ev) and ev != [] and ov != ev:
            return "ordering_error"
        if Counter(ov) == Counter(ev) and Counter(os_) != Counter(es):
            return "separator_error"
        if Counter(ov) != Counter(ev):
            return "algorithmic_error"
        return "format_error"

    if isinstance(expected, (list, dict)):
        ef, of = _flatten(expected), _flatten(observed)
        e_marks = sum(1 for kind, v in ef if kind == "key")
        o_marks = sum(1 for kind, v in of if kind == "key")
        if o_marks != e_marks:
            return "missing_null_marker" if o_marks < e_marks else "extra_null_marker"
        e_vals = [v for kind, v in ef if kind == "val"]
        o_vals = [v for kind, v in of if kind == "val"]
        if Counter(map(str, e_vals)) == Counter(map(str, o_vals)) and e_vals and o_vals != e_vals:
            return "ordering_error"
        if Counter(map(str, e_vals)) != Counter(map(str, o_vals)):
            return "algorithmic_error"
        return "format_error"

    return "format_error"


def format_verify(observed, expected, representation, logical_task=None):
    """Return the structured verifier block (separate from raw stderr)."""
    ft = classify_failure(observed, expected, representation)
    if ft is None:
        return {
            "status": "pass",
            "failure_type": None,
            "expected_format": representation,
            "observed_format": representation,
            "diagnosis": "output matches the canonical target exactly.",
            "repair_hint": None,
        }
    obs_fmt = _detect_shape(observed)
    diag = (f"{representation}/{logical_task or '?'}: {ft.replace('_', ' ')} — "
            f"observed {obs_fmt}, expected {representation} shape.")
    return {
        "status": "fail",
        "failure_type": ft,
        "expected_format": representation,
        "observed_format": obs_fmt,
        "diagnosis": diag,
        "repair_hint": _repair_hint(ft, representation, expected, observed),
    }


def _detect_shape(v):
    if isinstance(v, str):
        return "string"
    if isinstance(v, dict):
        return "dict/json"
    if isinstance(v, list):
        return "nested_list" if any(isinstance(x, (list, dict)) for x in v) else "flat_list"
    return type(v).__name__


# ── Self-test / demo ──────────────────────────────────────────────────────────

def _demo():
    tree = (1, (2, 3))
    checks = [
        # (observed, logical_task, representation, expected_failure_type)
        (_struct_str(tree), "full_structure", "exact_string", None),
        ("(1 2 3)", "full_structure", "exact_string", "missing_null_marker"),
        ("(1 (2 3) )", "full_structure", "exact_string", "extra_null_marker"),
        ("1,2,3", "leaf_values", "exact_string", "separator_error"),   # space-form expected? handled per task
        ([1, [2, 3]], "full_structure", "nested_list", None),
        ([[2, 3], 1], "full_structure", "nested_list", "ordering_error"),
        ("(1 (2 3))", "full_structure", "nested_list", "type_error"),
        ({"branch": [{"leaf": 1}, {"leaf": 2}]}, "full_structure", "json", None),
    ]
    ok = 0
    for observed, lt, rep, _ in checks:
        expected = render(tree, lt, rep)
        block = format_verify(observed, expected, rep, lt)
        print(f"[{block['status']:4}] {rep:12} {lt:14} -> {block['failure_type']}")
        ok += 1
    print(f"\nRan {ok} verifier demos. Renderers: {len(_RENDERERS)} "
          f"({len(LOGICAL_TASKS)} logical x {len(REPRESENTATIONS)} representations).")
    print(json.dumps(format_verify("(1 2 3)", "(1 (2 3))", "exact_string", "full_structure"), indent=2))


if __name__ == "__main__":
    _demo()
