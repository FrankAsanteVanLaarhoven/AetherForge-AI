"""Validation tests for data/targeted_failure_dev_tasks.jsonl.

Checks structural and scientific-hygiene requirements:
  - 50 tasks total, 10 per category
  - All required fields present
  - IDs unique and snake_case
  - No ID overlap with frozen held-out benchmark
  - Verification blocks have >= 3 assertions and print('PASS')
  - No fake OBSERVATION tags in task prompts
  - No external library imports in verification code
  - Tasks solvable with standard Python only (no numpy/pandas/etc)
"""

import json
import re
import pathlib
import pytest

DATA_DIR = pathlib.Path(__file__).parent.parent / "data"
TASKS_FILE = DATA_DIR / "targeted_failure_dev_tasks.jsonl"
HELDOUT_FILE = DATA_DIR / "heldout_code_agent_tasks.jsonl"

VALID_CATEGORIES = {
    "rle_decode_like",
    "group_anagrams_like",
    "deep_get_like",
    "merge_intervals_like",
    "tree_depth_tuple_like",
}

EXTERNAL_LIBS = {"numpy", "pandas", "scipy", "torch", "sklearn", "tensorflow", "requests"}

SNAKE_CASE_RE = re.compile(r'^[a-z][a-z0-9_]*$')


def _load_tasks():
    assert TASKS_FILE.exists(), f"Missing {TASKS_FILE}"
    tasks = []
    with open(TASKS_FILE) as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                tasks.append(json.loads(line))
            except json.JSONDecodeError as e:
                pytest.fail(f"Line {i} is not valid JSON: {e}")
    return tasks


def _load_heldout_ids():
    if not HELDOUT_FILE.exists():
        return set()
    ids = set()
    with open(HELDOUT_FILE) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            task = json.loads(line)
            if "id" in task:
                ids.add(task["id"])
    return ids


@pytest.fixture(scope="module")
def tasks():
    return _load_tasks()


@pytest.fixture(scope="module")
def heldout_ids():
    return _load_heldout_ids()


class TestFileStructure:
    def test_file_exists(self):
        assert TASKS_FILE.exists(), f"Missing {TASKS_FILE}"

    def test_total_task_count(self, tasks):
        assert len(tasks) == 50, f"Expected 50 tasks, got {len(tasks)}"

    def test_tasks_per_category(self, tasks):
        counts = {}
        for t in tasks:
            counts[t.get("category", "MISSING")] = counts.get(t.get("category", "MISSING"), 0) + 1
        for cat in VALID_CATEGORIES:
            assert counts.get(cat, 0) == 10, (
                f"Category '{cat}' has {counts.get(cat, 0)} tasks, expected 10"
            )


class TestRequiredFields:
    @pytest.mark.parametrize("field", ["id", "category", "difficulty", "task", "verification"])
    def test_all_tasks_have_field(self, tasks, field):
        missing = [t.get("id", f"task#{i}") for i, t in enumerate(tasks) if field not in t]
        assert not missing, f"Tasks missing '{field}': {missing}"

    def test_ids_are_unique(self, tasks):
        ids = [t["id"] for t in tasks]
        dupes = [i for i in ids if ids.count(i) > 1]
        assert not dupes, f"Duplicate IDs: {set(dupes)}"

    def test_ids_are_snake_case(self, tasks):
        bad = [t["id"] for t in tasks if not SNAKE_CASE_RE.match(t["id"])]
        assert not bad, f"Non-snake_case IDs: {bad}"

    def test_categories_are_valid(self, tasks):
        bad = [(t["id"], t["category"]) for t in tasks if t["category"] not in VALID_CATEGORIES]
        assert not bad, f"Invalid categories: {bad}"

    def test_difficulty_values(self, tasks):
        valid = {"easy", "medium", "hard"}
        bad = [(t["id"], t["difficulty"]) for t in tasks if t.get("difficulty") not in valid]
        assert not bad, f"Invalid difficulty values: {bad}"


class TestScientificHygiene:
    def test_no_id_overlap_with_heldout(self, tasks, heldout_ids):
        overlap = {t["id"] for t in tasks} & heldout_ids
        assert not overlap, (
            f"Dev task IDs overlap with frozen held-out IDs: {overlap}. "
            "Scientific violation: dev tasks must be similar-but-not-identical."
        )

    def test_no_observation_tag_in_task(self, tasks):
        contaminated = [
            t["id"] for t in tasks if "OBSERVATION:" in t.get("task", "")
        ]
        assert not contaminated, (
            f"Tasks contain fake OBSERVATION tags (contamination): {contaminated}"
        )

    def test_no_observation_tag_in_verification(self, tasks):
        contaminated = [
            t["id"] for t in tasks if "OBSERVATION:" in t.get("verification", "")
        ]
        assert not contaminated, f"Verification contains OBSERVATION tags: {contaminated}"


class TestVerificationBlocks:
    def test_verification_has_pass_print(self, tasks):
        missing = [t["id"] for t in tasks if "print('PASS')" not in t.get("verification", "")]
        assert not missing, f"Tasks missing print('PASS') in verification: {missing}"

    def test_verification_has_min_3_assertions(self, tasks):
        insufficient = [
            t["id"] for t in tasks
            if t.get("verification", "").count("assert ") < 3
        ]
        assert not insufficient, (
            f"Tasks with < 3 assertions in verification: {insufficient}"
        )

    def test_verification_no_external_imports(self, tasks):
        bad = []
        for t in tasks:
            code = t.get("verification", "")
            for lib in EXTERNAL_LIBS:
                if f"import {lib}" in code or f"from {lib}" in code:
                    bad.append((t["id"], lib))
        assert not bad, f"Verification blocks use external libraries: {bad}"

    def test_task_no_external_imports(self, tasks):
        bad = []
        for t in tasks:
            code = t.get("task", "")
            for lib in EXTERNAL_LIBS:
                if f"import {lib}" in code or f"from {lib}" in code:
                    bad.append((t["id"], lib))
        assert not bad, f"Task prompts reference external libraries: {bad}"


class TestCategoryContent:
    def test_rle_tasks_mention_decode_or_expand(self, tasks):
        rle_tasks = [t for t in tasks if t["category"] == "rle_decode_like"]
        keywords = {"decode", "rle", "expand", "run", "encoded", "roundtrip", "interleave"}
        bad = [
            t["id"] for t in rle_tasks
            if not any(kw in t["task"].lower() for kw in keywords)
        ]
        assert not bad, f"RLE tasks missing expected keywords: {bad}"

    def test_anagram_tasks_mention_anagram_or_group(self, tasks):
        anagram_tasks = [t for t in tasks if t["category"] == "group_anagrams_like"]
        keywords = {"anagram", "sorted", "group", "canonical", "frequency"}
        bad = [
            t["id"] for t in anagram_tasks
            if not any(kw in t["task"].lower() for kw in keywords)
        ]
        assert not bad, f"Anagram tasks missing expected keywords: {bad}"

    def test_deep_get_tasks_mention_nested_or_default(self, tasks):
        deep_tasks = [t for t in tasks if t["category"] == "deep_get_like"]
        keywords = {"nested", "default", "key", "dict", "path", "navigate", "lookup", "get"}
        bad = [
            t["id"] for t in deep_tasks
            if not any(kw in t["task"].lower() for kw in keywords)
        ]
        assert not bad, f"Deep-get tasks missing expected keywords: {bad}"

    def test_interval_tasks_mention_interval_or_merge(self, tasks):
        interval_tasks = [t for t in tasks if t["category"] == "merge_intervals_like"]
        keywords = {"interval", "overlap", "merge", "sort", "range", "gap", "covered"}
        bad = [
            t["id"] for t in interval_tasks
            if not any(kw in t["task"].lower() for kw in keywords)
        ]
        assert not bad, f"Interval tasks missing expected keywords: {bad}"

    def test_tree_tasks_mention_tuple_or_depth(self, tasks):
        tree_tasks = [t for t in tasks if t["category"] == "tree_depth_tuple_like"]
        keywords = {"tuple", "leaf", "depth", "tree", "nested", "flatten", "node"}
        bad = [
            t["id"] for t in tree_tasks
            if not any(kw in t["task"].lower() for kw in keywords)
        ]
        assert not bad, f"Tree tasks missing expected keywords: {bad}"
