"""Dependency-free regression checks for data-policy safety.

These tests intentionally avoid importing the training stack (torch,
transformers, datasets) so they can run in lightweight CI.
"""
from __future__ import annotations

import ast
import random
import unittest
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple


ROOT = Path(__file__).resolve().parents[1]


@dataclass
class Row:
    example_id: int
    split: str


def _load_split_function():
    path = ROOT / "src" / "pipeline" / "stages.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    fn = next(
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_split_rows"
    )
    module = ast.Module(body=[fn], type_ignores=[])
    ast.fix_missing_locations(module)
    ns = {
        "random": random,
        "List": List,
        "Tuple": Tuple,
        "StandardRow": Row,
    }
    exec(compile(module, str(path), "exec"), ns)
    return ns["_split_rows"]


class SplitSafetyTest(unittest.TestCase):
    def test_eval_only_sample_is_seeded_before_truncation(self):
        split_rows = _load_split_function()
        rows = [Row(i, "test") for i in range(100)]
        _, _, test_a = split_rows(rows, 0, 0, 20, 42)
        _, _, test_b = split_rows(rows, 0, 0, 20, 42)
        self.assertEqual([r.example_id for r in test_a], [r.example_id for r in test_b])
        self.assertNotEqual([r.example_id for r in test_a], list(range(20)))

    def test_custom_partition_is_disjoint(self):
        split_rows = _load_split_function()
        rows = [Row(i, "test") for i in range(100)]
        train, dev, test = split_rows(rows, 50, 10, 20, 42)
        ids = [set(r.example_id for r in part) for part in (train, dev, test)]
        self.assertEqual([len(x) for x in ids], [50, 10, 20])
        self.assertFalse(ids[0] & ids[1])
        self.assertFalse(ids[0] & ids[2])
        self.assertFalse(ids[1] & ids[2])


class ColdStartSafetyTest(unittest.TestCase):
    def test_oracle_drafts_are_not_the_default(self):
        src = (ROOT / "src" / "manager" / "evolve.py").read_text(encoding="utf-8")
        self.assertNotIn("_draft_answer_str(row.ground_truth)", src)
        self.assertNotIn('current_draft": row.ground_truth', src)
        self.assertNotIn("candidate_answer=(row.ground_truth", src)
        self.assertIn('draft_source: str = "base_stepwise"', src)
        self.assertIn("updated = draft_generator.predict(history", src)
        self.assertNotIn("if i + 1 < len(seq) and draft_generator is not None", src)
        self.assertIn('"final_on_policy_draft": draft_answer', src)
        self.assertIn('candidate_answer=(draft_answer if kind == "verifier"', src)

    def test_mmlu_test_training_requires_explicit_acknowledgement(self):
        src = (ROOT / "src" / "pipeline" / "cli.py").read_text(encoding="utf-8")
        self.assertIn("--mmlu_pro_allow_test_training", src)
        self.assertIn("Refusing to train on the official MMLU-Pro test split", src)

    def test_gpqa_training_excludes_all_diamond_questions(self):
        src = (ROOT / "scripts" / "build_gpqa_splits.py").read_text(encoding="utf-8")
        self.assertIn("train_rows = list(non_dia)", src)
        self.assertNotIn("dia_in_ext[args.eval_n :] + non_dia", src)
        self.assertIn("all_diamond_h & train_h", src)

    def test_base_direct_evaluation_is_supported(self):
        src = (ROOT / "src" / "pipeline" / "stages.py").read_text(encoding="utf-8")
        self.assertIn('use_base = manager_dir in {"base", ctx.base_model}', src)
        self.assertIn('"macro_task_accuracy"', src)
        self.assertIn('suffix = "_base" if use_base else ""', src)

    def test_tool_evaluation_can_reuse_remote_advisors(self):
        src = (ROOT / "src" / "pipeline" / "stages.py").read_text(encoding="utf-8")
        self.assertIn("RemoteSubagentPool", src)
        self.assertIn('"subagent_server_url": subagent_server_url', src)


if __name__ == "__main__":
    unittest.main()
