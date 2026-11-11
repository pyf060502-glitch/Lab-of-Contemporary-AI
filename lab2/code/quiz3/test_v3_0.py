"""Quiz3：独立求值、剪枝边界、顺序固定与测量汇总检查。"""

import copy
import itertools
import json
import random
import runpy
import unittest
from pathlib import Path


MODULE = runpy.run_path(str(Path(__file__).with_name("v3.0.py")))
DATA_PATH = Path(__file__).with_name("tree")


def oracle(node, maximizing=True):
    """测试用完整递归求值，不调用被测搜索。"""
    if "value" in node:
        return node["value"]
    values = [oracle(child, not maximizing) for child in node["children"]]
    return (max if maximizing else min)(values)


def structure(node):
    if "value" in node:
        return 1, 1, 0
    child_stats = [structure(c) for c in node["children"]]
    return (1 + sum(c[0] for c in child_stats),
            sum(c[1] for c in child_stats), 1 + max(c[2] for c in child_stats))


class GameTreeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cases = json.loads(DATA_PATH.read_text(encoding="utf-8"))["cases"]

    def check_tree(self, raw):
        before = copy.deepcopy(raw)
        expected = oracle(raw)
        nodes, leaves, depth = structure(raw)
        self.assertEqual(MODULE["independent_reference"](raw), (expected, nodes, leaves, depth))
        root = MODULE["from_json"](raw)
        orders = MODULE["prepare_orders"](root, 42)
        for name, ordered_root in orders.items():
            fn = MODULE["minimax_search"] if name == "Minimax" else MODULE["alpha_beta_search"]
            result = fn(ordered_root)
            MODULE["validate"](result, (expected, nodes, leaves, depth), name)
            self.assertEqual(result.best_score, expected)
            self.assertEqual(fn(ordered_root), result)
        self.assertEqual(MODULE["prepare_orders"](root, 42), orders)
        self.assertEqual(raw, before)
        self.assertEqual(MODULE["alpha_beta"](root), expected)
        self.assertEqual(MODULE["minimax"](root), expected)
        return root

    def test_example_and_equality_cutoff_counts(self):
        example = self.check_tree(self.cases[0]["tree_structure"])
        full = MODULE["minimax_search"](example)
        pruned = MODULE["alpha_beta_search"](example)
        self.assertEqual((full.best_score, full.visited_nodes, full.leaf_evaluations), (5, 15, 8))
        self.assertEqual((pruned.visited_nodes, pruned.leaf_evaluations, pruned.effective_cutoffs),
                         (14, 7, 1))
        raw = next(c["tree_structure"] for c in self.cases if c["id"] == "equality_cutoff")
        result = MODULE["alpha_beta_search"](self.check_tree(raw))
        self.assertEqual((result.best_score, result.visited_nodes,
                          result.leaf_evaluations, result.effective_cutoffs), (5, 6, 3, 1))

    def test_exhaustive_small_binary_trees(self):
        # 正负数、零和平局的全部 3^8 种深度 3 二叉树。
        for values in itertools.product((-1, 0, 1), repeat=8):
            leaves = [{"value": value} for value in values]
            pairs = [{"children": leaves[i:i+2]} for i in range(0, 8, 2)]
            raw = {"children": [{"children": pairs[:2]}, {"children": pairs[2:]}]}
            self.check_tree(raw)

    def test_dataset_including_irregular_and_boundaries(self):
        for case in self.cases:
            self.check_tree(case["tree_structure"])

    def test_invalid_roles_and_leaf(self):
        with self.assertRaises(ValueError):
            MODULE["from_json"]({"is_maximizing_player": False, "children": [{"value": 1}]})
        with self.assertRaises(ValueError):
            MODULE["from_json"]({"children": []})
        with self.assertRaises(ValueError):
            MODULE["from_json"]({"value": 1, "children": [{"value": 2}]})

    def test_benchmark_samples_and_equal_weight_aggregation(self):
        cases = [self.cases[0], self.cases[1]]
        rows = MODULE["benchmark"](cases, repeats=2, memory_repeats=2)
        self.assertEqual(len(rows), 8)
        for row in rows:
            self.assertTrue(row["correct"])
            self.assertEqual(len(row["time_samples_ms"]), 2)
            self.assertEqual(len(row["memory_samples_bytes"]), 2)
            self.assertTrue(all(v > 0 for v in row["time_samples_ms"]))
            self.assertGreater(row["memory_peak_median_kib"], 0)
        summary = MODULE["aggregate"](rows)
        row = next(r for r in summary if r["dimension"] == "overall"
                   and r["algorithm"] == "AB original")
        self.assertEqual(row["case_count"], 2)
        self.assertAlmostEqual(row["visited_nodes_mean"], 7.5)
        self.assertAlmostEqual(row["node_saving_percent_mean"], 100 / 30)
        # 不以总体访问量比率代替逐树节省率的平均值。
        self.assertNotAlmostEqual(row["node_saving_percent_mean"], 100 * (1 - 15 / 16))


if __name__ == "__main__":
    unittest.main()
