"""生成 Quiz3 的模拟博弈树 tree（UTF-8 JSON，仅使用 Python 标准库）。

运行：python code/quiz3/generate_tree.py
根节点为 MAX，各层 MAX/MIN 交替；深度按根到叶的边数计算。
children 的列表顺序是原始从左到右顺序，叶节点仅保存 value。
expected 由完整 Minimax 生成，仅用于校验，不作为搜索或排序的输入。
"""

import argparse
import json
import random
from collections import Counter
from pathlib import Path


SEEDS = (20261109, 20261110, 20261111)
BRANCHING_FACTORS = (2, 3, 4)
DEPTHS = (3, 5, 7)


def branch(children, depth=0):
    return {"is_maximizing_player": depth % 2 == 0, "children": children}


def regular_tree(branching_factor, depth, rng, values=None, level=0):
    """生成完整规则树；values 非空时从指定分数集合抽样。"""
    if level == depth:
        value = rng.choice(values) if values is not None else rng.randint(-100, 100)
        return {"value": value}
    return branch([
        regular_tree(branching_factor, depth, rng, values, level + 1)
        for _ in range(branching_factor)
    ], level)


def irregular_tree(max_depth, rng):
    """保留一条最深分支及一个根的叶子孩子，确保深度不均衡。"""
    def build(level, force_depth=False):
        if level == max_depth or (
            not force_depth and level > 0 and rng.random() < 0.30
        ):
            return {"value": rng.randint(-100, 100)}
        count = rng.randint(2, 4)
        children = []
        for index in range(count):
            if level == 0 and index == 0:
                children.append({"value": rng.randint(-100, 100)})
            else:
                # 根的第二个孩子、随后每层的第一个孩子构成最深路径。
                forced_index = 1 if level == 0 else 0
                children.append(build(level + 1, force_depth and index == forced_index))
        return branch(children, level)

    return build(0, True)


def minimax_reference(node, maximizing=True):
    """完整递归求值，不使用剪枝、缓存或子节点排序。"""
    if "value" in node:
        return node["value"]
    scores = [minimax_reference(child, not maximizing) for child in node["children"]]
    return max(scores) if maximizing else min(scores)


def tree_statistics(tree):
    nodes_by_depth = Counter()
    leaves_by_depth = Counter()
    branching_counts = Counter()
    scores = []
    stack = [(tree, 0)]
    while stack:
        node, depth = stack.pop()
        nodes_by_depth[depth] += 1
        if "value" in node:
            leaves_by_depth[depth] += 1
            scores.append(node["value"])
        else:
            branching_counts[len(node["children"])] += 1
            stack.extend((child, depth + 1) for child in node["children"])
    return {
        "node_count": sum(nodes_by_depth.values()),
        "internal_node_count": sum(branching_counts.values()),
        "leaf_count": len(scores),
        "actual_depth": max(nodes_by_depth),
        "min_leaf_depth": min(leaves_by_depth),
        "max_branching_factor": max(branching_counts, default=0),
        "nodes_by_depth": dict(sorted(nodes_by_depth.items())),
        "leaves_by_depth": dict(sorted(leaves_by_depth.items())),
        "branching_factor_counts": dict(sorted(branching_counts.items())),
        "leaf_value_min": min(scores),
        "leaf_value_max": max(scores),
        "distinct_leaf_values": len(set(scores)),
    }


def make_case(case_id, kind, description, tree, seed=None, **parameters):
    statistics = tree_statistics(tree)
    score = minimax_reference(tree)
    children = tree.get("children", [])
    child_scores = [minimax_reference(child, False) for child in children]
    return {
        "id": case_id,
        "kind": kind,
        "description": description,
        "seed": seed,
        "parameters": parameters,
        "statistics": statistics,
        "expected": {
            "best_score": score,
            # 索引从 0 开始，分数并列时记录全部最优选择。
            "optimal_root_child_indices": [
                index for index, value in enumerate(child_scores) if value == score
            ],
            "full_minimax_node_visits": statistics["node_count"],
            "full_minimax_leaf_evaluations": statistics["leaf_count"],
        },
        "tree_structure": tree,
    }


def build_dataset():
    leaf = lambda value: {"value": value}
    pair = lambda a, b: branch([leaf(a), leaf(b)], 2)
    # 与题目 JSON 一致：两处 [6,9] 是独立子树，不共享 Python 对象。
    example = branch([
        branch([pair(3, 5), pair(6, 9)], 1),
        branch([pair(6, 9), pair(1, 2)], 1),
    ])
    cases = [
        make_case("example", "example", "题目示例，最佳得分为 5。", example),
        make_case("single_leaf", "single_leaf", "根即叶节点，深度为 0。", leaf(-7)),
    ]
    chain = leaf(11)
    for level in reversed(range(7)):
        chain = branch([chain], level)
    cases.append(make_case(
        "single_branch", "single_branch", "深度 7 的单分支，无可跳过的兄弟节点。", chain))
    cases.append(make_case(
        "all_negative", "all_negative", "所有分数为负，检查极值初始化。",
        regular_tree(2, 3, random.Random(SEEDS[0]), (-20, -9, -3, -1)),
        SEEDS[0], branching_factor=2, depth=3))
    cases.append(make_case(
        "repeated_scores", "repeated_scores", "少量离散分数，产生大量平局。",
        regular_tree(3, 3, random.Random(SEEDS[1]), (-1, 0, 1)),
        SEEDS[1], branching_factor=3, depth=3))
    cases.append(make_case(
        "all_equal", "all_equal", "所有叶节点得分为 5，观察相等边界的剪枝。",
        regular_tree(3, 3, random.Random(SEEDS[2]), (5,)),
        SEEDS[2], branching_factor=3, depth=3))
    cases.append(make_case(
        "equality_cutoff", "equality_cutoff",
        "左侧返回 5 后，右侧首叶为 5，alpha=beta 时可跳过 100。",
        branch([branch([leaf(5), leaf(5)], 1),
                branch([leaf(5), leaf(100)], 1)])))
    for factor in BRANCHING_FACTORS:
        for depth in DEPTHS:
            for seed in SEEDS:
                cases.append(make_case(
                    f"regular_b{factor}_d{depth}_s{seed}", "regular",
                    "完整规则树，叶分数均匀抽取自整数区间 [-100,100]。",
                    regular_tree(factor, depth, random.Random(seed)), seed,
                    branching_factor=factor, depth=depth, leaf_value_range=[-100, 100]))
    for depth in (5, 7):
        for seed in SEEDS:
            cases.append(make_case(
                f"irregular_d{depth}_s{seed}", "irregular",
                "分支数 2～4，允许提前终止；同时保证浅叶和指定深度的叶子。",
                irregular_tree(depth, random.Random(seed)), seed,
                maximum_depth=depth, branching_factor_range=[2, 4],
                early_leaf_probability=0.30, leaf_value_range=[-100, 100]))
    return {
        "schema_version": "quiz3.tree.v1",
        "description": "Quiz3：完整 Minimax 与 Alpha-Beta 访问顺序对照的固定输入。",
        "rules": {
            "root_player": "MAX",
            "players_alternate_by_depth": True,
            "depth_definition": "根深度为 0，深度按边数计算。",
            "default_child_order": "children 列表从左到右。",
            "leaf_schema": {"value": "整数得分"},
            "internal_node_schema": {
                "is_maximizing_player": "布尔值，与深度奇偶一致",
                "children": "非空子节点列表",
            },
            "representation": "展开树，无共享子节点或环。",
            "comparison": "各策略使用同一树，仅改变兄弟节点访问顺序。",
            "expected_usage": "参考答案只用于校验，不作为算法或实用排序的输入。",
            "counting": "每次递归访问计一个节点，叶子求值另计；单叶根最优孩子列表为空。",
        },
        "configuration": {
            "seeds": list(SEEDS),
            "regular_branching_factors": list(BRANCHING_FACTORS),
            "regular_depths": list(DEPTHS),
            "irregular_maximum_depths": [5, 7],
        },
        "case_count": len(cases),
        "case_counts_by_kind": dict(sorted(Counter(c["kind"] for c in cases).items())),
        "cases": cases,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("tree"),
                        help="输出文件路径，默认为脚本同目录的 tree")
    args = parser.parse_args()
    dataset = build_dataset()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(dataset, ensure_ascii=False, indent=2) + "\n",
                           encoding="utf-8")
    print(f"Generated {dataset['case_count']} trees: {args.output}")
    print(f"Regular: 27; irregular: 6; example and boundary cases: 7")


if __name__ == "__main__":
    main()
