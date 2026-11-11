"""Quiz3：Minimax 与三种访问顺序的 Alpha-Beta 对照。

运行：python code/quiz3/v3.0.py
输入 code/quiz3/tree，输出 result/quiz3/*_v3.0。
只搜索真实叶节点，不使用深度截断、理想排序或缓存。
"""

import argparse
import csv
import gc
import hashlib
import json
import math
import os
import platform
import random
import statistics
import sys
import time
import tracemalloc
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path


LAB2 = Path(__file__).resolve().parents[2]
ALGORITHMS = ("Minimax", "AB original", "AB reverse", "AB random")
ORDER_SEED = 20261110
MEASUREMENT_SEED = 20261109


@dataclass(frozen=True)
class TreeNode:
    value: object = None
    children: tuple = ()


@dataclass
class SearchResult:
    best_score: int = 0
    visited_nodes: int = 0
    leaf_evaluations: int = 0
    effective_cutoffs: int = 0
    maximum_depth: int = 0


def from_json(node, depth=0):
    """在测量外构建输入树，并检查角色与深度一致。"""
    if "value" in node:
        if type(node["value"]) is not int or node.get("children"):
            raise ValueError("叶节点应具有整数 value 且没有子节点")
        return TreeNode(value=node["value"])
    if not node.get("children"):
        raise ValueError("内部节点 children 不能为空")
    if node.get("is_maximizing_player", depth % 2 == 0) != (depth % 2 == 0):
        raise ValueError("MAX/MIN 角色必须逐层交替，根为 MAX")
    return TreeNode(children=tuple(from_json(child, depth + 1) for child in node["children"]))


def prepare_orders(root, seed):
    """预先固定遍历顺序；建树、复制与随机打乱均不计入搜索测量。"""
    rng = random.Random(seed)

    def reorder(node, mode):
        if not node.children:
            return node
        children = list(node.children)
        if mode == "reverse":
            children.reverse()
        else:
            rng.shuffle(children)
        return TreeNode(children=tuple(reorder(child, mode) for child in children))

    return {
        "Minimax": root, "AB original": root,
        "AB reverse": reorder(root, "reverse"),
        "AB random": reorder(root, "random"),
    }


def minimax_search(root):
    result = SearchResult()

    def visit(node, maximizing, depth):
        result.visited_nodes += 1
        result.maximum_depth = max(result.maximum_depth, depth)
        if not node.children:
            result.leaf_evaluations += 1
            return node.value
        value = -math.inf if maximizing else math.inf
        for child in node.children:
            candidate = visit(child, not maximizing, depth + 1)
            if maximizing:
                value = max(value, candidate)
            else:
                value = min(value, candidate)
        return value

    result.best_score = visit(root, True, 0)
    return result


def alpha_beta_search(root):
    """完整窗口搜索；返回值可能为剪枝分支的界，但根值为精确值。"""
    result = SearchResult()

    def visit(node, maximizing, alpha, beta, depth):
        result.visited_nodes += 1
        result.maximum_depth = max(result.maximum_depth, depth)
        if not node.children:
            result.leaf_evaluations += 1
            return node.value
        value = -math.inf if maximizing else math.inf
        for index, child in enumerate(node.children):
            candidate = visit(child, not maximizing, alpha, beta, depth + 1)
            if maximizing:
                value = max(value, candidate)
                alpha = max(alpha, value)
            else:
                value = min(value, candidate)
                beta = min(beta, value)
            if alpha >= beta:
                if index + 1 < len(node.children):
                    result.effective_cutoffs += 1
                break
        return value

    result.best_score = visit(root, True, -math.inf, math.inf, 0)
    return result


def minimax(root):
    """题目得分接口。"""
    return minimax_search(root).best_score


def alpha_beta(root):
    """题目正式接口：按 children 原始从左到右顺序，返回最佳得分。"""
    return alpha_beta_search(root).best_score


def independent_reference(raw):
    """迭代后序求值，独立于递归搜索；不读取 expected。"""
    stack = [(raw, 0, False)]
    scores = {}
    nodes = leaves = maximum_depth = 0
    while stack:
        node, depth, expanded = stack.pop()
        if not expanded:
            nodes += 1
            maximum_depth = max(maximum_depth, depth)
            if "value" in node:
                leaves += 1
                scores[id(node)] = node["value"]
            else:
                stack.append((node, depth, True))
                stack.extend((child, depth + 1, False) for child in reversed(node["children"]))
        else:
            values = [scores[id(child)] for child in node["children"]]
            scores[id(node)] = (max if depth % 2 == 0 else min)(values)
    return scores[id(raw)], nodes, leaves, maximum_depth


def validate(result, reference, algorithm):
    score, nodes, leaves, depth = reference
    assert result.best_score == score, (algorithm, result.best_score, score)
    assert 1 <= result.leaf_evaluations <= leaves
    assert result.leaf_evaluations <= result.visited_nodes <= nodes
    assert result.maximum_depth <= depth
    assert result.effective_cutoffs >= 0
    if algorithm == "Minimax":
        assert (result.visited_nodes, result.leaf_evaluations, result.maximum_depth) == (nodes, leaves, depth)
        assert result.effective_cutoffs == 0


def benchmark(cases, repeats=7, memory_repeats=3):
    records = []
    scheduling = random.Random(MEASUREMENT_SEED)
    for case in cases:
        reference = independent_reference(case["tree_structure"])
        assert reference[0] == case["expected"]["best_score"], case["id"]
        assert reference[1] == case["statistics"]["node_count"], case["id"]
        assert reference[2] == case["statistics"]["leaf_count"], case["id"]
        seed = int.from_bytes(hashlib.sha256(
            f"{ORDER_SEED}:{case['id']}".encode("utf-8")).digest()[:8], "big")
        root = from_json(case["tree_structure"])
        roots = prepare_orders(root, seed)
        tasks = {
            name: (minimax_search if name == "Minimax" else alpha_beta_search)
            for name in ALGORITHMS
        }
        baseline = {}
        timing = {name: [] for name in ALGORITHMS}
        memory = {name: [] for name in ALGORITHMS}
        for name in ALGORITHMS:
            baseline[name] = tasks[name](roots[name])  # 每个组合预热一次
            validate(baseline[name], reference, name)
        # 每轮打乱策略测量顺序，输入顺序本身保持不变。
        for _ in range(repeats):
            names = list(ALGORITHMS)
            scheduling.shuffle(names)
            for name in names:
                gc.collect()
                begin = time.perf_counter_ns()
                result = tasks[name](roots[name])
                elapsed = (time.perf_counter_ns() - begin) / 1_000_000
                timing[name].append(elapsed)
                validate(result, reference, name)
                assert result == baseline[name]
        for _ in range(memory_repeats):
            names = list(ALGORITHMS)
            scheduling.shuffle(names)
            for name in names:
                gc.collect()
                tracemalloc.start()
                try:
                    result = tasks[name](roots[name])
                    _, peak = tracemalloc.get_traced_memory()
                finally:
                    tracemalloc.stop()
                memory[name].append(peak)
                validate(result, reference, name)
                assert result == baseline[name]
        for name in ALGORITHMS:
            result = baseline[name]
            records.append({
                "case_id": case["id"], "kind": case["kind"], "algorithm": name,
                "input_seed": case["seed"], "random_order_seed": seed,
                "branching_factor": case["parameters"].get("branching_factor"),
                "input_depth": reference[3], "input_node_count": reference[1],
                "input_leaf_count": reference[2], "correct": True,
                **asdict(result),
                "node_saving_percent": 100 * (1 - result.visited_nodes / reference[1]),
                "time_median_ms": statistics.median(timing[name]),
                "memory_peak_median_kib": statistics.median(memory[name]) / 1024,
                "time_samples_ms": timing[name],
                "memory_samples_bytes": memory[name],
            })
        # 再次读取输入，核对构建、顺序准备与搜索均未改变原始树。
        assert from_json(case["tree_structure"]) == root
    return records


METRICS = (
    "visited_nodes", "leaf_evaluations", "node_saving_percent",
    "time_median_ms", "memory_peak_median_kib",
)


def aggregate(records):
    groups = {}
    for row in records:
        dimensions = [("overall", "all"), ("kind", row["kind"])]
        if row["kind"] == "regular":
            dimensions += [
                ("branching_factor", str(row["branching_factor"])),
                ("regular_depth", str(row["input_depth"])),
                ("regular_b_depth", f"b{row['branching_factor']}_d{row['input_depth']}"),
            ]
        for dimension, group in dimensions:
            groups.setdefault((dimension, group, row["algorithm"]), []).append(row)
    summary = []
    for (dimension, group, name), rows in groups.items():
        result = {
            "dimension": dimension, "group": group, "algorithm": name,
            "case_count": len(rows), "correct_count": sum(r["correct"] for r in rows),
        }
        for metric in METRICS:
            values = [r[metric] for r in rows]
            result[metric + "_mean"] = statistics.mean(values)
            result[metric + "_min"] = min(values)
            result[metric + "_max"] = max(values)
        result["effective_cutoffs_mean"] = statistics.mean(r["effective_cutoffs"] for r in rows)
        summary.append(result)
    return summary


def write_csv(path, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_results(summary, output, repeats=7):
    """一张分面图：行按分支数分组，列比较访问量、求值量和耗时。"""
    os.environ.setdefault("MPLCONFIGDIR", str(LAB2 / ".mplconfig"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FixedLocator

    lookup = {(r["group"], r["algorithm"]): r for r in summary
              if r["dimension"] == "regular_b_depth"}
    available = sorted({int(r["group"].split("_")[0][1:]) for r in summary
                        if r["dimension"] == "regular_b_depth"})
    if not available:
        raise ValueError("统计图需要至少一组 regular 规则树数据")
    columns = (
        ("visited_nodes_mean", "访问节点数"),
        ("leaf_evaluations_mean", "叶节点求值数"),
        ("time_median_ms_mean", "搜索耗时（ms）"),
    )
    colors = ("#2878B5", "#F28E2B", "#31965B", "#D75455")
    with plt.rc_context({"font.family": "sans-serif", "font.sans-serif": ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "DejaVu Sans"], "axes.unicode_minus": False, "font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False}):
        fig, axes = plt.subplots(len(available), 3, figsize=(12.8, 3 * len(available) + 1.0),
                                 squeeze=False)
        for ri, factor in enumerate(available):
            depths = sorted({int(group.split("_d")[1]) for group, _ in lookup
                             if group.startswith(f"b{factor}_")})
            for ci, (metric, title) in enumerate(columns):
                ax = axes[ri, ci]
                for name, color in zip(ALGORITHMS, colors):
                    values = [lookup[(f"b{factor}_d{d}", name)][metric] for d in depths]
                    ax.plot(depths, values, marker="o", markersize=5, color=color,
                            linewidth=1.7, label={"Minimax": "Minimax", "AB original": "Alpha-Beta：原始顺序", "AB reverse": "Alpha-Beta：反向顺序", "AB random": "Alpha-Beta：随机顺序"}[name])
                ax.set_yscale("log")
                ax.xaxis.set_major_locator(FixedLocator(depths))
                ax.set_title(f"b = {factor} | {title}")
                ax.set_xlabel("树深度（边数）")
                ax.grid(alpha=0.20, which="major")
                if ci == 0:
                    ax.set_ylabel("数量（对数刻度）")
                elif ci == 2:
                    ax.set_ylabel("耗时 / ms（对数刻度）")
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, .955),
                   ncol=4, frameon=False)
        fig.suptitle("Minimax 与 Alpha-Beta 对照｜深度与子节点访问顺序的影响", fontsize=14, y=.993)
        fig.text(.5, .018,
                 "仅展示规则树；每个点为 3 个固定种子的算术平均。"
                 f"耗时先取单树 {repeats} 次运行的中位数；数值越小越好。\n"
                 "纵轴均使用对数刻度；测量不含建树及访问顺序准备；"
                 "全部 40 棵树的结果见表格。",
                 ha="center", fontsize=9)
        fig.tight_layout(rect=(0, .07, 1, .915), h_pad=1.6, w_pad=2)
        for extension in ("png", "svg"):
            fig.savefig(output / f"performance_comparison_v3.0.{extension}", dpi=220)
        plt.close(fig)
    return matplotlib.__version__


def write_table(summary, output):
    rows = [r for r in summary if r["dimension"] == "overall"]
    by_name = {r["algorithm"]: r for r in rows}
    lines = [
        "# Quiz3：Minimax 与 Alpha-Beta 对照",
        "",
        "各指标逐树计算后等权平均；耗时、内存先取单树重复测量中位数。",
        "节点节省率相对于同树完整 Minimax。剪枝次数保留在明细表中。",
        "",
        "| 算法 | 正确组合 | 访问节点数 | 叶节点求值数 | 节点节省率/% | 耗时/ms | 峰值内存/KiB |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name in ALGORITHMS:
        r = by_name[name]
        lines.append(
            f"| {name} | {r['correct_count']}/{r['case_count']} "
            f"| {r['visited_nodes_mean']:.1f} | {r['leaf_evaluations_mean']:.1f} "
            f"| {r['node_saving_percent_mean']:.2f} | {r['time_median_ms_mean']:.4f} "
            f"| {r['memory_peak_median_kib_mean']:.2f} |")
    lines += [
        "", "计时含递归搜索和统一计数，不含读取、建树、顺序准备与校验。",
        "随机顺序逐树固定，各次重复使用同一顺序。内存为 tracemalloc 搜索新增分配峰值，",
        "不含预载树及预先准备的访问顺序，并非进程总内存。",
        "图中仅使用规则树，按分支数、深度分别汇总，避免把不同规模混为一条趋势。",
    ]
    (output / "comparison_v3.0.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tree", type=Path, default=LAB2 / "code/quiz3/tree")
    parser.add_argument("--output", type=Path, default=LAB2 / "result/quiz3")
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--memory-repeats", type=int, default=3)
    args = parser.parse_args()
    if min(args.repeats, args.memory_repeats) < 1:
        parser.error("重复次数应大于 0")
    raw = args.tree.read_bytes()
    cases = json.loads(raw.decode("utf-8"))["cases"]
    if not cases:
        parser.error("输入 cases 不能为空")
    records = benchmark(cases, args.repeats, args.memory_repeats)
    summary = aggregate(records)
    args.output.mkdir(parents=True, exist_ok=True)
    detail = [{k: v for k, v in row.items()
               if k not in ("time_samples_ms", "memory_samples_bytes")} for row in records]
    write_csv(args.output / "details_v3.0.csv", detail)
    write_csv(args.output / "summary_v3.0.csv", summary)
    write_table(summary, args.output)
    mpl_version = plot_results(summary, args.output, args.repeats)
    metadata = {
        "created_at": datetime.now().astimezone().isoformat(),
        "python": sys.version, "platform": platform.platform(),
        "matplotlib": mpl_version, "tree_path": str(args.tree.resolve()),
        "tree_sha256": hashlib.sha256(raw).hexdigest(), "case_count": len(cases),
        "repeats": args.repeats, "memory_repeats": args.memory_repeats,
        "random_order_base_seed": ORDER_SEED, "measurement_seed": MEASUREMENT_SEED,
        "gc_enabled": gc.isenabled(), "timer": "perf_counter_ns",
        "memory": "tracemalloc search-only Python allocation peak; preloaded trees excluded",
        "order_preparation": "outside measurement; per-case SHA256-derived seed; fixed across repeats",
        "correctness": "independent iterative postorder oracle + dataset reference; every measured run checked",
        "counting": "recursive calls include root/leaves; cutoffs count only skipped remaining children",
        "aggregation": "equal-weight mean of per-tree values; savings computed per tree first",
        "plot": "regular trees only; rows=branching factor; columns=nodes/leaves/time; log y",
        "command": [sys.executable] + sys.argv,
    }
    (args.output / "results_v3.0.json").write_text(json.dumps({
        "metadata": metadata, "records": records, "summary": summary,
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"All {len(records)} combinations passed. Results: {args.output.resolve()}")


if __name__ == "__main__":
    main()
