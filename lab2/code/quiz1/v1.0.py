"""读取同目录的 map，评测五种搜索算法，数据和图输出至 lab2/result/quiz1。

激活 conda 环境 ai-lab2 后，使用 python 运行本文件。
可选参数：--map 路径 --output 路径 --repeats 7 --memory-repeats 3。
"""

import argparse
import csv
import gc
import hashlib
import heapq
import json
import math
import os
import platform
import random
import statistics
import sys
import time
import tracemalloc
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


LAB2 = Path(__file__).resolve().parents[2]
ALGORITHMS = ("BFS", "Dijkstra", "A* Manhattan", "A* Euclidean", "A* Chebyshev")
DIRECTIONS = ((-1, 0), (1, 0), (0, -1), (0, 1))


@dataclass
class SearchResult:
    path: list
    expanded_nodes: int
    popped_entries: int
    frontier_peak: int


def manhattan_distance(position, goal):
    return abs(position[0] - goal[0]) + abs(position[1] - goal[1])


def heuristic(position, goal, algorithm):
    dr, dc = abs(position[0] - goal[0]), abs(position[1] - goal[1])
    if algorithm == "A* Manhattan":
        return dr + dc
    if algorithm == "A* Euclidean":
        return math.hypot(dr, dc)
    if algorithm == "A* Chebyshev":
        return max(dr, dc)
    return 0


def neighbors(position, grid):
    row, col = position
    for dr, dc in DIRECTIONS:
        nr, nc = row + dr, col + dc
        if 0 <= nr < len(grid) and 0 <= nc < len(grid[0]) and grid[nr][nc] != "X":
            yield nr, nc


def reconstruct(parent, end):
    path = [end]
    while path[-1] in parent:
        path.append(parent[path[-1]])
    path.reverse()
    return path


def search(start, end, grid, algorithm):
    """扩展节点：有效非终点节点开始遍历邻居的次数。

    三种 A* 与 Dijkstra 统一按 (f, g, 坐标) 排序，沿用原程序的平局规则。
    BFS 使用 FIFO 队列，各算法都按上下左右顺序生成邻居。
    """
    if algorithm not in ALGORITHMS:
        raise ValueError(f"Unknown algorithm: {algorithm}")
    if not grid or not grid[0]:
        return SearchResult([], 0, 0, 0)
    if any(len(row) != len(grid[0]) for row in grid):
        raise ValueError("地图必须是矩形")
    for row, col in (start, end):
        if not (0 <= row < len(grid) and 0 <= col < len(grid[0])) or grid[row][col] == "X":
            return SearchResult([], 0, 0, 0)

    parent = {}
    expanded = popped = 0
    peak = 1
    if algorithm == "BFS":
        frontier = deque([start])
        visited = {start}
        while frontier:
            current = frontier.popleft()
            popped += 1
            if current == end:
                return SearchResult(reconstruct(parent, end), expanded, popped, peak)
            expanded += 1
            for neighbor in neighbors(current, grid):
                if neighbor not in visited:
                    visited.add(neighbor)
                    parent[neighbor] = current
                    frontier.append(neighbor)
            peak = max(peak, len(frontier))
    else:
        frontier = [(heuristic(start, end, algorithm), 0, start)]
        g_score = {start: 0}
        while frontier:
            _, cost, current = heapq.heappop(frontier)
            popped += 1
            if cost != g_score[current]:
                continue
            if current == end:
                return SearchResult(reconstruct(parent, end), expanded, popped, peak)
            expanded += 1
            for neighbor in neighbors(current, grid):
                new_cost = cost + 1
                if new_cost < g_score.get(neighbor, math.inf):
                    g_score[neighbor] = new_cost
                    parent[neighbor] = current
                    priority = new_cost + heuristic(neighbor, end, algorithm)
                    heapq.heappush(frontier, (priority, new_cost, neighbor))
            peak = max(peak, len(frontier))
    return SearchResult([], expanded, popped, peak)


def a_star(start, end, grid):
    """保留原来的曼哈顿 A* 路径接口。"""
    return search(start, end, grid, "A* Manhattan").path


def reference_distance(grid, start, end):
    """独立参考 BFS，不调用被测算法的邻居或路径恢复函数。"""
    distances = {start: 0}
    queue = deque([start])
    while queue:
        row, col = queue.popleft()
        if (row, col) == end:
            return distances[(row, col)]
        for nr, nc in ((row - 1, col), (row + 1, col), (row, col - 1), (row, col + 1)):
            if 0 <= nr < len(grid) and 0 <= nc < len(grid[0]):
                if grid[nr][nc] != "X" and (nr, nc) not in distances:
                    distances[(nr, nc)] = distances[(row, col)] + 1
                    queue.append((nr, nc))
    return None


def validate(result, grid, start, end, expected):
    path = result.path
    if expected is None:
        if path:
            raise AssertionError("无解地图返回了非空路径")
        return
    if not path or path[0] != start or path[-1] != end:
        raise AssertionError("路径端点错误")
    if len(path) - 1 != expected:
        raise AssertionError("路径长度不等于 BFS 参考距离")
    for row, col in path:
        if not (0 <= row < len(grid) and 0 <= col < len(grid[0])) or grid[row][col] == "X":
            raise AssertionError("路径越界或穿越障碍")
    if any(manhattan_distance(a, b) != 1 for a, b in zip(path, path[1:])):
        raise AssertionError("路径包含非法移动")


def benchmark(cases, repeats, memory_repeats):
    records = []
    rng = random.Random(20261109)
    for index, case in enumerate(cases, 1):
        grid = [list(row) for row in case["grid"]]
        original = [row[:] for row in grid]
        start, end = tuple(case["start"]), tuple(case["end"])
        expected = reference_distance(grid, start, end)
        if case["expected"] != {"reachable": expected is not None, "shortest_steps": expected}:
            raise AssertionError(f"{case['id']}: map 的参考答案不一致")
        baseline, times, peaks = {}, {}, {}
        for algorithm in ALGORITHMS:
            baseline[algorithm] = search(start, end, grid, algorithm)  # 预热与首次校验
            validate(baseline[algorithm], grid, start, end, expected)
            times[algorithm], peaks[algorithm] = [], []

        for phase, count in (("time", repeats), ("memory", memory_repeats)):
            for _ in range(count):
                order = list(ALGORITHMS)
                rng.shuffle(order)
                for algorithm in order:
                    gc.collect()  # 位于计时区间外，测量期间保留默认 GC 行为
                    if phase == "time":
                        begin = time.perf_counter_ns()
                        result = search(start, end, grid, algorithm)
                        times[algorithm].append((time.perf_counter_ns() - begin) / 1e6)
                    else:
                        tracemalloc.start()
                        try:
                            result = search(start, end, grid, algorithm)
                            peaks[algorithm].append(tracemalloc.get_traced_memory()[1])
                        finally:
                            tracemalloc.stop()
                    # 校验在测量区间外，包括每次结果的稳定性及输入未修改。
                    validate(result, grid, start, end, expected)
                    if result != baseline[algorithm] or grid != original:
                        raise AssertionError(f"{case['id']} / {algorithm}: 结果不稳定或输入被修改")

        for algorithm in ALGORITHMS:
            result = baseline[algorithm]
            records.append({
                "case_id": case["id"], "size": f"{case['rows']}x{case['cols']}",
                "kind": case["kind"],
                "target_density": case["parameters"].get("target_obstacle_density", ""),
                "actual_density": case["actual_obstacle_density"],
                "seed": case["parameters"].get("seed", ""),
                "reachable": expected is not None, "algorithm": algorithm, "correct": True,
                "expected_steps": expected, "path_steps": len(result.path) - 1 if result.path else None,
                "expanded_nodes": result.expanded_nodes, "popped_entries": result.popped_entries,
                "frontier_peak": result.frontier_peak,
                "time_median_ms": statistics.median(times[algorithm]),
                "time_min_ms": min(times[algorithm]), "time_max_ms": max(times[algorithm]),
                "memory_peak_median_kib": statistics.median(peaks[algorithm]) / 1024,
                "memory_peak_max_kib": max(peaks[algorithm]) / 1024,
                "time_samples_ms": times[algorithm], "memory_samples_bytes": peaks[algorithm],
                "path": result.path,
            })
        print(f"[{index:02d}/{len(cases)}] {case['id']}: five algorithms passed", flush=True)
    return records


def aggregate(records):
    groups = {}
    for record in records:
        keys = [("overall", "all"), ("size", record["size"]), ("kind", record["kind"]),
                ("reachable", str(record["reachable"])),
                ("size_kind", f"{record['size']}/{record['kind']}"),
                ("size_reachable", f"{record['size']}/{record['reachable']}")]
        if record["kind"] == "random":
            keys.extend([("random_density", str(record["target_density"])),
                         ("size_random_density", f"{record['size']}/{record['target_density']}")])
        for dimension, group in keys:
            groups.setdefault((dimension, group, record["algorithm"]), []).append(record)
    return [{
        "dimension": dimension, "group": group, "algorithm": algorithm,
        "case_count": len(rows), "correct_count": sum(r["correct"] for r in rows),
        "reachable_count": sum(r["reachable"] for r in rows),
        "expanded_nodes_mean": statistics.mean(r["expanded_nodes"] for r in rows),
        "time_median_ms_mean": statistics.mean(r["time_median_ms"] for r in rows),
        "memory_peak_median_kib_mean": statistics.mean(r["memory_peak_median_kib"] for r in rows),
    } for (dimension, group, algorithm), rows in groups.items()]


def write_csv(path, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_results(summary, output):
    """Compare overall performance with three horizontal bar panels."""
    os.environ.setdefault("MPLCONFIGDIR", str(LAB2 / ".mplconfig"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    overall = {row["algorithm"]: row for row in summary if row["dimension"] == "overall"}
    metrics = (
        ("expanded_nodes_mean", "扩展节点数", ".1f"),
        ("time_median_ms_mean", "耗时（ms）", ".3f"),
        ("memory_peak_median_kib_mean", "峰值内存（KiB）", ".1f"),
    )
    colors = ("#2878B5", "#F28E2B", "#31965B", "#D75455", "#9169B5")
    with plt.rc_context({"font.family": "sans-serif", "font.sans-serif": ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "DejaVu Sans"], "axes.unicode_minus": False, "font.size": 11, "axes.spines.top": False, "axes.spines.right": False}):
        fig, axes = plt.subplots(1, 3, figsize=(12, 4.2), sharey=True)
        for ax, (metric, title, number_format) in zip(axes, metrics):
            values = [overall[name][metric] for name in ALGORITHMS]
            best = min(values)
            bars = ax.barh(range(len(ALGORITHMS)), values, color=colors, height=0.60)
            limit = max(values) or 1
            for bar, value in zip(bars, values):
                winner = math.isclose(value, best, rel_tol=1e-12, abs_tol=1e-12)
                if winner:
                    bar.set_edgecolor("#172B40")
                    bar.set_linewidth(1.8)
                label = format(value, number_format) + (" *" if winner else "")
                ax.text(value + limit * 0.025, bar.get_y() + bar.get_height()/2,
                        label, va="center", fontsize=10, fontweight="bold" if winner else "normal")
            ax.set_title(title, fontsize=12, pad=12)
            ax.set_xlim(0, limit * 1.32)
            ax.set_yticks(range(len(ALGORITHMS)), ALGORITHMS)
            ax.tick_params(axis="y", length=0)
            ax.tick_params(axis="x", labelsize=9)
            ax.grid(axis="x", alpha=0.18)
            ax.set_axisbelow(True)
        axes[0].invert_yaxis()
        count = next(iter(overall.values()))["case_count"]
        fig.suptitle("算法性能对照｜数值越小越好（* 表示最优）", fontsize=14, y=0.98)
        fig.text(0.5, 0.035, f"{count} 张地图等权平均；耗时与内存先取单图中位数，再跨图取算术平均。",
                 ha="center", fontsize=9)
        fig.tight_layout(rect=(0, 0.08, 1, 0.93), w_pad=1.8)
        for extension in ("png", "svg"):
            fig.savefig(output / f"performance_comparison_v1.0.{extension}", dpi=220)
        plt.close(fig)
    return matplotlib.__version__


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--map", type=Path, default=Path(__file__).with_name("map"))
    parser.add_argument("--output", type=Path, default=LAB2 / "result" / "quiz1")
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--memory-repeats", type=int, default=3)
    args = parser.parse_args()
    if args.repeats < 1 or args.memory_repeats < 1:
        parser.error("重复次数必须大于 0")
    raw = args.map.read_bytes()
    cases = json.loads(raw.decode("utf-8"))["cases"]
    args.output.mkdir(parents=True, exist_ok=True)
    records = benchmark(cases, args.repeats, args.memory_repeats)
    summary = aggregate(records)
    detail = [{k: v for k, v in row.items() if k not in ("path", "time_samples_ms", "memory_samples_bytes")}
              for row in records]
    write_csv(args.output / "details_v1.0.csv", detail)
    write_csv(args.output / "summary_v1.0.csv", summary)
    matplotlib_version = plot_results(summary, args.output)
    metadata = {
        "created_at": datetime.now().astimezone().isoformat(),
        "python": sys.version, "platform": platform.platform(), "processor": platform.processor(),
        "matplotlib": matplotlib_version, "map_path": str(args.map.resolve()),
        "map_sha256": hashlib.sha256(raw).hexdigest(),
        "case_count": len(cases), "repeats": args.repeats, "memory_repeats": args.memory_repeats,
        "measurement_order_seed": 20261109, "gc_enabled": gc.isenabled(),
        "timer": "perf_counter_ns", "memory": "tracemalloc Python allocation peak",
        "command": [sys.executable] + sys.argv,
    }
    (args.output / "results_v1.0.json").write_text(json.dumps(
        {"metadata": metadata, "records": records, "summary": summary}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"All {len(records)} combinations passed. Results: {args.output.resolve()}")


if __name__ == "__main__":
    main()
