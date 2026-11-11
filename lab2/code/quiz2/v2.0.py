"""Quiz2 加权地形：Dijkstra、修正曼哈顿 A*、最小权重缩放 A*。

在 lab2 目录运行：python code/quiz2/v2.0.py。
默认输入为脚本旁的 map，输出至 result/quiz2，文件名带 _v2.0。
进入终点免费，初始起点不收费；再次进入 S 使用 start_terrain_cost。
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
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


LAB2 = Path(__file__).resolve().parents[2]
ALGORITHMS = ("Dijkstra", "A* corrected", "A* scaled")
DIRECTIONS = ((-1, 0), (1, 0), (0, -1), (0, 1))


@dataclass
class SearchResult:
    minimum_cost: int
    path: list
    expanded_nodes: int
    popped_entries: int
    frontier_peak: int
    min_non_goal_cost: int = 0
    heuristic_scale: int = 0


def prepare_terrain(start, end, grid, start_terrain_cost):
    """各算法统一转换代价矩阵并扫描非终点最小权重，计入搜索测量。"""
    if not grid:
        return None, 0
    cols = len(grid[0])
    if any(len(row) != cols for row in grid):
        raise ValueError("地图必须是矩形")
    for row, col in (start, end):
        if not (0 <= row < len(grid) and 0 <= col < cols) or grid[row][col] == "X":
            return None, 0
    if start_terrain_cost < 0:
        raise ValueError("地形权重必须非负")
    costs, minimum = [], math.inf
    for r, row in enumerate(grid):
        converted = []
        for c, value in enumerate(row):
            if value == "X":
                converted.append(-1)
                continue
            cost = 0 if (r, c) == end else start_terrain_cost if value == "S" else int(value)
            if cost < 0:
                raise ValueError("地形权重必须非负")
            converted.append(cost)
            if (r, c) != end:
                minimum = min(minimum, cost)
        costs.append(converted)
    return costs, 0 if minimum == math.inf else minimum


def heuristic(position, end, scale):
    if scale == 0:
        return 0
    distance = abs(position[0] - end[0]) + abs(position[1] - end[1])
    return scale * max(distance - 1, 0)


def search(start, end, grid, algorithm, start_terrain_cost=1):
    """按 (f,g,坐标) 排序，四方向松弛；终点有效出堆时返回最小代价。

    corrected：非终点最小权重至少为 1 时 h=max(d-1,0)，存在零权时 h=0。
    scaled：h=c_min*max(d-1,0)，c_min=0 时自然退化为 Dijkstra。
    """
    if algorithm not in ALGORITHMS:
        raise ValueError(f"Unknown algorithm: {algorithm}")
    costs, minimum = prepare_terrain(start, end, grid, start_terrain_cost)
    if costs is None:
        return SearchResult(-1, [], 0, 0, 0)
    scale = 0
    if algorithm == "A* corrected":
        scale = 1 if minimum >= 1 else 0
    elif algorithm == "A* scaled":
        scale = minimum
    frontier = [(heuristic(start, end, scale), 0, start)]
    distance, parent = {start: 0}, {}
    expanded, popped, peak = 0, 0, 1
    while frontier:
        _, current_cost, current = heapq.heappop(frontier)
        popped += 1
        if current_cost != distance[current]:
            continue
        if current == end:
            path = [current]
            while path[-1] in parent:
                path.append(parent[path[-1]])
            path.reverse()
            return SearchResult(current_cost, path, expanded, popped, peak, minimum, scale)
        expanded += 1
        row, col = current
        for dr, dc in DIRECTIONS:
            nr, nc = row + dr, col + dc
            if not (0 <= nr < len(costs) and 0 <= nc < len(costs[0])) or costs[nr][nc] < 0:
                continue
            neighbor = (nr, nc)
            candidate = current_cost + costs[nr][nc]
            if candidate < distance.get(neighbor, math.inf):
                distance[neighbor] = candidate
                parent[neighbor] = current
                heapq.heappush(frontier, (candidate + heuristic(neighbor, end, scale), candidate, neighbor))
        peak = max(peak, len(frontier))
    return SearchResult(-1, [], expanded, popped, peak, minimum, scale)


def a_star(start, end, grid, start_terrain_cost=1):
    """题目接口：返回最小总代价，无解返回 -1，默认使用最小权重缩放启发。"""
    return search(start, end, grid, "A* scaled", start_terrain_cost).minimum_cost


def reference_cost(grid, start, end, start_terrain_cost):
    """独立反向 Dijkstra：反向 v→u 的代价为正向进入 v 的代价。"""
    distances, queue = {end: 0}, [(0, end)]
    while queue:
        total, position = heapq.heappop(queue)
        if total != distances[position]:
            continue
        if position == start:
            return total
        row, col = position
        value = grid[row][col]
        step = 0 if position == end else start_terrain_cost if value == "S" else int(value)
        for previous in ((row-1,col), (row+1,col), (row,col-1), (row,col+1)):
            r, c = previous
            if 0 <= r < len(grid) and 0 <= c < len(grid[0]) and grid[r][c] != "X":
                candidate = total + step
                if candidate < distances.get(previous, math.inf):
                    distances[previous] = candidate
                    heapq.heappush(queue, (candidate, previous))
    return -1


def validate(result, grid, start, end, start_terrain_cost, expected):
    if result.minimum_cost != expected:
        raise AssertionError("返回代价与独立参考结果不同")
    path = result.path
    if expected == -1:
        if path:
            raise AssertionError("无解地图返回了路径")
        return
    if not path or path[0] != start or path[-1] != end:
        raise AssertionError("路径端点错误")
    if len(path) != len(set(path)):
        raise AssertionError("路径包含循环")
    cost = 0
    for index, (row, col) in enumerate(path):
        if not (0 <= row < len(grid) and 0 <= col < len(grid[0])) or grid[row][col] == "X":
            raise AssertionError("路径越界或经过障碍")
        if index:
            previous = path[index-1]
            if abs(previous[0]-row) + abs(previous[1]-col) != 1:
                raise AssertionError("路径不是四方向连续移动")
            if (row,col) != end:
                cost += start_terrain_cost if grid[row][col] == "S" else int(grid[row][col])
    if cost != expected:
        raise AssertionError("路径累计进入代价不等于最优值")


def benchmark(cases, repeats, memory_repeats):
    records = []
    rng = random.Random(20261109)
    for index, case in enumerate(cases, 1):
        grid = [list(row) for row in case["grid"]]
        original = [row[:] for row in grid]
        start, end = tuple(case["start"]), tuple(case["end"])
        start_cost = case["start_terrain_cost"]
        expected = reference_cost(grid, start, end, start_cost)
        if expected != case["expected"]["minimum_cost"] or (expected != -1) != case["expected"]["reachable"]:
            raise AssertionError(f"{case['id']}: map 参考答案不一致")
        baseline, times, peaks = {}, {}, {}
        for algorithm in ALGORITHMS:
            baseline[algorithm] = search(start, end, grid, algorithm, start_cost)
            validate(baseline[algorithm], grid, start, end, start_cost, expected)
            if baseline[algorithm].min_non_goal_cost != case["min_non_goal_cost"]:
                raise AssertionError("从输入计算的最小权重与地图记录不一致")
            times[algorithm], peaks[algorithm] = [], []
        for phase, count in (("time", repeats), ("memory", memory_repeats)):
            for _ in range(count):
                order = list(ALGORITHMS)
                rng.shuffle(order)
                for algorithm in order:
                    result = None
                    gc.collect()
                    if phase == "time":
                        begin = time.perf_counter_ns()
                        result = search(start, end, grid, algorithm, start_cost)
                        times[algorithm].append((time.perf_counter_ns()-begin)/1e6)
                    else:
                        tracemalloc.start()
                        try:
                            result = search(start, end, grid, algorithm, start_cost)
                            peaks[algorithm].append(tracemalloc.get_traced_memory()[1])
                        finally:
                            tracemalloc.stop()
                    validate(result, grid, start, end, start_cost, expected)
                    if result != baseline[algorithm] or grid != original:
                        raise AssertionError(f"{case['id']} / {algorithm}: 结果不稳定或输入被修改")
        for algorithm in ALGORITHMS:
            result = baseline[algorithm]
            records.append({
                "case_id": case["id"], "size": f"{case['rows']}x{case['cols']}", "kind": case["kind"],
                "terrain_weights": "/".join(map(str, case["parameters"].get("terrain_weights", []))),
                "target_density": case["parameters"].get("target_obstacle_density", ""),
                "actual_density": case["actual_obstacle_density"], "seed": case["parameters"].get("seed", ""),
                "start_terrain_cost": start_cost, "min_non_goal_cost": result.min_non_goal_cost,
                "heuristic_scale": result.heuristic_scale, "reachable": expected != -1,
                "algorithm": algorithm, "correct": True, "expected_cost": expected,
                "minimum_cost": result.minimum_cost, "path_steps": len(result.path)-1 if result.path else None,
                "reference_min_cost_steps": case["expected"]["steps_among_minimum_cost_paths"],
                "shortest_steps_ignoring_weights": case["expected"]["shortest_steps_ignoring_weights"],
                "expanded_nodes": result.expanded_nodes, "popped_entries": result.popped_entries,
                "frontier_peak": result.frontier_peak,
                "time_median_ms": statistics.median(times[algorithm]),
                "time_min_ms": min(times[algorithm]), "time_max_ms": max(times[algorithm]),
                "memory_peak_median_kib": statistics.median(peaks[algorithm])/1024,
                "memory_peak_max_kib": max(peaks[algorithm])/1024,
                "time_samples_ms": times[algorithm], "memory_samples_bytes": peaks[algorithm], "path": result.path,
            })
        print(f"[{index:03d}/{len(cases)}] {case['id']}: three algorithms passed", flush=True)
    return records


def aggregate(records):
    groups = {}
    for record in records:
        keys = [("overall", "all"), ("size", record["size"]), ("kind", record["kind"]),
                ("reachable", str(record["reachable"])), ("min_cost", str(record["min_non_goal_cost"])),
                ("size_min_cost", f"{record['size']}/{record['min_non_goal_cost']}"),
                ("size_kind", f"{record['size']}/{record['kind']}")]
        if record["kind"] == "random":
            keys.extend([("random_density", str(record["target_density"])),
                         ("terrain_weights", record["terrain_weights"]),
                         ("density_min_cost", f"{record['target_density']}/{record['min_non_goal_cost']}")])
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
            fig.savefig(output / f"performance_comparison_v2.0.{extension}", dpi=220)
        plt.close(fig)
    return matplotlib.__version__


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--map", type=Path, default=Path(__file__).with_name("map"))
    parser.add_argument("--output", type=Path, default=LAB2 / "result" / "quiz2")
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
    write_csv(args.output / "details_v2.0.csv", detail)
    write_csv(args.output / "summary_v2.0.csv", summary)
    matplotlib_version = plot_results(summary, args.output)
    metadata = {
        "created_at": datetime.now().astimezone().isoformat(),
        "python": sys.version, "platform": platform.platform(), "processor": platform.processor(),
        "matplotlib": matplotlib_version, "map_path": str(args.map.resolve()),
        "map_sha256": hashlib.sha256(raw).hexdigest(),
        "strategy_version": "v2.0",
        "heuristics": {"Dijkstra": "0", "A* corrected": "max(d-1,0), or 0 if a non-goal terrain costs zero", "A* scaled": "c_min*max(d-1,0)"},
        "preprocessing": "All algorithms convert terrain and scan minimum non-goal weight inside measurement",
        "case_count": len(cases), "repeats": args.repeats, "memory_repeats": args.memory_repeats,
        "measurement_order_seed": 20261109, "gc_enabled": gc.isenabled(),
        "timer": "perf_counter_ns", "memory": "tracemalloc Python allocation peak",
        "command": [sys.executable] + sys.argv,
    }
    (args.output / "results_v2.0.json").write_text(json.dumps(
        {"metadata": metadata, "records": records, "summary": summary}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"All {len(records)} combinations passed. Results: {args.output.resolve()}")


if __name__ == "__main__":
    main()
