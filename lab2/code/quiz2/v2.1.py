"""Quiz2 地标拓展：Dijkstra、缩放 A*、2 地标 A*、4 地标 A*。

固定起终点的单次查询，分别测量预处理、查询及总成本；每次重建索引。
在 lab2 目录运行 python code/quiz2/v2.1.py，输出 result/quiz2/*_v2.1.*。
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
ALGORITHMS = ("Dijkstra", "A* scaled", "ALT A* (2)", "ALT A* (4)")


@dataclass
class SearchResult:
    minimum_cost: int
    path: list
    expanded_nodes: int
    popped_entries: int
    frontier_peak: int
    min_non_goal_cost: int = 0
    heuristic_scale: int = 0
    landmarks: tuple = ()
    preprocessing_expanded_nodes: int = 0


@dataclass
class Prepared:
    rows: int
    cols: int
    costs: list
    start: int
    end: int
    minimum: int
    scale: int
    landmarks: tuple
    tables: list
    preprocessing_expanded_nodes: int


def adjacent(node, rows, cols):
    row, col = divmod(node, cols)
    if row:
        yield node - cols
    if row + 1 < rows:
        yield node + cols
    if col:
        yield node - 1
    if col + 1 < cols:
        yield node + 1


def select_landmarks(costs, rows, cols, count):
    """仅依据可通行位置和四个内部锚点，不按查询起终点挑选。

    选点顺序为左上、右下、右上、左下的四分位附近；2 点为 4 点的前缀。
    锚点遇障碍时取最近的未选可通行格，同距按坐标排序。
    """
    vertices = [i for i, cost in enumerate(costs) if cost >= 0]
    anchors = (((rows-1)//4, (cols-1)//4),
               (3*(rows-1)//4, 3*(cols-1)//4),
               ((rows-1)//4, 3*(cols-1)//4),
               (3*(rows-1)//4, (cols-1)//4))
    selected = []
    for row, col in anchors[:count]:
        if len(selected) == len(vertices):
            break
        selected.append(min((i for i in vertices if i not in selected),
                            key=lambda i: (abs(i//cols-row)+abs(i%cols-col), i)))
    return tuple(selected)


def all_distances(costs, rows, cols, origin, reverse=False):
    """正向边 u→v 收取 costs[v]；反向边 v→u 收取 costs[v]。"""
    distances = [math.inf] * len(costs)
    distances[origin] = 0
    queue = [(0, origin)]
    expanded = 0
    while queue:
        total, current = heapq.heappop(queue)
        if total != distances[current]:
            continue
        expanded += 1
        for neighbor in adjacent(current, rows, cols):
            if costs[neighbor] < 0:
                continue
            step = costs[current] if reverse else costs[neighbor]
            candidate = total + step
            if candidate < distances[neighbor]:
                distances[neighbor] = candidate
                heapq.heappush(queue, (candidate, neighbor))
    return distances, expanded


def prepare(start, end, grid, algorithm, start_terrain_cost=1):
    """转换地形、扫描最小权重，地标组再独立执行每个地标的两次 Dijkstra。"""
    if algorithm not in ALGORITHMS:
        raise ValueError(f"Unknown algorithm: {algorithm}")
    rows, cols = len(grid), len(grid[0]) if grid else 0
    if any(len(row) != cols for row in grid):
        raise ValueError("地图必须为矩形")
    for r, c in (start, end):
        if not (0 <= r < rows and 0 <= c < cols) or grid[r][c] == "X":
            return Prepared(rows, cols, [], 0, 0, 0, 0, (), [], 0)
    if start_terrain_cost < 0:
        raise ValueError("地形权重必须非负")
    costs, minimum = [], math.inf
    for r, row in enumerate(grid):
        for c, value in enumerate(row):
            if value == "X":
                costs.append(-1)
                continue
            cost = 0 if (r,c) == end else start_terrain_cost if value == "S" else int(value)
            if cost < 0:
                raise ValueError("地形权重必须非负")
            costs.append(cost)
            if (r,c) != end:
                minimum = min(minimum, cost)
    minimum = 0 if minimum == math.inf else minimum
    scale = 0 if algorithm == "Dijkstra" else minimum
    count = 2 if algorithm == "ALT A* (2)" else 4 if algorithm == "ALT A* (4)" else 0
    landmarks = select_landmarks(costs, rows, cols, count) if count else ()
    tables, expanded = [], 0
    for landmark in landmarks:
        forward, count_f = all_distances(costs, rows, cols, landmark)
        reverse, count_b = all_distances(costs, rows, cols, landmark, reverse=True)
        tables.append((forward, reverse))
        expanded += count_f + count_b
    return Prepared(rows, cols, costs, start[0]*cols+start[1], end[0]*cols+end[1],
                    minimum, scale, landmarks, tables, expanded)


def heuristic(node, data):
    if data.scale:
        distance = abs(node//data.cols-data.end//data.cols) + abs(node%data.cols-data.end%data.cols)
        estimate = data.scale * max(distance-1, 0)
    else:
        estimate = 0
    # 有向图 ALT；只对均有限的距离作差，避免 inf-inf。
    for forward, reverse in data.tables:
        if forward[data.end] != math.inf and forward[node] != math.inf:
            estimate = max(estimate, forward[data.end]-forward[node])
        if reverse[node] != math.inf and reverse[data.end] != math.inf:
            estimate = max(estimate, reverse[node]-reverse[data.end])
    return estimate


def query(data):
    """仅查询阶段；索引在调用期间保留，搜索不读取参考答案。"""
    if not data.costs:
        return SearchResult(-1, [], 0, 0, 0)
    distances = [math.inf] * len(data.costs)
    parents = [-1] * len(data.costs)
    distances[data.start] = 0
    queue = [(heuristic(data.start, data), 0, data.start)]
    expanded, popped, peak = 0, 0, 1
    points = tuple(divmod(i, data.cols) for i in data.landmarks)
    while queue:
        _, total, current = heapq.heappop(queue)
        popped += 1
        if total != distances[current]:
            continue
        if current == data.end:
            path = [current]
            while parents[path[-1]] != -1:
                path.append(parents[path[-1]])
            path.reverse()
            return SearchResult(total, [divmod(i,data.cols) for i in path], expanded, popped, peak,
                                data.minimum, data.scale, points, data.preprocessing_expanded_nodes)
        expanded += 1
        for neighbor in adjacent(current, data.rows, data.cols):
            if data.costs[neighbor] < 0:
                continue
            candidate = total + data.costs[neighbor]
            if candidate < distances[neighbor]:
                distances[neighbor] = candidate
                parents[neighbor] = current
                heapq.heappush(queue, (candidate+heuristic(neighbor,data), candidate, neighbor))
        peak = max(peak, len(queue))
    return SearchResult(-1, [], expanded, popped, peak, data.minimum, data.scale, points,
                        data.preprocessing_expanded_nodes)


def search(start, end, grid, algorithm, start_terrain_cost=1):
    return query(prepare(start,end,grid,algorithm,start_terrain_cost))


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


def timed_run(start, end, grid, algorithm, start_cost):
    begin = time.perf_counter_ns()
    data = prepare(start, end, grid, algorithm, start_cost)
    prepared = time.perf_counter_ns()
    result = query(data)
    finished = time.perf_counter_ns()
    return result, ((prepared-begin)/1e6, (finished-prepared)/1e6, (finished-begin)/1e6)


def memory_run(start, end, grid, algorithm, start_cost):
    tracemalloc.start()
    try:
        data = prepare(start, end, grid, algorithm, start_cost)
        retained, preprocessing_peak = tracemalloc.get_traced_memory()
        tracemalloc.reset_peak()
        result = query(data)
        _, query_peak = tracemalloc.get_traced_memory()
        sample = {"preprocessing_peak": preprocessing_peak, "retained_preprocessing": retained,
                  "query_peak_with_index": query_peak,
                  "query_additional_peak": max(0, query_peak-retained),
                  "total_peak": max(preprocessing_peak, query_peak)}
    finally:
        tracemalloc.stop()
    return result, sample


def benchmark(cases, repeats, memory_repeats):
    records = []
    rng = random.Random(20261109)
    for index, case in enumerate(cases, 1):
        grid = [list(row) for row in case["grid"]]
        original = [row[:] for row in grid]
        start, end = tuple(case["start"]), tuple(case["end"])
        start_cost = case["start_terrain_cost"]
        expected = reference_cost(grid, start, end, start_cost)
        if expected != case["expected"]["minimum_cost"] or (expected!=-1) != case["expected"]["reachable"]:
            raise AssertionError(f"{case['id']}: map 参考答案不一致")
        baseline, timings, memories = {}, {}, {}
        for algorithm in ALGORITHMS:
            baseline[algorithm] = search(start,end,grid,algorithm,start_cost)
            validate(baseline[algorithm],grid,start,end,start_cost,expected)
            if baseline[algorithm].min_non_goal_cost != case["min_non_goal_cost"]:
                raise AssertionError("输入计算的最小权重与地图记录不同")
            timings[algorithm], memories[algorithm] = [], []
        for phase, count in (("time",repeats),("memory",memory_repeats)):
            for _ in range(count):
                order = list(ALGORITHMS)
                rng.shuffle(order)
                for algorithm in order:
                    result = None
                    gc.collect()
                    if phase == "time":
                        result, sample = timed_run(start,end,grid,algorithm,start_cost)
                        timings[algorithm].append(sample)
                    else:
                        result, sample = memory_run(start,end,grid,algorithm,start_cost)
                        memories[algorithm].append(sample)
                    validate(result,grid,start,end,start_cost,expected)
                    if result != baseline[algorithm] or grid != original:
                        raise AssertionError(f"{case['id']} / {algorithm}: 结果不稳定或输入被修改")
        for algorithm in ALGORITHMS:
            result = baseline[algorithm]
            stages = timings[algorithm]
            memory = memories[algorithm]
            record = {
                "case_id": case["id"], "size": f"{case['rows']}x{case['cols']}", "kind": case["kind"],
                "terrain_weights": "/".join(map(str,case["parameters"].get("terrain_weights",[]))),
                "target_density": case["parameters"].get("target_obstacle_density",""),
                "actual_density": case["actual_obstacle_density"], "seed": case["parameters"].get("seed",""),
                "algorithm": algorithm, "correct": True, "reachable": expected!=-1,
                "expected_cost": expected, "minimum_cost": result.minimum_cost,
                "path_steps": len(result.path)-1 if result.path else None,
                "start_terrain_cost": start_cost, "min_non_goal_cost": result.min_non_goal_cost,
                "heuristic_scale": result.heuristic_scale,
                "requested_landmarks": 2 if algorithm=="ALT A* (2)" else 4 if algorithm=="ALT A* (4)" else 0,
                "actual_landmarks": len(result.landmarks), "landmarks": list(result.landmarks),
                "expanded_nodes": result.expanded_nodes, "popped_entries": result.popped_entries,
                "frontier_peak": result.frontier_peak,
                "preprocessing_expanded_nodes": result.preprocessing_expanded_nodes,
                "total_expanded_nodes": result.expanded_nodes+result.preprocessing_expanded_nodes,
                "preprocessing_time_median_ms": statistics.median(s[0] for s in stages),
                "query_time_median_ms": statistics.median(s[1] for s in stages),
                "time_median_ms": statistics.median(s[2] for s in stages),
                "time_min_ms": min(s[2] for s in stages), "time_max_ms": max(s[2] for s in stages),
                "preprocessing_time_samples_ms": [s[0] for s in stages],
                "query_time_samples_ms": [s[1] for s in stages], "time_samples_ms": [s[2] for s in stages],
                "memory_samples_bytes": [s["total_peak"] for s in memory],
                "memory_stage_samples_bytes": memory,
                "memory_peak_max_kib": max(s["total_peak"] for s in memory)/1024,
                "path": result.path,
            }
            for raw, name in (
                ("total_peak","memory_peak_median_kib"),
                ("preprocessing_peak","preprocessing_memory_peak_median_kib"),
                ("retained_preprocessing","retained_preprocessing_median_kib"),
                ("query_peak_with_index","query_peak_with_index_median_kib"),
                ("query_additional_peak","query_additional_peak_median_kib"),
            ):
                record[name] = statistics.median(s[raw] for s in memory)/1024
            records.append(record)
        print(f"[{index:03d}/{len(cases)}] {case['id']}: four algorithms passed",flush=True)
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
        "preprocessing_time_median_ms_mean": statistics.mean(r["preprocessing_time_median_ms"] for r in rows),
        "query_time_median_ms_mean": statistics.mean(r["query_time_median_ms"] for r in rows),
        "preprocessing_expanded_nodes_mean": statistics.mean(r["preprocessing_expanded_nodes"] for r in rows),
        "total_expanded_nodes_mean": statistics.mean(r["total_expanded_nodes"] for r in rows),
        "preprocessing_memory_peak_median_kib_mean": statistics.mean(r["preprocessing_memory_peak_median_kib"] for r in rows),
        "retained_preprocessing_median_kib_mean": statistics.mean(r["retained_preprocessing_median_kib"] for r in rows),
        "query_peak_with_index_median_kib_mean": statistics.mean(r["query_peak_with_index_median_kib"] for r in rows),
        "query_additional_peak_median_kib_mean": statistics.mean(r["query_additional_peak_median_kib"] for r in rows),
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
        ("expanded_nodes_mean", "查询扩展节点数", ".1f"),
        ("time_median_ms_mean", "总耗时（ms）", ".3f"),
        ("memory_peak_median_kib_mean", "总峰值内存（KiB）", ".1f"),
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
        fig.suptitle("单次查询性能对照｜数值越小越好（* 表示最优）", fontsize=14, y=0.98)
        fig.text(0.5, 0.035, f"{count} 张地图等权平均；耗时和内存包含预处理，先取单图中位数，再跨图平均。",
                 ha="center", fontsize=9)
        fig.tight_layout(rect=(0, 0.08, 1, 0.93), w_pad=1.8)
        for extension in ("png", "svg"):
            fig.savefig(output / f"performance_comparison_v2.1.{extension}", dpi=220)
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
    detail = [{k: v for k, v in row.items() if k not in ("path", "time_samples_ms", "memory_samples_bytes", "preprocessing_time_samples_ms", "query_time_samples_ms", "memory_stage_samples_bytes")}
              for row in records]
    write_csv(args.output / "details_v2.1.csv", detail)
    write_csv(args.output / "summary_v2.1.csv", summary)
    matplotlib_version = plot_results(summary, args.output)
    metadata = {
        "created_at": datetime.now().astimezone().isoformat(),
        "python": sys.version, "platform": platform.platform(), "processor": platform.processor(),
        "matplotlib": matplotlib_version, "map_path": str(args.map.resolve()),
        "map_sha256": hashlib.sha256(raw).hexdigest(),
        "strategy_version": "v2.1",
        "heuristics": {"Dijkstra": "0", "A* scaled": "c_min*max(d-1,0)", "ALT": "max(scaled, d(L,t)-d(L,n), d(n,L)-d(t,L)) using finite pairs"},
        "landmark_selection": "Nearest distinct walkable cells to interior quarter anchors; 2 landmarks are prefix of 4",
        "query_scope": "Fixed start/end, free goal; fresh preprocessing on every measured run, no cross-query cache",
        "time_stages": "preprocessing includes terrain conversion, minimum scan, landmark selection and both distance directions; query includes heuristic evaluation and path reconstruction; total measured per sample",
        "memory_stages": "total=max(preprocessing peak, query peak with retained index); query additional=max(0, query peak - retained preprocessing allocation); Python tracemalloc, input excluded",
        "expansion_scope": "main expanded_nodes counts query only, goal and stale entries excluded; preprocessing expansions recorded separately",
        "case_count": len(cases), "repeats": args.repeats, "memory_repeats": args.memory_repeats,
        "measurement_order_seed": 20261109, "gc_enabled": gc.isenabled(),
        "timer": "perf_counter_ns", "memory": "tracemalloc Python allocation peak",
        "command": [sys.executable] + sys.argv,
    }
    (args.output / "results_v2.1.json").write_text(json.dumps(
        {"metadata": metadata, "records": records, "summary": summary}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"All {len(records)} combinations passed. Results: {args.output.resolve()}")


if __name__ == "__main__":
    main()
