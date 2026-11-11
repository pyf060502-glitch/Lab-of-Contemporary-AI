"""生成 Quiz2 加权地形原始数据 map（UTF-8 JSON），仅使用标准库。

在 lab2 目录运行：python code/quiz2/generate_map.py。
沿用 Quiz1 的 cases/start/end/grid 结构，grid 每个字符串对应地图一行。
数值字符为进入代价，X 为障碍，E 为免费终点。S 的进入代价单独记录，
初始 g(S)=0；起终点重合时只保留 S，坐标优先，答案为 0。
expected 用于评测校验，不作为搜索算法的输入或启发信息。
"""

import heapq
import json
import random
from collections import Counter, deque
from pathlib import Path


SIZES = (20, 50, 100)
DENSITIES = (0.0, 0.15, 0.30)
SEEDS = (20261109, 20261110, 20261111)
PALETTES = ((1, 3, 8), (2, 5, 9), (3, 6, 9))
DIRECTIONS = ((-1, 0), (1, 0), (0, -1), (0, 1))


def entry_cost(grid, position, end, start_terrain_cost):
    if position == end:
        return 0
    value = grid[position[0]][position[1]]
    return start_terrain_cost if value == "S" else int(value)


def dijkstra_reference(grid, start, end, start_terrain_cost):
    """参考最小代价，以及所有最小代价路径中最少的移动次数。

    使用 (cost, steps) 字典序更新；steps 仅用于参考答案的同代价平局处理。
    """
    best = {start: (0, 0)}
    queue = [(0, 0, start)]
    while queue:
        cost, steps, current = heapq.heappop(queue)
        if (cost, steps) != best[current]:
            continue
        if current == end:
            return cost, steps
        row, col = current
        for dr, dc in DIRECTIONS:
            neighbor = (row + dr, col + dc)
            nr, nc = neighbor
            if not (0 <= nr < len(grid) and 0 <= nc < len(grid[0])) or grid[nr][nc] == "X":
                continue
            candidate = (cost + entry_cost(grid, neighbor, end, start_terrain_cost), steps + 1)
            if candidate < best.get(neighbor, (float("inf"), float("inf"))):
                best[neighbor] = candidate
                heapq.heappush(queue, (candidate[0], candidate[1], neighbor))
    return -1, None


def bfs_steps(grid, start, end):
    """忽略地形权重的最少步数，仅用于分析“步数最少”与“代价最低”的差异。"""
    queue = deque([(start, 0)])
    visited = {start}
    while queue:
        (row, col), steps = queue.popleft()
        if (row, col) == end:
            return steps
        for dr, dc in DIRECTIONS:
            nr, nc = row + dr, col + dc
            if 0 <= nr < len(grid) and 0 <= nc < len(grid[0]) and grid[nr][nc] != "X" and (nr, nc) not in visited:
                visited.add((nr, nc))
                queue.append(((nr, nc), steps + 1))
    return None


def make_case(case_id, kind, description, grid, start, end, start_terrain_cost=1, **parameters):
    grid = [list(row) for row in grid]
    rows, cols = len(grid), len(grid[0])
    grid[start[0]][start[1]] = "S"
    if start != end:
        grid[end[0]][end[1]] = "E"
    weights = [entry_cost(grid, (r, c), end, start_terrain_cost)
               for r in range(rows) for c in range(cols) if grid[r][c] != "X" and (r, c) != end]
    obstacles = sum(row.count("X") for row in grid)
    cost, cost_steps = dijkstra_reference(grid, start, end, start_terrain_cost)
    return {
        "id": case_id, "kind": kind, "description": description,
        "rows": rows, "cols": cols, "start": list(start), "end": list(end),
        "parameters": parameters,
        "start_terrain_cost": start_terrain_cost,
        "min_non_goal_cost": min(weights) if weights else 0,
        "max_non_goal_cost": max(weights) if weights else 0,
        "terrain_counts_excluding_goal": dict(sorted(Counter(str(w) for w in weights).items())),
        "obstacle_count": obstacles, "actual_obstacle_density": obstacles / (rows * cols),
        "expected": {"reachable": cost != -1, "minimum_cost": cost,
                     "steps_among_minimum_cost_paths": cost_steps,
                     "shortest_steps_ignoring_weights": bfs_steps(grid, start, end)},
        "grid": ["".join(row) for row in grid],
    }


def build_dataset():
    cases = []
    cases.append(make_case(
        "example_5x5", "example", "题目示例：进入终点免费，最小代价为 7。",
        ["S1181", "18311", "18181", "11181", "8811E"], (0, 0), (4, 4)))
    cases.append(make_case(
        "adjacent_free_goal", "adjacent_goal", "终点相邻，代价为 0；普通曼哈顿距离会高估。",
        ["SE"], (0, 0), (0, 1)))
    cases.append(make_case(
        "same_endpoint", "same_endpoint", "起终点重合，坐标为准，只保留 S，移动代价为 0。",
        ["S"], (0, 0), (0, 0), 3))
    cases.append(make_case(
        "single_row_weighted", "single_row", "只有一条路径，检查累计进入代价，不计初始起点，终点免费。",
        ["S381E"], (0, 0), (0, 4), 2))
    cases.append(make_case(
        "zero_cost_strip", "zero_terrain", "除终点外还有零代价地形；最小权重为 0，缩放启发退化为 0。",
        ["S000E", "83838", "11111"], (0, 0), (0, 4)))
    cases.append(make_case(
        "blocked_pair", "blocked_pair", "最小无解例，障碍 X 不可进入，期望返回 -1。",
        ["SXE"], (0, 0), (0, 2)))

    for size in SIZES:
        prefix = f"{size}x{size}"
        start, end = (0, 0), (size - 1, size - 1)
        for weight in (1, 2, 3):
            cases.append(make_case(
                f"{prefix}_uniform_{weight}", "uniform", "均匀正权地形，可验证修正曼哈顿及最小权重缩放。",
                [[str(weight)] * size for _ in range(size)], start, end, weight, uniform_weight=weight))

        grid = [["8"] * size for _ in range(size)]
        grid[1] = ["1"] * size
        cases.append(make_case(
            f"{prefix}_cheap_detour", "cheap_detour", "顶行直走步数最少但代价高；经第二行多走两步更便宜。",
            grid, start, (0, size - 1)))

        grid = [["2"] * size for _ in range(size)]
        for row in range(size - 1):
            grid[row][size // 2] = "X"
        cases.append(make_case(
            f"{prefix}_forced_detour", "forced_detour", "起终点位于顶行，必须经过竖墙底部缺口，非终点权重均为 2。",
            grid, start, (0, size - 1), 2))

        grid = [[str((3, 6, 9)[(r + c) % 3]) for c in range(size)] for r in range(size)]
        for row in range(size):
            grid[row][size // 2] = "X"
        cases.append(make_case(
            f"{prefix}_disconnected", "disconnected", "混合权重地图被完整竖墙分隔，起终点无路径。",
            grid, start, end, 3))

        for palette in PALETTES:
            for density in DENSITIES:
                for seed in SEEDS:
                    rng = random.Random(seed)
                    grid = [["X" if rng.random() < density else str(rng.choice(palette))
                             for _ in range(size)] for _ in range(size)]
                    name = "".join(map(str, palette))
                    cases.append(make_case(
                        f"{prefix}_random_{name}_density{int(density * 100):02d}_seed{seed}", "random",
                        "随机混合地形与障碍，强制起终点可通行，不挖通路径、不筛选可达性。",
                        grid, start, end, min(palette), terrain_weights=list(palette),
                        target_obstacle_density=density, seed=seed))

    return {
        "format_version": 1, "quiz": "quiz2",
        "description": "四方向加权地形最小代价搜索原始输入数据。",
        "coordinate_convention": "从 0 开始的 [row, col]，加载后用 tuple 转为坐标。",
        "grid_encoding": "沿用 Quiz1：每个字符串是一行；用 [list(row) for row in case['grid']] 转为二维字符列表。当前数字权重为 0..9。",
        "symbols": {"S": "起点，其再次进入代价为 case.start_terrain_cost", "E": "终点，进入代价为 0", "X": "障碍", "0..9": "进入该地形的整数代价"},
        "cost_rules": [
            "只能上下左右移动；从 A 进入相邻 B 的代价为 B 的权重。",
            "初始 g(start)=0，不收取起点自身权重；再次进入起点使用 start_terrain_cost。",
            "终点坐标优先，进入终点总是免费；起终点重合时只保留 S，答案为 0。",
            "找不到路径时 minimum_cost=-1；不存在负权重。",
        ],
        "reference": "minimum_cost 由独立 Dijkstra 计算；同最小代价路径的最少步数仅作诊断；忽略权重的最少步数由 BFS 计算。",
        "heuristic_notes": [
            "普通曼哈顿距离可能高估免费终点的剩余代价，adjacent_free_goal 是直接反例。",
            "只有终点免费、其余权重至少为 c_min 时，可用 c_min*max(Manhattan-1,0)。",
            "min_non_goal_cost 包含起点的进入代价；零地形用例中该值为 0，缩放启发退化为 Dijkstra。",
            "expected 仅用于校验，不传入被测搜索；启发所需最小地形权重可自行从地图计算。",
        ],
        "random_policy": "每个配置独立 random.Random(seed)，按行生成；同规模/密度/种子跨三组权重具有相同障碍布局。目标密度为生成概率，实际密度另行记录。",
        "benchmark_notes": [
            "各算法使用同一批 cases，读取和参考答案计算不计入搜索耗时。",
            "分别按规模、地形最小权重、障碍密度、结构和可达性汇总。",
            "测量时间与峰值内存分开进行，重复计时取中位数。",
        ],
        "case_count": len(cases), "cases": cases,
    }


if __name__ == "__main__":
    dataset = build_dataset()
    destination = Path(__file__).with_name("map")
    destination.write_text(json.dumps(dataset, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    reachable = sum(c["expected"]["reachable"] for c in dataset["cases"])
    print(f"Generated {destination.name}: {dataset['case_count']} cases, "
          f"{reachable} reachable, {dataset['case_count'] - reachable} unreachable.")
