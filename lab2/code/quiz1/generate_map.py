"""生成同目录下的原始测试数据文件 map（UTF-8 JSON）。

运行：py -3 Lab-of-Contemporary-AI/lab2/code/quiz1/generate_map.py
读取示例：
    data = json.loads(Path("map").read_text(encoding="utf-8"))
    case = data["cases"][0]
    start, end = tuple(case["start"]), tuple(case["end"])
    grid = [list(row) for row in case["grid"]]
    path = a_star(start, end, grid)

grid 用字符串列表紧凑保存，每个字符串对应一行；转换后即题目要求的
List[List[str]]。expected 中的数据仅用于校验，不应传给被测算法。
"""

import json
import random
from collections import deque
from pathlib import Path


SIZES = (20, 50, 100)
DENSITIES = (0.10, 0.25, 0.40)
SEEDS = (20261109, 20261110, 20261111)


def bfs_distance(grid, start, end):
    """独立的 BFS 参考答案；不可达返回 None（JSON 中为 null）。"""
    queue = deque([(start, 0)])
    visited = {start}
    while queue:
        (row, col), distance = queue.popleft()
        if (row, col) == end:
            return distance
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = row + dr, col + dc
            if (
                0 <= nr < len(grid)
                and 0 <= nc < len(grid[0])
                and grid[nr][nc] != "X"
                and (nr, nc) not in visited
            ):
                visited.add((nr, nc))
                queue.append(((nr, nc), distance + 1))
    return None


def make_case(case_id, kind, description, grid, start, end, **parameters):
    grid[start[0]][start[1]] = "S"
    grid[end[0]][end[1]] = "E"
    rows, cols = len(grid), len(grid[0])
    obstacles = sum(row.count("X") for row in grid)
    distance = bfs_distance(grid, start, end)
    return {
        "id": case_id,
        "kind": kind,
        "description": description,
        "rows": rows,
        "cols": cols,
        "start": list(start),
        "end": list(end),
        "parameters": parameters,
        "obstacle_count": obstacles,
        "actual_obstacle_density": obstacles / (rows * cols),
        "expected": {"reachable": distance is not None, "shortest_steps": distance},
        "grid": ["".join(row) for row in grid],
    }


def build_dataset():
    cases = []
    for size in SIZES:
        prefix = f"{size}x{size}"
        start, end = (0, 0), (size - 1, size - 1)
        grid = [["."] * size for _ in range(size)]
        cases.append(make_case(
            f"{prefix}_open", "open", "无障碍，多条等长最短路径，可观察相同 f 值时的搜索行为。",
            grid, start, end,
        ))

        grid = [["X"] * size for _ in range(size)]
        for band, row in enumerate(range(0, size, 2)):
            grid[row] = ["."] * size
            if row + 2 < size:
                grid[row + 1][size - 1 if band % 2 == 0 else 0] = "."
        last_band = (size - 1) // 2
        corridor_end = (2 * last_band, size - 1 if last_band % 2 == 0 else 0)
        cases.append(make_case(
            f"{prefix}_corridor", "corridor", "单格宽蛇形长走廊，只有一条简单路径。",
            grid, start, corridor_end,
        ))

        grid = [["."] * size for _ in range(size)]
        for row in range(size - 1):
            grid[row][size // 2] = "X"
        cases.append(make_case(
            f"{prefix}_detour", "detour", "起终点位于顶行，必须经过竖墙底部缺口绕行。",
            grid, start, (0, size - 1),
        ))

        grid = [["."] * size for _ in range(size)]
        for row in range(size):
            grid[row][size // 2] = "X"
        cases.append(make_case(
            f"{prefix}_disconnected", "disconnected", "完整竖墙分隔起终点，起点侧有大片可搜索区域。",
            grid, start, end,
        ))

        for density in DENSITIES:
            for seed in SEEDS:
                rng = random.Random(seed)
                grid = [
                    ["X" if rng.random() < density else "." for _ in range(size)]
                    for _ in range(size)
                ]
                cases.append(make_case(
                    f"{prefix}_random_{int(density * 100):02d}_seed{seed}",
                    "random", "独立随机障碍，仅强制起终点可通行，不挖通路径、不筛选可达性。",
                    grid, start, end, target_obstacle_density=density, seed=seed,
                ))

    return {
        "format_version": 1,
        "description": "四方向、单位移动代价的迷宫搜索基准原始输入数据。",
        "coordinate_convention": "从 0 开始的 [row, col]；加载后用 tuple 转换坐标。",
        "symbols": {"S": "起点", "E": "终点", ".": "可通行", "X": "障碍"},
        "grid_encoding": "每个字符串是一行；使用 [list(row) for row in case['grid']] 转为二维字符列表。",
        "reference": "expected 由独立 BFS 计算；shortest_steps 是移动次数，无解为 null。",
        "random_policy": "每张随机地图使用独立 random.Random(seed)，按行生成；密度为概率，实际密度另行记录。",
        "benchmark_notes": [
            "所有算法加载同一批 cases；解析输入和 BFS 参考答案计算不计入搜索耗时。",
            "按 kind、规模、障碍密度、可达性分别汇总，不只比较总体平均值。",
            "重复计时并报告中位数；峰值内存与计时分开测量。",
            "每个随机配置仅三个样本，适合初步比较，更稳定的统计结论需要增加种子。",
        ],
        "case_count": len(cases),
        "cases": cases,
    }


if __name__ == "__main__":
    dataset = build_dataset()
    destination = Path(__file__).with_name("map")
    destination.write_text(json.dumps(dataset, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    reachable = sum(case["expected"]["reachable"] for case in dataset["cases"])
    print(f"Generated {destination.name}: {dataset['case_count']} cases, "
          f"{reachable} reachable, {dataset['case_count'] - reachable} unreachable.")
