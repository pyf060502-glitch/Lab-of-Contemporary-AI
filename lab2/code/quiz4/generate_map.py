"""生成 Quiz4 输入 map：题目地图与少量规则检查地图，标准库即可运行。

在 lab2 下运行：python code/quiz4/generate_map.py
地图为二维字符串列表，支持 T1 等多字符标记；全部用例为固定手工配置。
参考距离使用 Dijkstra，参考决策使用完整递归求值，只用于校验。
"""

import argparse
import heapq
import json
from pathlib import Path


LABELS = ("A", "B", "T1", "T2", "T3", "T4")


def distance_reference(grid, start, goal):
    """进入 A/B/T 所在格收取 1，目标不免费。"""
    queue = [(0, start)]
    best = {start: 0}
    while queue:
        cost, (row, col) = heapq.heappop(queue)
        if cost != best[(row, col)]:
            continue
        if (row, col) == goal:
            return cost
        for nr, nc in ((row-1, col), (row+1, col), (row, col-1), (row, col+1)):
            if not (0 <= nr < len(grid) and 0 <= nc < len(grid[0])):
                continue
            cell = grid[nr][nc]
            if cell == "X":
                continue
            weight = 1 if cell in LABELS else int(cell)
            candidate = cost + weight
            if candidate < best.get((nr, nc), float("inf")):
                best[(nr, nc)] = candidate
                heapq.heappush(queue, (candidate, (nr, nc)))
    return None


def game_reference(distances, values, horizon):
    """完整求值所有合法选择；平局时记录全部最优首步。"""
    def visit(a, b, remaining, turn, difference, depth):
        if depth == 0 or not remaining:
            return difference
        candidates = []
        for target in remaining:
            gain = values[target-2] - distances[b if turn else a][target]
            rest = tuple(t for t in remaining if t != target)
            candidates.append(visit(
                a if turn else target, target if turn else b, rest, 1-turn,
                difference-gain if turn else difference+gain, depth-1))
        return (min if turn else max)(candidates)

    first_move_values = {}
    for target in range(2, 6):
        first_move_values[LABELS[target]] = visit(
            target, 1, tuple(t for t in range(2, 6) if t != target), 1,
            values[target-2] - distances[0][target], horizon-1)
    best = max(first_move_values.values())
    optimal = [name for name, score in first_move_values.items() if score == best]
    return {
        "horizon": horizon, "best_target": optimal[0], "optimal_targets": optimal,
        "best_value": best, "first_move_values": first_move_values,
    }


def make_case(case_id, kind, description, grid, values):
    grid = [list(row) for row in grid]
    rows, cols = len(grid), len(grid[0])
    assert all(len(row) == cols for row in grid)
    positions = {}
    for r, row in enumerate(grid):
        for c, cell in enumerate(row):
            if cell in LABELS:
                assert cell not in positions
                positions[cell] = (r, c)
            else:
                assert cell == "X" or (cell.isdigit() and int(cell) >= 1)
    assert set(positions) == set(LABELS)
    distances = [[distance_reference(grid, positions[a], positions[b])
                  for b in LABELS] for a in LABELS]
    # 本批用例用于规定的四次行动，所有位置两两可达。
    assert all(d is not None for row in distances for d in row)
    decisions = {str(h): game_reference(distances, values, h) for h in (2, 4)}
    return {
        "id": case_id, "kind": kind, "description": description,
        "rows": rows, "cols": cols, "seed": None,
        "grid": grid,
        "players": {
            "A": {"position": list(positions["A"]), "role": "MAX"},
            "B": {"position": list(positions["B"]), "role": "MIN"},
        },
        "treasures": [{"id": label, "position": list(positions[label]), "value": value}
                      for label, value in zip(LABELS[2:], values)],
        "initial_state": {
            "pos_a": "A", "pos_b": "B", "remaining": list(LABELS[2:]),
            "remaining_mask": 15, "score_a": 0, "score_b": 0, "current_player": "A",
        },
        "statistics": {
            "obstacle_count": sum(cell == "X" for row in grid for cell in row),
            "passable_count": sum(cell != "X" for row in grid for cell in row),
            "maximum_entry_cost": max(1 if cell in LABELS else int(cell)
                                      for row in grid for cell in row if cell != "X"),
        },
        "expected": {
            "distance_order": list(LABELS), "distance_matrix": distances,
            "decisions": decisions,
            "full_minimax_counts": {
                "2": {"visited_nodes": 17, "leaf_evaluations": 12},
                "4": {"visited_nodes": 65, "leaf_evaluations": 24},
            },
        },
    }


def build_dataset():
    cases = [
        make_case(
            "example_7x7", "example", "题目主用例：H=2 选 T2、收益差 -1；H=4 选 T3、收益差 3。",
            [["A","1","1","X","2","1","T1"],
             ["1","X","6","X","1","X","1"],
             ["1","1","T2","1","1","1","1"],
             ["X","1","X","8","X","1","X"],
             ["1","1","1","1","T3","1","1"],
             ["1","X","9","X","1","X","1"],
             ["T4","1","1","X","1","1","B"]], [9,6,7,7]),
        make_case(
            "pass_through_treasures", "pass_through",
            "单行地图：前往较远宝藏时必经其他宝藏及其位置，但每次只获取选中的一个。",
            [["A","T1","T2","T3","T4","B"]], [3,5,7,9]),
        make_case(
            "negative_gains", "negative_gains",
            "宝藏价值均为 1，初始玩家及宝藏间的距离均大于 1；负收益行动仍必须正常执行。",
            [["A","1","T1","1","1","1","T2","1","1","1","T3","1","1","1","T4","1","B"]],
            [1,1,1,1]),
        make_case(
            "symmetric_ties", "ties",
            "上下对称、价值相等，存在并列最优首步；输出应区分唯一选择与最优集合。",
            [["T1","1","T3"], ["A","1","B"], ["T2","1","T4"]], [5,5,5,5]),
        make_case(
            "weighted_detour", "weighted_detour",
            "包含权重 8 与障碍，最低代价路径可能绕行；所有特殊格的进入代价仍为 1。",
            [["A","1","T3","8","T1"],
             ["1","X","1","X","1"],
             ["1","1","1","1","1"],
             ["1","X","1","X","1"],
             ["T2","1","T4","1","B"]], [9,7,5,6]),
        make_case(
            "position_updates", "position_updates",
            "开放矩形地图，宝藏分散：第二次行动必须从该玩家首次选择的宝藏位置出发。",
            [["A","1","T3","1","T1"],
             ["1","1","1","1","1"],
             ["T2","1","T4","1","B"]], [9,7,5,6]),
    ]
    return {
        "format_version": 1, "quiz": "quiz4",
        "description": "竞争环境下的宝藏分配：题目地图及五个手工规则检查用例。",
        "primary_case_id": "example_7x7",
        "coordinate_convention": "从 0 开始的 [row,col]；grid 是二维字符串列表，T1 等标记占一个格子。",
        "rules": {
            "movement": "仅上下左右；X 不可通行；进入数字格支付该数字，进入 A/B/T 所在格支付 1。",
            "initial_cost": "初始位置不收费；目标宝藏格不免费。",
            "turn_order": ["A","B","A","B"],
            "selection": "每次必须选择一个剩余宝藏，净收益为负也可选；只获取选中的宝藏。",
            "positions": "行动后仅更新当前玩家位置；另一玩家不阻挡寻路。",
            "terrain": "经过其他宝藏不自动获取；已获取宝藏的格子仍可通行且代价保持为 1。",
            "gain": "value(Ti) - distance(current_position, Ti)",
            "utility": "score_a - score_b；A 最大化，B 最小化。",
            "horizons": [2,4],
            "horizon_definition": "H 是单次玩家行动数，H=2 表示双方各一次，H=4 表示双方各两次。",
            "evaluation": "达到 H 或宝藏取完时，直接使用当前收益差，不估计剩余宝藏。",
            "mask_bits": {"T1":0,"T2":1,"T3":2,"T4":3},
            "default_target_order": ["T1","T2","T3","T4"],
            "tie_break": "best_target 取原始编号最小的最优宝藏；optimal_targets 记录全部最优选择。",
        },
        "generation": "固定手工地图，无随机抽样；重复生成得到相同数据。",
        "reference": "距离由独立 Dijkstra 计算，决策由完整递归求值生成；expected 仅用于校验。",
        "benchmark_notes": [
            "主实验使用 primary_case_id；其余地图用于规则和正确性检查，可单独评测。",
            "被测算法应自行使用 A* 构建距离矩阵；参考矩阵不作为输入。",
            "静态距离矩阵可在同地图各搜索策略间共享，预处理与博弈搜索成本分别记录。",
            "所有用例的六个特殊位置均两两可达，均有四个宝藏。",
        ],
        "case_count": len(cases), "cases": cases,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("map"))
    args = parser.parse_args()
    dataset = build_dataset()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(dataset, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Generated {dataset['case_count']} cases: {args.output}")


if __name__ == "__main__":
    main()
