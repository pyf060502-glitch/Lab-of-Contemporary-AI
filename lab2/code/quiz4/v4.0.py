"""Quiz4：A* 距离预处理与四组博弈策略对照。

在 lab2 下运行：python code/quiz4/v4.0.py
输入 code/quiz4/map，输出 result/quiz4，文件名均带 _v4.0。
H 按玩家单次行动计数；进入宝藏收费 1，负收益行动仍合法。
"""

import argparse
import csv
import gc
import hashlib
import heapq
import itertools
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
LABELS = ("A", "B", "T1", "T2", "T3", "T4")
ALGORITHMS = ("Minimax", "AB original", "AB reverse", "AB gain")
DISPLAY = {"Minimax": "Minimax", "AB original": "Alpha-Beta：原始顺序",
           "AB reverse": "Alpha-Beta：反向顺序", "AB gain": "Alpha-Beta：即时收益排序"}
HORIZONS = (2, 4)
MEASUREMENT_SEED = 20261109


@dataclass(frozen=True)
class State:
    pos_a: int = 0
    pos_b: int = 1
    remaining_mask: int = 15
    score_a: int = 0
    score_b: int = 0
    current_player: int = 0  # 0=A/MAX，1=B/MIN

    @property
    def utility(self):
        return self.score_a - self.score_b


@dataclass(frozen=True)
class Game:
    distances: tuple
    values: tuple


@dataclass
class SearchResult:
    best_value: int = 0
    best_target: object = None  # 0～3 对应 T1～T4
    visited_nodes: int = 0
    leaf_evaluations: int = 0
    effective_cutoffs: int = 0


def read_terrain(case):
    grid = case["grid"]
    if not grid or not grid[0] or any(len(row) != len(grid[0]) for row in grid):
        raise ValueError("grid 应为非空矩形二维数组")
    positions = {}
    costs = []
    for r, row in enumerate(grid):
        converted = []
        for c, cell in enumerate(row):
            if cell in LABELS:
                if cell in positions:
                    raise ValueError(f"重复标记：{cell}")
                positions[cell] = (r, c)
                converted.append(1)
            elif cell == "X":
                converted.append(-1)
            else:
                weight = int(cell)
                if weight < 1:
                    raise ValueError("本题地形权重至少为 1")
                converted.append(weight)
        costs.append(tuple(converted))
    if set(positions) != set(LABELS):
        raise ValueError("地图需要 A、B、T1～T4 六个标记")
    for name in ("A", "B"):
        if tuple(case["players"][name]["position"]) != positions[name]:
            raise ValueError(f"{name} 的坐标与地图标记不一致")
    treasures = {t["id"]: t for t in case["treasures"]}
    if set(treasures) != set(LABELS[2:]):
        raise ValueError("需要四个宝藏")
    for name in LABELS[2:]:
        if tuple(treasures[name]["position"]) != positions[name]:
            raise ValueError(f"{name} 的坐标与地图标记不一致")
    return tuple(costs), positions, tuple(treasures[name]["value"] for name in LABELS[2:])


def a_star(costs, start, goal):
    """曼哈顿 A*；扩展数排除终点与过期堆条目。"""
    def heuristic(pos):
        return abs(pos[0] - goal[0]) + abs(pos[1] - goal[1])

    frontier = [(heuristic(start), 0, start)]
    best = {start: 0}
    parent = {}
    expanded = 0
    while frontier:
        _, distance, current = heapq.heappop(frontier)
        if distance != best[current]:
            continue
        if current == goal:
            path = [current]
            while path[-1] != start:
                path.append(parent[path[-1]])
            path.reverse()
            return distance, path, expanded
        expanded += 1
        r, c = current
        for nr, nc in ((r-1,c), (r+1,c), (r,c-1), (r,c+1)):
            if not (0 <= nr < len(costs) and 0 <= nc < len(costs[0])):
                continue
            if costs[nr][nc] < 0:
                continue
            neighbor = (nr, nc)
            candidate = distance + costs[nr][nc]
            if candidate < best.get(neighbor, math.inf):
                best[neighbor] = candidate
                parent[neighbor] = current
                heapq.heappush(frontier, (candidate + heuristic(neighbor), candidate, neighbor))
    raise ValueError("本批博弈要求六个特殊位置两两可达")


def prepare_game(case):
    """独立计算 30 个有序点对，包含地形转换和全部路径恢复。"""
    costs, positions, values = read_terrain(case)
    matrix = [[0] * 6 for _ in range(6)]
    paths = {}
    total_expanded = 0
    for i, source in enumerate(LABELS):
        for j, target in enumerate(LABELS):
            if i == j:
                continue
            distance, path, expanded = a_star(costs, positions[source], positions[target])
            matrix[i][j] = distance
            paths[f"{source}->{target}"] = {
                "minimum_cost": distance, "path": path, "expanded_nodes": expanded,
            }
            total_expanded += expanded
    return Game(tuple(tuple(row) for row in matrix), values), paths, total_expanded


def initial_state(case):
    raw = case["initial_state"]
    mask = sum(1 << LABELS[2:].index(name) for name in raw["remaining"])
    if mask != raw["remaining_mask"]:
        raise ValueError("剩余宝藏列表与位掩码不一致")
    return State(LABELS.index(raw["pos_a"]), LABELS.index(raw["pos_b"]), mask,
                 raw["score_a"], raw["score_b"], ("A", "B").index(raw["current_player"]))


def gain(game, state, target):
    pos = state.pos_a if state.current_player == 0 else state.pos_b
    return game.values[target] - game.distances[pos][target + 2]


def transition(game, state, target):
    """只删除选中的宝藏，分别更新位置和收益；途经位置不参与此转移。"""
    if not state.remaining_mask & (1 << target):
        raise ValueError("不能重复选择已获取宝藏")
    reward = gain(game, state, target)
    mask = state.remaining_mask & ~(1 << target)
    if state.current_player == 0:
        return State(target+2, state.pos_b, mask, state.score_a+reward, state.score_b, 1)
    return State(state.pos_a, target+2, mask, state.score_a, state.score_b+reward, 0)


def actions(game, state, algorithm):
    targets = [i for i in range(4) if state.remaining_mask & (1 << i)]
    if algorithm == "AB reverse":
        targets.reverse()
    elif algorithm == "AB gain":
        # B 也按自身 Gain 降序，相当于优先降低 U；平局按 T 编号。
        targets.sort(key=lambda i: (-gain(game, state, i), i))
    return targets


def search(game, state, horizon, algorithm="AB original"):
    """根值精确；平局保留首个访问的最优行动，不用界值覆盖已确定的选择。"""
    if algorithm not in ALGORITHMS:
        raise ValueError(f"未知策略：{algorithm}")
    if horizon < 0:
        raise ValueError("H 不能为负")
    result = SearchResult()
    pruning = algorithm != "Minimax"

    def visit(current, depth, alpha, beta):
        result.visited_nodes += 1
        if depth == 0 or current.remaining_mask == 0:
            result.leaf_evaluations += 1
            return current.utility, None
        maximizing = current.current_player == 0
        value = -math.inf if maximizing else math.inf
        chosen = None
        targets = actions(game, current, algorithm)
        for index, target in enumerate(targets):
            candidate, _ = visit(transition(game, current, target), depth-1, alpha, beta)
            if chosen is None or (candidate > value if maximizing else candidate < value):
                value, chosen = candidate, target
            if pruning:
                if maximizing:
                    alpha = max(alpha, value)
                else:
                    beta = min(beta, value)
                if alpha >= beta:
                    if index + 1 < len(targets):
                        result.effective_cutoffs += 1
                    break
        return value, chosen

    result.best_value, result.best_target = visit(state, horizon, -math.inf, math.inf)
    return result


def dijkstra_matrix(case):
    """校验专用：无启发的距离搜索，不读取参考矩阵。"""
    costs, positions, _ = read_terrain(case)
    matrix = []
    for source in LABELS:
        best = {positions[source]: 0}
        queue = [(0, positions[source])]
        while queue:
            value, (r, c) = heapq.heappop(queue)
            if best[(r, c)] != value:
                continue
            for v in ((r-1,c), (r+1,c), (r,c-1), (r,c+1)):
                nr, nc = v
                if 0 <= nr < len(costs) and 0 <= nc < len(costs[0]) and costs[nr][nc] >= 0:
                    candidate = value + costs[nr][nc]
                    if candidate < best.get(v, math.inf):
                        best[v] = candidate
                        heapq.heappush(queue, (candidate, v))
        matrix.append(tuple(best[positions[name]] for name in LABELS))
    return tuple(matrix)


def exhaustive_reference(game, state, horizon):
    """枚举行动排列后逐层回传，独立于 search/transition 的递归实现。"""
    available = tuple(i for i in range(4) if state.remaining_mask & (1 << i))
    depth = min(horizon, len(available))
    levels = [{} for _ in range(depth + 1)]
    for sequence in itertools.permutations(available, depth):
        positions = [state.pos_a, state.pos_b]
        scores = [state.score_a, state.score_b]
        for step, target in enumerate(sequence):
            player = (state.current_player + step) % 2
            scores[player] += game.values[target] - game.distances[positions[player]][target+2]
            positions[player] = target+2
        levels[depth][sequence] = scores[0] - scores[1]
    for k in reversed(range(depth)):
        groups = {}
        for prefix, value in levels[k+1].items():
            groups.setdefault(prefix[:-1], []).append(value)
        choose = max if (state.current_player+k) % 2 == 0 else min
        levels[k] = {prefix: choose(values) for prefix, values in groups.items()}
    root_value = levels[0][()]
    first = {prefix[0]: value for prefix, value in levels[1].items()} if depth else {}
    optimum = sorted(t for t, value in first.items() if value == root_value)
    # 输出一条按最小编号打破平局的精确最优应对序列。
    sequence = ()
    for k in range(depth):
        target = min(prefix[-1] for prefix, value in levels[k+1].items()
                     if prefix[:-1] == sequence and value == levels[k][sequence])
        sequence += (target,)
    return {
        "best_value": root_value, "optimal_targets": optimum, "first_move_values": first,
        "principal_sequence": sequence, "visited_nodes": sum(len(level) for level in levels),
        "leaf_evaluations": len(levels[-1]),
    }


def validate_result(result, oracle, algorithm):
    assert result.best_value == oracle["best_value"]
    assert result.best_target in oracle["optimal_targets"] or (
        not oracle["optimal_targets"] and result.best_target is None)
    assert 1 <= result.leaf_evaluations <= oracle["leaf_evaluations"]
    assert result.leaf_evaluations <= result.visited_nodes <= oracle["visited_nodes"]
    if algorithm == "Minimax":
        assert result.visited_nodes == oracle["visited_nodes"]
        assert result.leaf_evaluations == oracle["leaf_evaluations"]
        assert result.effective_cutoffs == 0


def validate_preprocessing(case, prepared):
    game, paths, _ = prepared
    assert game.distances == dijkstra_matrix(case), case["id"]
    expected = case["expected"]
    assert expected["distance_order"] == list(LABELS)
    assert [list(row) for row in game.distances] == expected["distance_matrix"]
    costs, positions, _ = read_terrain(case)
    for pair, entry in paths.items():
        source, target = pair.split("->")
        path = entry["path"]
        assert path[0] == positions[source] and path[-1] == positions[target]
        assert all(abs(a[0]-b[0])+abs(a[1]-b[1]) == 1 for a,b in zip(path,path[1:]))
        assert all(costs[r][c] >= 1 for r,c in path)
        assert sum(costs[r][c] for r,c in path[1:]) == entry["minimum_cost"]


def memory_call(function):
    gc.collect()
    tracemalloc.start()
    try:
        result = function()
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    return result, peak


def benchmark(cases, repeats=7, memory_repeats=3, batch_size=200):
    rng = random.Random(MEASUREMENT_SEED)
    records, preprocessing, diagnostics = [], [], []
    for case in cases:
        original = json.dumps(case, sort_keys=True)
        baseline_prepared = prepare_game(case)
        validate_preprocessing(case, baseline_prepared)
        pre_times, pre_memory = [], []
        for _ in range(repeats):
            gc.collect()
            begin = time.perf_counter_ns()
            prepared = prepare_game(case)
            pre_times.append((time.perf_counter_ns()-begin)/1e6)
            assert prepared == baseline_prepared
        for _ in range(memory_repeats):
            prepared, peak = memory_call(lambda: prepare_game(case))
            pre_memory.append(peak)
            assert prepared == baseline_prepared
        game, paths, expanded = baseline_prepared
        preprocessing.append({
            "case_id": case["id"], "pair_count": 30, "expanded_nodes": expanded,
            "time_median_ms": statistics.median(pre_times),
            "memory_peak_median_kib": statistics.median(pre_memory)/1024,
            "time_samples_ms": pre_times, "memory_samples_bytes": pre_memory,
            "distance_order": LABELS, "distance_matrix": game.distances, "paths": paths,
        })
        state = initial_state(case)
        for horizon in HORIZONS:
            oracle = exhaustive_reference(game, state, horizon)
            expected = case["expected"]["decisions"][str(horizon)]
            assert oracle["best_value"] == expected["best_value"]
            assert [LABELS[i+2] for i in oracle["optimal_targets"]] == expected["optimal_targets"]
            assert {LABELS[i+2]:v for i,v in oracle["first_move_values"].items()} == expected["first_move_values"]
            baseline = {}
            for name in ALGORITHMS:
                result = search(game, state, horizon, name)  # 预热一次
                validate_result(result, oracle, name)
                baseline[name] = result
            # 分项值使用各候选的独立完整窗口求值，诊断成本不计入策略性能。
            per_algorithm = {}
            for name in ALGORITHMS:
                exact = {t: search(game, transition(game,state,t), horizon-1,name).best_value
                         for t in oracle["first_move_values"]}
                assert exact == oracle["first_move_values"]
                per_algorithm[name] = {LABELS[t+2]:v for t,v in exact.items()}
            current = state
            play = []
            for step, target in enumerate(oracle["principal_sequence"]):
                reward = gain(game,current,target)
                old = current
                current = transition(game,current,target)
                play.append({
                    "step": step+1, "player": ("A","B")[old.current_player],
                    "target": LABELS[target+2], "gain": reward,
                    "move_cost": game.values[target]-reward,
                    "state_after": asdict(current), "utility_after": current.utility,
                })
            assert current.utility == oracle["best_value"]
            diagnostics.append({
                "case_id": case["id"], "horizon": horizon,
                "best_value": oracle["best_value"],
                "optimal_targets": [LABELS[t+2] for t in oracle["optimal_targets"]],
                "first_move_values": {LABELS[t+2]:v for t,v in oracle["first_move_values"].items()},
                "full_window_checks": per_algorithm, "optimal_play": play,
            })
            timing = {name: [] for name in ALGORITHMS}
            memory = {name: [] for name in ALGORITHMS}
            for _ in range(repeats):
                names = list(ALGORITHMS)
                rng.shuffle(names)
                for name in names:
                    gc.collect()
                    begin = time.perf_counter_ns()
                    for _ in range(batch_size):
                        result = search(game, state, horizon, name)
                    timing[name].append((time.perf_counter_ns()-begin)/1e6)
                    validate_result(result, oracle, name)
                    assert result == baseline[name]
            for _ in range(memory_repeats):
                names = list(ALGORITHMS)
                rng.shuffle(names)
                for name in names:
                    result, peak = memory_call(lambda: search(game,state,horizon,name))
                    memory[name].append(peak)
                    validate_result(result, oracle, name)
                    assert result == baseline[name]
            for name in ALGORITHMS:
                result = baseline[name]
                records.append({
                    "case_id": case["id"], "kind": case["kind"], "horizon": horizon,
                    "algorithm": name, "correct": True,
                    "best_target": LABELS[result.best_target+2] if result.best_target is not None else None,
                    "best_value": result.best_value,
                    "visited_nodes": result.visited_nodes, "leaf_evaluations": result.leaf_evaluations,
                    "effective_cutoffs": result.effective_cutoffs,
                    "node_saving_percent": 100*(1-result.visited_nodes/oracle["visited_nodes"]),
                    "time_median_ms": statistics.median(timing[name])/batch_size,
                    "memory_peak_median_kib": statistics.median(memory[name])/1024,
                    "time_batch_samples_ms": timing[name],
                    "time_samples_ms": [value/batch_size for value in timing[name]],
                    "memory_samples_bytes": memory[name],
                })
        assert json.dumps(case,sort_keys=True) == original
    return records, preprocessing, diagnostics


METRICS = ("visited_nodes", "leaf_evaluations", "node_saving_percent",
           "time_median_ms", "memory_peak_median_kib")


def aggregate(records):
    groups = {}
    for row in records:
        for dimension,group in (("overall","all"),("horizon",str(row["horizon"])),
                                ("case_horizon",f"{row['case_id']}:H{row['horizon']}")):
            groups.setdefault((dimension,group,row["algorithm"]),[]).append(row)
    result = []
    for (dimension,group,name),rows in groups.items():
        result.append({
            "dimension":dimension, "group":group, "algorithm":name,
            "case_count":len(rows), "correct_count":sum(row["correct"] for row in rows),
            **{metric+"_mean":statistics.mean(row[metric] for row in rows) for metric in METRICS},
        })
    return result


def write_csv(path, rows):
    with path.open("w",encoding="utf-8-sig",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_outputs(output, records, preprocessing, diagnostics, summary, primary):
    write_csv(output/"details_v4.0.csv",[
        {k:v for k,v in row.items() if not k.endswith("_samples_ms") and k!="memory_samples_bytes"}
        for row in records])
    write_csv(output/"summary_v4.0.csv",summary)
    write_csv(output/"distances_v4.0.csv",[
        {"case_id":p["case_id"],"source":source,**dict(zip(LABELS,row))}
        for p in preprocessing for source,row in zip(LABELS,p["distance_matrix"])])
    write_csv(output/"preprocessing_v4.0.csv",[
        {k:v for k,v in row.items() if k in ("case_id","pair_count","expanded_nodes",
                                           "time_median_ms","memory_peak_median_kib")}
        for row in preprocessing])
    write_csv(output/"first_moves_v4.0.csv",[
        {"case_id":d["case_id"],"horizon":d["horizon"],"target":t,"value":value,
         "is_optimal":t in d["optimal_targets"]}
        for d in diagnostics for t,value in d["first_move_values"].items()])
    lines=["# Quiz4：A* 与博弈搜索对照","",
           f"主实验：{primary}。其余地图用于规则检查；全部组合 {len(records)} 项均通过独立校验。","",
           "## 距离矩阵","", "| 出发 / 到达 | "+" | ".join(LABELS)+" |",
           "| --- | "+" | ".join(["---:"]*6)+" |"]
    pre=next(p for p in preprocessing if p["case_id"]==primary)
    for source,row in zip(LABELS,pre["distance_matrix"]):
        lines.append("| "+source+" | "+" | ".join(map(str,row))+" |")
    lines += ["",f"A* 预处理：30 个有序点对，扩展 {pre['expanded_nodes']} 个节点；"
              f"耗时中位数 {pre['time_median_ms']:.4f} ms，峰值 {pre['memory_peak_median_kib']:.2f} KiB。",
              "预处理独立测量后共享距离矩阵，以下耗时与内存仅计博弈搜索。","",
              "## 主地图决策与五项性能指标","",
              "| H | 策略 | 首步 | U | 访问节点 | 叶求值 | 节省率/% | 耗时/ms | 峰值/KiB |",
              "| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for row in records:
        if row["case_id"]==primary:
            lines.append(f"| {row['horizon']} | {DISPLAY[row['algorithm']]} | {row['best_target']} "
                         f"| {row['best_value']} | {row['visited_nodes']} | {row['leaf_evaluations']} "
                         f"| {row['node_saving_percent']:.2f} | {row['time_median_ms']:.4f} "
                         f"| {row['memory_peak_median_kib']:.2f} |")
    lines += ["","## 首步精确收益差","", "| H | T1 | T2 | T3 | T4 | 最优首步集合 |",
              "| ---: | ---: | ---: | ---: | ---: | --- |"]
    for d in diagnostics:
        if d["case_id"]!=primary:
            continue
        lines.append(f"| {d['horizon']} | "+" | ".join(str(d["first_move_values"][t]) for t in LABELS[2:])
                     +" | "+", ".join(d["optimal_targets"])+" |")
    lines += ["","## 精确最优应对序列","",
              "| H | 行动 | 玩家 | 目标 | 移动代价 | 净收益 | 累计 U |",
              "| ---: | ---: | --- | --- | ---: | ---: | ---: |"]
    for d in diagnostics:
        if d["case_id"]==primary:
            for step in d["optimal_play"]:
                lines.append(f"| {d['horizon']} | {step['step']} | {step['player']} | {step['target']} "
                             f"| {step['move_cost']} | {step['gain']} | {step['utility_after']} |")
    lines += ["","计时按多次搜索组成一批，批耗时除以批大小，再取各批中位数；排序在搜索内执行。",
              "内存为独立单次搜索的 tracemalloc 新增分配峰值，预载地图、距离矩阵及参考求值不计入。",
              "平局保留当前顺序中首个最优行动；最优集合由独立完整求值记录，策略所选首步均属于该集合。",
              "首步分项值单独使用完整窗口核验，诊断和最优序列恢复不计入策略性能。",
              "H=2 的 U 是截断局面的收益差，H=4 的 U 是四次行动后的最终收益差。"]
    (output/"comparison_v4.0.md").write_text("\n".join(lines)+"\n",encoding="utf-8")


def plot_results(records, diagnostics, primary, output, repeats, batch_size):
    os.environ.setdefault("MPLCONFIGDIR",str(LAB2/".mplconfig"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colors=("#2878B5","#F28E2B","#31965B","#D75455")
    chosen={(row["horizon"],row["algorithm"]):row for row in records if row["case_id"]==primary}
    decisions={d["horizon"]:d for d in diagnostics if d["case_id"]==primary}
    with plt.rc_context({"font.family":"sans-serif",
                         "font.sans-serif":["Microsoft YaHei","SimHei","Noto Sans CJK SC","DejaVu Sans"],
                         "axes.unicode_minus":False,"font.size":10,
                         "axes.spines.top":False,"axes.spines.right":False}):
        fig,axes=plt.subplots(1,3,figsize=(14,4.8))
        for offset,h,color in ((-.18,2,colors[0]),(.18,4,colors[1])):
            values=[decisions[h]["first_move_values"][t] for t in LABELS[2:]]
            bars=axes[0].bar([i+offset for i in range(4)],values,width=.34,color=color,label=f"H={h}")
            for bar,value in zip(bars,values):
                axes[0].annotate(str(value),(bar.get_x()+bar.get_width()/2,value),
                                 xytext=(0,4 if value>=0 else -4),textcoords="offset points",
                                 ha="center",va="bottom" if value>=0 else "top",fontsize=9)
        axes[0].axhline(0,color="#718096",linewidth=.8)
        axes[0].set_xticks(range(4),LABELS[2:])
        axes[0].set_ylim(min(d["first_move_values"][t] for d in decisions.values() for t in LABELS[2:])-1.5,
                         max(d["first_move_values"][t] for d in decisions.values() for t in LABELS[2:])+1.5)
        axes[0].set_title("各首步的精确收益差")
        axes[0].set_xlabel("A 的首个目标")
        axes[0].set_ylabel("U = Score_A − Score_B")
        axes[0].legend(frameon=False)
        for ax,metric,title,ylabel in (
            (axes[1],"visited_nodes","访问节点数","节点数"),
            (axes[2],"time_median_ms","博弈搜索耗时","耗时（ms）"),
        ):
            for index,(name,color) in enumerate(zip(ALGORITHMS,colors)):
                values=[chosen[(h,name)][metric] for h in HORIZONS]
                ax.bar([i+(index-1.5)*.18 for i in range(2)],values,width=.17,color=color,label=DISPLAY[name])
            ax.set_xticks([0,1],["H=2","H=4"])
            ax.set_title(title)
            ax.set_xlabel("搜索深度（单次行动数）")
            ax.set_ylabel(ylabel)
            ax.set_ylim(bottom=0)
        handles,labels=axes[1].get_legend_handles_labels()
        fig.legend(handles,labels,loc="upper center",bbox_to_anchor=(.5,.92),ncol=4,frameon=False,fontsize=9)
        for ax in axes:
            ax.grid(axis="y",alpha=.18)
            ax.set_axisbelow(True)
        fig.suptitle(f"宝藏分配：深度、首步决策与搜索开销｜{primary}",fontsize=14,y=.995)
        fig.text(.5,.025,f"仅展示主地图；其余地图见表格。耗时为 {repeats} 批中位数，每批 {batch_size} 次搜索后折算单次耗时。\n"
                 "A* 预处理单独记录；排序计入搜索；H=2 为截断局面评价，H=4 为完整四次行动结果。",
                 ha="center",fontsize=9)
        fig.tight_layout(rect=(0,.13,1,.85),w_pad=2.0)
        for extension in ("png","svg"):
            fig.savefig(output/f"performance_comparison_v4.0.{extension}",dpi=220)
        plt.close(fig)
    return matplotlib.__version__


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--map",type=Path,default=Path(__file__).with_name("map"))
    parser.add_argument("--output",type=Path,default=LAB2/"result/quiz4")
    parser.add_argument("--case",help="仅运行指定用例 ID；默认运行全部地图")
    parser.add_argument("--repeats",type=int,default=7)
    parser.add_argument("--memory-repeats",type=int,default=3)
    parser.add_argument("--batch-size",type=int,default=200)
    args=parser.parse_args()
    if min(args.repeats,args.memory_repeats,args.batch_size)<1:
        parser.error("重复次数及批大小应大于 0")
    raw=args.map.read_bytes()
    dataset=json.loads(raw.decode("utf-8"))
    cases=[c for c in dataset["cases"] if args.case is None or c["id"]==args.case]
    if not cases:
        parser.error("没有匹配的用例")
    primary=dataset.get("primary_case_id",cases[0]["id"])
    if primary not in {c["id"] for c in cases}:
        primary=cases[0]["id"]
    records,preprocessing,diagnostics=benchmark(cases,args.repeats,args.memory_repeats,args.batch_size)
    summary=aggregate(records)
    args.output.mkdir(parents=True,exist_ok=True)
    write_outputs(args.output,records,preprocessing,diagnostics,summary,primary)
    mpl=plot_results(records,diagnostics,primary,args.output,args.repeats,args.batch_size)
    metadata={
        "created_at":datetime.now().astimezone().isoformat(),"python":sys.version,
        "platform":platform.platform(),"matplotlib":mpl,"case_count":len(cases),
        "primary_case_id":primary,"map_path":str(args.map.resolve()),
        "map_sha256":hashlib.sha256(raw).hexdigest(),"horizons":HORIZONS,
        "repeats":args.repeats,"memory_repeats":args.memory_repeats,"batch_size":args.batch_size,
        "measurement_seed":MEASUREMENT_SEED,"timer":"perf_counter_ns","gc_enabled":gc.isenabled(),
        "preprocessing":"30 ordered A* pairs; terrain conversion and path recovery included; separately measured",
        "search_timing":"median batch time / batch size; state transitions, sorting and counters included",
        "memory":"separate single-run tracemalloc peak; preloaded map/matrix/oracles excluded",
        "validation":"Dijkstra + path costs + input references; exhaustive permutations; full-window first-move checks",
        "tie_policy":"retain first visited optimum; all exact optima recorded separately",
        "diagnostics":"first-move evaluation and optimal-play reconstruction outside performance measurement",
        "aggregation":"per-case values averaged equally; savings computed against same-case same-H Minimax",
        "command":[sys.executable]+sys.argv,
    }
    (args.output/"results_v4.0.json").write_text(json.dumps({
        "metadata":metadata,"records":records,"preprocessing":preprocessing,
        "diagnostics":diagnostics,"summary":summary,
    },ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(f"All {len(records)} combinations passed. Results: {args.output.resolve()}")


if __name__=="__main__":
    main()
