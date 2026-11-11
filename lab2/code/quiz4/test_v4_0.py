"""Quiz4：独立最短路、博弈穷举、转移与指标检查。"""

import copy
import itertools
import json
import random
import runpy
import unittest
from pathlib import Path


M = runpy.run_path(str(Path(__file__).with_name("v4.0.py")))
State, Game = M["State"], M["Game"]
ALGORITHMS = M["ALGORITHMS"]


def reference(game, state, horizon):
    """测试专用：使用位置、分数元组递归，不调用被测状态转移。"""
    def visit(a,b,remaining,sa,sb,player,depth):
        if depth == 0 or not remaining:
            return sa-sb, {}
        scores={}
        for t in remaining:
            benefit=game.values[t]-game.distances[a if player==0 else b][t+2]
            rest=tuple(v for v in remaining if v!=t)
            if player==0:
                value,_=visit(t+2,b,rest,sa+benefit,sb,1,depth-1)
            else:
                value,_=visit(a,t+2,rest,sa,sb+benefit,0,depth-1)
            scores[t]=value
        return (max if player==0 else min)(scores.values()),scores
    remaining=tuple(i for i in range(4) if state.remaining_mask&(1<<i))
    return visit(state.pos_a,state.pos_b,remaining,state.score_a,state.score_b,
                 state.current_player,horizon)


def floyd_matrix(case):
    grid=case["grid"]
    labels=M["LABELS"]
    vertices=[(r,c) for r,row in enumerate(grid) for c,cell in enumerate(row) if cell!="X"]
    index={v:i for i,v in enumerate(vertices)}
    points={cell:(r,c) for r,row in enumerate(grid) for c,cell in enumerate(row) if cell in labels}
    d=[[float("inf")]*len(vertices) for _ in vertices]
    for r,c in vertices:
        i=index[(r,c)]
        d[i][i]=0
        for v in ((r-1,c),(r+1,c),(r,c-1),(r,c+1)):
            if v in index:
                cell=grid[v[0]][v[1]]
                d[i][index[v]]=1 if cell in labels else int(cell)
    for k in range(len(vertices)):
        for i in range(len(vertices)):
            for j in range(len(vertices)):
                d[i][j]=min(d[i][j],d[i][k]+d[k][j])
    return tuple(tuple(d[index[points[a]]][index[points[b]]] for b in labels) for a in labels)


class TreasureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cases=json.loads(Path(__file__).with_name("map").read_text(encoding="utf-8"))["cases"]

    def check_game(self,game,state,horizon):
        value,first=reference(game,state,horizon)
        optimal={t for t,v in first.items() if v==value}
        oracle=M["exhaustive_reference"](game,state,horizon)
        self.assertEqual(oracle["best_value"],value)
        self.assertEqual(oracle["first_move_values"],first)
        for name in ALGORITHMS:
            result=M["search"](game,state,horizon,name)
            M["validate_result"](result,oracle,name)
            self.assertEqual(result.best_value,value)
            self.assertIn(result.best_target,optimal if optimal else {None})
        return oracle

    def test_all_map_distances_and_decisions(self):
        for case in self.cases:
            before=copy.deepcopy(case)
            prepared=M["prepare_game"](case)
            M["validate_preprocessing"](case,prepared)
            game=prepared[0]
            self.assertEqual(game.distances,floyd_matrix(case))
            for horizon in (2,4):
                oracle=self.check_game(game,M["initial_state"](case),horizon)
                self.assertEqual(oracle["visited_nodes"],17 if horizon==2 else 65)
                self.assertEqual(oracle["leaf_evaluations"],12 if horizon==2 else 24)
            self.assertEqual(case,before)

    def test_given_example_and_full_optimal_sequence(self):
        game=M["prepare_game"](self.cases[0])[0]
        self.assertEqual(game.distances,(
            (0,12,10,4,8,8),(12,0,8,8,4,10),(10,8,0,6,6,12),
            (4,8,6,0,6,6),(8,4,6,6,0,6),(8,10,12,6,6,0)))
        for h,target,value,first in [
            (2,1,-1,[-4,-1,-2,-4]),(4,2,3,[-5,1,3,-7]),
        ]:
            result=M["search"](game,State(),h)
            self.assertEqual((result.best_target,result.best_value),(target,value))
            oracle=M["exhaustive_reference"](game,State(),h)
            self.assertEqual([oracle["first_move_values"][t] for t in range(4)],first)
        self.assertEqual(M["exhaustive_reference"](game,State(),4)["principal_sequence"],(2,0,1,3))

    def test_pass_through_negative_gain_and_position_updates(self):
        case=next(c for c in self.cases if c["kind"]=="pass_through")
        game,paths,_=M["prepare_game"](case)
        self.assertEqual(game.distances[0][2],1)  # 进入宝藏收费
        moved=M["transition"](game,State(),3)  # 经过 T1/T2/T3，只获得 T4
        self.assertEqual((moved.pos_a,moved.remaining_mask,moved.score_a),(5,7,5))
        moved=M["transition"](game,moved,0)
        self.assertEqual((moved.pos_b,moved.remaining_mask,moved.score_b),(2,6,-1))
        again=M["transition"](game,moved,1)
        self.assertEqual(again.score_a,5+(5-2))  # A 从 T4 出发，非初始 A
        self.assertEqual(moved.score_a,5)
        with self.assertRaises(ValueError):
            M["transition"](game,moved,0)
        negative=next(c for c in self.cases if c["kind"]=="negative_gains")
        game=M["prepare_game"](negative)[0]
        self.assertTrue(all(M["gain"](game,State(),t)<0 for t in range(4)))
        self.assertEqual(len(M["actions"](game,State(),"AB gain")),4)

    def test_ordering_both_players_and_equal_scores(self):
        case=next(c for c in self.cases if c["kind"]=="ties")
        game=M["prepare_game"](case)[0]
        oracle=self.check_game(game,State(),4)
        self.assertGreater(len(oracle["optimal_targets"]),1)
        for player in (0,1):
            state=State(current_player=player)
            order=M["actions"](game,state,"AB gain")
            self.assertEqual(order,sorted(range(4),key=lambda t:(-M["gain"](game,state,t),t)))
            self.assertEqual(M["actions"](game,state,"AB reverse"),[3,2,1,0])

    def test_small_value_combinations_and_directed_random_games(self):
        distances=tuple(tuple(abs(i-j) for j in range(6)) for i in range(6))
        for values in itertools.product((-2,0,4),repeat=4):
            game=Game(distances,values)
            for h in (1,2,3,4):
                self.check_game(game,State(),h)
        rng=random.Random(20261110)
        for _ in range(150):
            matrix=tuple(tuple(0 if i==j else rng.randint(1,12) for j in range(6)) for i in range(6))
            game=Game(matrix,tuple(rng.randint(-5,15) for _ in range(4)))
            state=State(rng.randrange(6),rng.randrange(6),rng.randrange(16),
                        rng.randint(-9,9),rng.randint(-9,9),rng.randrange(2))
            self.check_game(game,state,rng.randrange(5))

    def test_weighted_detour_and_heuristic_consistency(self):
        case=next(c for c in self.cases if c["kind"]=="weighted_detour")
        game,paths,_=M["prepare_game"](case)
        self.assertEqual(game.distances[0][2],8)
        self.assertEqual(len(paths["A->T1"]["path"])-1,8)  # 四步直走代价 11，八步绕行代价 8
        for case in self.cases:
            costs,points,_=M["read_terrain"](case)
            for goal in points.values():
                h=lambda r,c:abs(r-goal[0])+abs(c-goal[1])
                for r,row in enumerate(costs):
                    for c,w in enumerate(row):
                        if w<0:
                            continue
                        for nr,nc in ((r-1,c),(r+1,c),(r,c-1),(r,c+1)):
                            if 0<=nr<len(costs) and 0<=nc<len(row) and costs[nr][nc]>0:
                                self.assertLessEqual(h(r,c),costs[nr][nc]+h(nr,nc))

    def test_batched_measurement_and_aggregation(self):
        rows,pre,diag=M["benchmark"]([self.cases[0]],repeats=2,memory_repeats=2,batch_size=3)
        self.assertEqual(len(rows),8)
        self.assertEqual(pre[0]["pair_count"],30)
        self.assertEqual(len(diag),2)
        for row in rows:
            self.assertTrue(row["correct"])
            self.assertEqual(len(row["time_samples_ms"]),2)
            self.assertEqual(len(row["memory_samples_bytes"]),2)
            self.assertGreater(row["memory_peak_median_kib"],0)
            for batch,single in zip(row["time_batch_samples_ms"],row["time_samples_ms"]):
                self.assertAlmostEqual(batch/3,single)
        summary=M["aggregate"](rows)
        full=next(r for r in summary if r["dimension"]=="overall" and r["algorithm"]=="Minimax")
        self.assertEqual(full["visited_nodes_mean"],41)
        self.assertEqual(full["node_saving_percent_mean"],0)


if __name__=="__main__":
    unittest.main()
