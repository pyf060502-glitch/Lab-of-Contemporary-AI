"""地标距离方向、下界、嵌套选点和阶段测量检查。"""

import itertools
import random
import runpy
import unittest
from pathlib import Path


MODULE = runpy.run_path(str(Path(__file__).with_name("v2.1.py")))
ORACLE = runpy.run_path(str(Path(__file__).with_name("test_v2_0.py")))["oracle"]


def bellman_distances(data, origin, reverse=False):
    distances = [float("inf")] * len(data.costs)
    distances[origin] = 0
    edges = []
    for r in range(data.rows):
        for c in range(data.cols):
            u = r*data.cols+c
            if data.costs[u]<0:
                continue
            for nr,nc in ((r-1,c),(r+1,c),(r,c-1),(r,c+1)):
                if 0<=nr<data.rows and 0<=nc<data.cols:
                    v = nr*data.cols+nc
                    if data.costs[v]>=0:
                        edges.append((v,u,data.costs[v]) if reverse else (u,v,data.costs[v]))
    for _ in range(len(data.costs)-1):
        changed = False
        for u,v,w in edges:
            if distances[u]+w<distances[v]:
                distances[v]=distances[u]+w
                changed = True
        if not changed:
            break
    return distances


class LandmarkTests(unittest.TestCase):
    def check_case(self, grid, start, end, start_cost=1):
        expected = ORACLE(grid,start,end,start_cost)
        original = [row[:] for row in grid]
        prepared = {}
        for algorithm in MODULE["ALGORITHMS"]:
            data = MODULE["prepare"](start,end,grid,algorithm,start_cost)
            result = MODULE["query"](data)
            MODULE["validate"](result,grid,start,end,start_cost,expected)
            self.assertEqual(grid,original)
            self.assertEqual(MODULE["query"](data),result)
            prepared[algorithm]=data
        two,four = prepared["ALT A* (2)"],prepared["ALT A* (4)"]
        self.assertEqual(two.landmarks,four.landmarks[:2])
        if not four.costs:
            return
        true = bellman_distances(four,four.end,reverse=True)
        for index,landmark in enumerate(four.landmarks):
            self.assertEqual(four.tables[index][0],bellman_distances(four,landmark))
            self.assertEqual(four.tables[index][1],bellman_distances(four,landmark,reverse=True))
        for u,cost in enumerate(four.costs):
            if cost<0:
                continue
            h0=MODULE["heuristic"](u,prepared["A* scaled"])
            h2=MODULE["heuristic"](u,two)
            h4=MODULE["heuristic"](u,four)
            self.assertLessEqual(h0,h2)
            self.assertLessEqual(h2,h4)
            self.assertLessEqual(h4,true[u])
            for v in MODULE["adjacent"](u,four.rows,four.cols):
                if four.costs[v]>=0:
                    self.assertLessEqual(h4,four.costs[v]+MODULE["heuristic"](v,four))

    def test_all_two_by_two_weighted_maps(self):
        for values in itertools.product("013X",repeat=4):
            grid=[list(values[:2]),list(values[2:])]
            self.check_case(grid,(0,0),(1,1))

    def test_random_maps_and_endpoints(self):
        rng=random.Random(20261109)
        for _ in range(100):
            rows,cols=rng.randrange(2,6),rng.randrange(2,6)
            grid=[[rng.choice("001238X") for _ in range(cols)] for _ in range(rows)]
            grid[0][0],grid[-1][-1]="S","E"
            self.check_case(grid,(0,0),(rows-1,cols-1),rng.choice((1,2,3)))

    def test_free_goal_and_directed_distances(self):
        data=MODULE["prepare"]((0,0),(0,2),[list("S3E")],"ALT A* (4)",1)
        forward,_=MODULE["all_distances"](data.costs,1,3,0)
        reverse,_=MODULE["all_distances"](data.costs,1,3,0,reverse=True)
        self.assertEqual(forward[1],3)
        self.assertEqual(reverse[1],1)
        from_goal,_=MODULE["all_distances"](data.costs,1,3,2,reverse=True)
        self.assertEqual(from_goal,[3,0,0])
        self.check_case([list("S")],(0,0),(0,0),3)
        self.check_case([list("S000E")],(0,0),(0,4))
        self.check_case([list("SXE")],(0,0),(0,2))

    def test_goal_change_rebuilds_weighted_graph(self):
        grid=[list("138")]
        a=MODULE["prepare"]((0,0),(0,2),grid,"ALT A* (2)")
        b=MODULE["prepare"]((0,0),(0,1),grid,"ALT A* (2)")
        self.assertEqual(a.landmarks,b.landmarks)
        self.assertNotEqual(a.costs,b.costs)
        self.assertEqual(MODULE["query"](a).minimum_cost,3)
        self.assertEqual(MODULE["query"](b).minimum_cost,0)

    def test_stage_measurements(self):
        for name in MODULE["ALGORITHMS"]:
            result,times=MODULE["timed_run"]((0,0),(0,2),[list("S3E")],name,1)
            self.assertEqual(result.minimum_cost,3)
            self.assertAlmostEqual(times[0]+times[1],times[2])
            result,memory=MODULE["memory_run"]((0,0),(0,2),[list("S3E")],name,1)
            self.assertEqual(memory["total_peak"],max(memory["preprocessing_peak"],memory["query_peak_with_index"]))
            self.assertGreaterEqual(memory["query_peak_with_index"],memory["retained_preprocessing"])


if __name__ == "__main__":
    unittest.main()
