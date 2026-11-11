"""Quiz2：独立 Bellman-Ford 对照、启发函数及计数检查。"""

import itertools
import random
import runpy
import unittest
from pathlib import Path


SEARCH = runpy.run_path(str(Path(__file__).with_name("v2.0.py")))


def oracle(grid, start, end, start_cost):
    if grid[start[0]][start[1]] == "X" or grid[end[0]][end[1]] == "X":
        return -1
    vertices = [(r,c) for r in range(len(grid)) for c in range(len(grid[0])) if grid[r][c] != "X"]
    edges = []
    for r,c in vertices:
        for v in ((r-1,c),(r+1,c),(r,c-1),(r,c+1)):
            if v in vertices:
                value = grid[v[0]][v[1]]
                weight = 0 if v == end else start_cost if value == "S" else int(value)
                edges.append(((r,c),v,weight))
    distances = {v:float("inf") for v in vertices}
    distances[start] = 0
    for _ in range(len(vertices)-1):
        changed = False
        for u,v,weight in edges:
            candidate = distances[u]+weight
            if candidate < distances[v]:
                distances[v] = candidate
                changed = True
        if not changed:
            break
    return -1 if distances[end] == float("inf") else distances[end]


class WeightedSearchTests(unittest.TestCase):
    def check_case(self, grid, start, end, start_cost=1):
        original = [row[:] for row in grid]
        expected = oracle(grid, start, end, start_cost)
        results = []
        for algorithm in SEARCH["ALGORITHMS"]:
            result = SEARCH["search"](start, end, grid, algorithm, start_cost)
            SEARCH["validate"](result, grid, start, end, start_cost, expected)
            self.assertEqual(grid, original)
            if result.min_non_goal_cost == 0:
                self.assertEqual(result.heuristic_scale, 0)
            results.append(result)
        if results[0].min_non_goal_cost == 1:
            self.assertEqual(results[1], results[2])
        return results

    def test_all_two_by_two_weighted_maps(self):
        positions = list(itertools.product(range(2), repeat=2))
        for cells in itertools.product("013X", repeat=4):
            grid = [list(cells[:2]),list(cells[2:])]
            for start,end in itertools.product(positions,repeat=2):
                self.check_case(grid,start,end)

    def test_random_rectangular_weighted_maps(self):
        rng = random.Random(20261109)
        for _ in range(120):
            rows,cols = rng.randrange(2,7),rng.randrange(2,7)
            grid = [[rng.choice("001238X") for _ in range(cols)] for _ in range(rows)]
            start,end = (0,0),(rows-1,cols-1)
            grid[0][0],grid[-1][-1] = "S","E"
            results = self.check_case(grid,start,end,rng.choice((1,2,3)))
            costs,minimum = SEARCH["prepare_terrain"](start,end,grid,1)
            for scale in (0,1 if minimum>=1 else 0,minimum):
                for r in range(rows):
                    for c in range(cols):
                        if costs[r][c]<0:
                            continue
                        for nr,nc in ((r-1,c),(r+1,c),(r,c-1),(r,c+1)):
                            if 0<=nr<rows and 0<=nc<cols and costs[nr][nc]>=0:
                                self.assertLessEqual(SEARCH["heuristic"]((r,c),end,scale),costs[nr][nc]+SEARCH["heuristic"]((nr,nc),end,scale))

    def test_example_and_boundary_costs(self):
        cases = [(["S1181","18311","18181","11181","8811E"],(4,4),1,7),
                 (["SE"],(0,1),1,0), (["S"],(0,0),3,0),
                 (["S381E"],(0,4),2,12), (["S000E"],(0,4),1,0),
                 (["SXE"],(0,2),1,-1)]
        for rows,end,start_cost,expected in cases:
            for result in self.check_case(list(map(list,rows)),(0,0),end,start_cost):
                self.assertEqual(result.minimum_cost,expected)

    def test_cost_optimum_can_use_more_steps(self):
        results = self.check_case(list(map(list,["S888E","11111"])),(0,0),(0,4))
        for result in results:
            self.assertEqual(result.minimum_cost,5)
            self.assertEqual(len(result.path)-1,6)

    def test_expansion_count_excludes_goal(self):
        for algorithm in SEARCH["ALGORITHMS"]:
            result = SEARCH["search"]((0,0),(0,1),[list("SE")],algorithm)
            self.assertEqual((result.expanded_nodes,result.popped_entries,result.frontier_peak),(1,2,1))
            result = SEARCH["search"]((0,0),(0,0),[["S"]],algorithm)
            self.assertEqual((result.expanded_nodes,result.popped_entries,result.frontier_peak),(0,1,1))


if __name__ == "__main__":
    unittest.main()
