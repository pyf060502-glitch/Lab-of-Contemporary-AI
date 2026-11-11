"""四种策略的独立正确性、停止条件及统计口径检查。"""

import itertools
import random
import runpy
import unittest
from collections import deque
from pathlib import Path


SEARCH = runpy.run_path(str(Path(__file__).with_name("v1.1.py")))


def oracle(grid, start, end):
    if grid[start[0]][start[1]] == "X" or grid[end[0]][end[1]] == "X":
        return None
    queue, seen = deque([(start, 0)]), {start}
    while queue:
        (row, col), distance = queue.popleft()
        if (row, col) == end:
            return distance
        for r, c in ((row-1,col), (row+1,col), (row,col-1), (row,col+1)):
            if 0 <= r < len(grid) and 0 <= c < len(grid[0]) and grid[r][c] != "X" and (r,c) not in seen:
                seen.add((r,c))
                queue.append(((r,c), distance+1))
    return None


class BidirectionalTests(unittest.TestCase):
    def check_case(self, grid, start, end):
        expected = oracle(grid, start, end)
        original = [row[:] for row in grid]
        for algorithm in SEARCH["ALGORITHMS"]:
            result = SEARCH["search"](start, end, grid, algorithm)
            SEARCH["validate"](result, grid, start, end, expected)
            self.assertEqual(result.expanded_nodes, result.forward_expanded + result.backward_expanded)
            self.assertEqual(grid, original)
            if algorithm == "A* Manhattan":
                self.assertEqual(result.backward_expanded, 0)
            if result.path:
                self.assertEqual(len(result.path), len(set(result.path)))

    def test_exhaustive_small_maps(self):
        for size in (2, 3):
            endpoints = list(itertools.product(range(size), repeat=2))
            for mask in range(1 << (size*size)):
                grid = [["X" if mask & (1 << (r*size+c)) else "." for c in range(size)] for r in range(size)]
                for start, end in itertools.product(endpoints, repeat=2):
                    self.check_case(grid, start, end)

    def test_random_maps_and_endpoint_directions(self):
        rng = random.Random(20261109)
        for _ in range(200):
            rows, cols = rng.randrange(4,16), rng.randrange(4,16)
            grid = [["X" if rng.random()<0.28 else "." for _ in range(cols)] for _ in range(rows)]
            start, end = (rng.randrange(rows),rng.randrange(cols)), (rng.randrange(rows),rng.randrange(cols))
            grid[start[0]][start[1]] = grid[end[0]][end[1]] = "."
            self.check_case(grid, start, end)
            self.check_case(grid, end, start)

    def test_first_connection_is_not_always_optimal(self):
        grid = list(map(list, [
            "SXX.X...X...", ".X..X..XXX..", "..XX...X...X", "...X.....XX.",
            "..XX..XX...X", "..XX....XX..", "....X...XX..", "..XX.X......",
            ".XXX....X...", "...XX.X...X.", ".X....X.....", "..X.....X.XE",
        ]))
        result = SEARCH["search"]((0,0), (11,11), grid, "Bi-A* Manhattan")
        self.assertEqual(result.first_connection_steps, 26)
        self.assertEqual(len(result.path)-1, 24)
        self.assertEqual(result.termination, "lower_bound")

    def test_same_endpoint_has_no_expansion(self):
        for algorithm in SEARCH["ALGORITHMS"]:
            result = SEARCH["search"]((0,0), (0,0), [["."]], algorithm)
            self.assertEqual(result.path, [(0,0)])
            self.assertEqual(result.expanded_nodes, 0)


if __name__ == "__main__":
    unittest.main()
