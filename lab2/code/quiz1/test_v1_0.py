"""运行：py -3 -m unittest discover -s 本目录 -p test_v1_0.py。"""

import itertools
import runpy
import unittest
from collections import deque
from pathlib import Path


SEARCH = runpy.run_path(str(Path(__file__).with_name("v1.0.py")))


def oracle(grid, start, end):
    if grid[start[0]][start[1]] == "X" or grid[end[0]][end[1]] == "X":
        return None
    queue = deque([(start, 0)])
    seen = {start}
    while queue:
        (row, col), distance = queue.popleft()
        if (row, col) == end:
            return distance
        for position in ((row-1, col), (row+1, col), (row, col-1), (row, col+1)):
            r, c = position
            if 0 <= r < len(grid) and 0 <= c < len(grid[0]) and grid[r][c] != "X" and position not in seen:
                seen.add(position)
                queue.append((position, distance + 1))
    return None


class SearchTests(unittest.TestCase):
    def test_exhaustive_small_maps(self):
        # 2x2：所有障碍布局、所有起终点；3x3：所有布局、四个角的起终点。
        for size in (2, 3):
            endpoints = list(itertools.product(range(size), repeat=2)) if size == 2 else [(0,0), (0,2), (2,0), (2,2)]
            for mask in range(1 << (size * size)):
                grid = [["X" if mask & (1 << (r*size+c)) else "." for c in range(size)] for r in range(size)]
                original = [row[:] for row in grid]
                for start, end in itertools.product(endpoints, repeat=2):
                    distance = oracle(grid, start, end)
                    for algorithm in SEARCH["ALGORITHMS"]:
                        result = SEARCH["search"](start, end, grid, algorithm)
                        SEARCH["validate"](result, grid, start, end, distance)
                        self.assertEqual(grid, original)

    def test_expansion_definition(self):
        for algorithm in SEARCH["ALGORITHMS"]:
            for grid, end, expected in [([list("S")], (0,0), (0,1,1)),
                                        ([list("S.E")], (0,2), (2,3,1)),
                                        ([list("SXE")], (0,2), (1,1,1))]:
                result = SEARCH["search"]((0,0), end, grid, algorithm)
                self.assertEqual((result.expanded_nodes, result.popped_entries, result.frontier_peak), expected)

    def test_validator_rejects_invalid_paths(self):
        grid = [list("S.E")]
        for path in [[(0,0), (0,2)], [(0,1), (0,2)], [(0,0), (1,0), (0,2)], []]:
            with self.assertRaises(AssertionError):
                SEARCH["validate"](SEARCH["SearchResult"](path, 0, 0, 0), grid, (0,0), (0,2), 2)


if __name__ == "__main__":
    unittest.main()
