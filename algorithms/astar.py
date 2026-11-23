"""Optimal four-way grid search and a matching BFS benchmark."""

from __future__ import annotations

from dataclasses import dataclass
from heapq import heappop, heappush
from itertools import count
from time import perf_counter
from typing import Dict, FrozenSet, List, Optional, Tuple

Cell = Tuple[int, int]
MOVES = ((-1, 0), (0, 1), (1, 0), (0, -1))


@dataclass
class SearchResult:
    path: Optional[List[Cell]]
    cost: Optional[int]
    expanded: List[Cell]
    trace: List[dict]
    elapsed_ms: float

    @property
    def expanded_count(self) -> int:
        return len(self.expanded)


def _neighbors(cell: Cell, size: int, walls: FrozenSet[Cell]):
    r, c = cell
    for dr, dc in MOVES:
        nxt = (r + dr, c + dc)
        if 0 <= nxt[0] < size and 0 <= nxt[1] < size and nxt not in walls:
            yield nxt


def _reconstruct(parent: Dict[Cell, Optional[Cell]], goal: Cell) -> List[Cell]:
    path = []
    node: Optional[Cell] = goal
    while node is not None:
        path.append(node)
        node = parent[node]
    return list(reversed(path))


def _validate(size: int, start: Cell, goal: Cell, walls: FrozenSet[Cell]) -> None:
    if size < 1:
        raise ValueError("size must be positive")
    for name, cell in (("start", start), ("goal", goal), *[("wall", w) for w in walls]):
        if len(cell) != 2 or not all(isinstance(v, int) for v in cell):
            raise ValueError(f"{name} must be an integer (row, column) pair")
        if not (0 <= cell[0] < size and 0 <= cell[1] < size):
            raise ValueError(f"{name} {cell} is outside the {size} x {size} grid")


def astar(size: int, start: Cell, goal: Cell, walls=()) -> SearchResult:
    blocked = frozenset(walls)
    _validate(size, start, goal, blocked)
    if start in blocked or goal in blocked:
        return SearchResult(None, None, [], [], 0.0)
    started = perf_counter()
    heuristic = lambda p: abs(p[0] - goal[0]) + abs(p[1] - goal[1])
    serial = count()
    frontier = [(heuristic(start), 0, next(serial), start)]
    parent: Dict[Cell, Optional[Cell]] = {start: None}
    g_score = {start: 0}
    expanded, trace, closed = [], [], set()
    while frontier:
        f, g, _, cell = heappop(frontier)
        if cell in closed or g != g_score.get(cell):
            continue
        closed.add(cell)
        expanded.append(cell)
        trace.append({"cell": cell, "g": g, "h": heuristic(cell), "f": f})
        if cell == goal:
            elapsed = (perf_counter() - started) * 1000
            return SearchResult(_reconstruct(parent, goal), g, expanded, trace, elapsed)
        for nxt in _neighbors(cell, size, blocked):
            tentative = g + 1
            if tentative < g_score.get(nxt, float("inf")):
                parent[nxt] = cell
                g_score[nxt] = tentative
                heappush(frontier, (tentative + heuristic(nxt), tentative, next(serial), nxt))
    elapsed = (perf_counter() - started) * 1000
    return SearchResult(None, None, expanded, trace, elapsed)


def bfs(size: int, start: Cell, goal: Cell, walls=()) -> SearchResult:
    blocked = frozenset(walls)
    _validate(size, start, goal, blocked)
    if start in blocked or goal in blocked:
        return SearchResult(None, None, [], [], 0.0)
    started = perf_counter()
    queue, head = [start], 0
    parent: Dict[Cell, Optional[Cell]] = {start: None}
    expanded, trace = [], []
    while head < len(queue):
        cell = queue[head]
        head += 1
        expanded.append(cell)
        trace.append({"cell": cell, "g": len(_reconstruct(parent, cell)) - 1, "h": None, "f": None})
        if cell == goal:
            elapsed = (perf_counter() - started) * 1000
            return SearchResult(_reconstruct(parent, goal), len(_reconstruct(parent, goal)) - 1, expanded, trace, elapsed)
        for nxt in _neighbors(cell, size, blocked):
            if nxt not in parent:
                parent[nxt] = cell
                queue.append(nxt)
    elapsed = (perf_counter() - started) * 1000
    return SearchResult(None, None, expanded, trace, elapsed)


def compare_searches(size: int, start: Cell, goal: Cell, walls=()) -> list[dict]:
    rows = []
    for name, search in (("A*", astar), ("BFS", bfs)):
        result = search(size, start, goal, walls)
        rows.append({"Algorithm": name, "Path length": result.cost, "Nodes expanded": result.expanded_count,
                     "Runtime (ms)": result.elapsed_ms, "Reachable": result.path is not None})
    return rows
