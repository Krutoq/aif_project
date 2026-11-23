"""Shared, explicitly hypothetical inputs for the teaching simulation."""

from dataclasses import dataclass
from typing import FrozenSet, Tuple

Cell = Tuple[int, int]


@dataclass(frozen=True)
class Building:
    name: str
    size: int
    start: Cell
    exit_a: Cell
    exit_b: Cell
    walls: FrozenSet[Cell]
    monitored_corridor: Cell


BASELINE = Building(
    name="6 x 6 baseline",
    size=6,
    start=(2, 0),
    exit_a=(2, 5),
    exit_b=(5, 5),
    walls=frozenset({(1, 1), (3, 1), (4, 3)}),
    monitored_corridor=(2, 3),
)

STRESS = Building(
    name="10 x 10 benchmark",
    size=10,
    start=(0, 0),
    exit_a=(0, 9),
    exit_b=(9, 9),
    walls=frozenset(
        {(1, 2), (1, 3), (1, 4), (2, 4), (3, 1), (3, 2), (3, 3),
         (4, 6), (5, 6), (6, 2), (6, 3), (6, 4), (8, 7)}
    ),
    monitored_corridor=(0, 5),
)

DEFAULT_PRIOR = 0.25
SMOKE_TPR = 0.85
SMOKE_FPR = 0.15
HEAT_TPR = 0.80
HEAT_FPR = 0.20
REWARD_PROCEED_CLEAR = 100.0
REWARD_PROCEED_BLOCKED = -100.0
REWARD_DETOUR = 40.0
