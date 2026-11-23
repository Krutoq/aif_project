"""Closed loop: initial A*, Bayesian sensor update, MDP, optional reroute."""

from dataclasses import asdict, dataclass
from typing import Optional

from algorithms.astar import SearchResult, astar
from algorithms.bayes import BayesResult, sequential_update, update
from algorithms.mdp import Decision, decide
from config import (BASELINE, DEFAULT_PRIOR, HEAT_FPR, HEAT_TPR,
                    REWARD_DETOUR, REWARD_PROCEED_BLOCKED,
                    REWARD_PROCEED_CLEAR, SMOKE_FPR, SMOKE_TPR, Building)


@dataclass
class PipelineResult:
    initial_route: SearchResult
    bayes: list[BayesResult]
    posterior: float
    decision: Decision
    route: SearchResult
    selected_exit: str
    effective_walls: frozenset[tuple[int, int]]


def run_pipeline(building: Building = BASELINE, observation: str = "alert",
                 prior: float = DEFAULT_PRIOR, true_positive: float = SMOKE_TPR,
                 false_positive: float = SMOKE_FPR, risk_penalty: float = 100.0,
                 heat_observation: Optional[str] = None,
                 heat_true_positive: float = HEAT_TPR,
                 heat_false_positive: float = HEAT_FPR,
                 walls=None) -> PipelineResult:
    """Run the teaching pipeline on an assumed static grid."""
    wall_set = frozenset(building.walls if walls is None else walls)
    initial = astar(building.size, building.start, building.exit_a, wall_set)
    smoke = update(prior, observation, true_positive, false_positive)
    sensor_results = [smoke]
    posterior = smoke.posterior
    if heat_observation is not None:
        posterior, sensor_results = sequential_update(
            prior, [(observation, true_positive, false_positive),
                    (heat_observation, heat_true_positive, heat_false_positive)])
    decision = decide(posterior, REWARD_PROCEED_CLEAR, -abs(risk_penalty), REWARD_DETOUR)
    if decision.action == "DETOUR":
        effective_walls = wall_set | {building.monitored_corridor}
        selected_exit, route = "Exit B", astar(building.size, building.start, building.exit_b, effective_walls)
    else:
        effective_walls = wall_set
        selected_exit, route = "Exit A", initial
    return PipelineResult(initial, sensor_results, posterior, decision, route,
                          selected_exit, frozenset(effective_walls))


def result_summary(result: PipelineResult) -> dict:
    return {"initial_path": result.initial_route.path, "posterior": result.posterior,
            "decision": result.decision.action, "exit": result.selected_exit,
            "path": result.route.path, "path_cost": result.route.cost,
            "reachable": result.route.path is not None}
