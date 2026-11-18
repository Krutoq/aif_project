"""Reusable, deterministic A* + Bayes + MDP teaching scenario."""

from dataclasses import dataclass

from ai.road_risk import BayesianUpdate, update_road_closure_probability
from ai.route_mdp import MDPDecision, decide_route
from ai.routing import RouteResult, astar_route


class DemoRoadGraph:
    """Small road graph used to make the three course algorithms repeatable."""

    def __init__(self):
        # Coordinates are close together so straight-line distance stays an
        # admissible heuristic for the illustrative road lengths below.
        self.nodes = {
            "AMB": {"y": 52.000, "x": 0.000},
            "JCT": {"y": 52.000, "x": 0.001},
            "CALL": {"y": 52.000, "x": 0.002},
            "A": {"y": 52.001, "x": 0.000},
            "B": {"y": 52.001, "x": 0.001},
        }
        self._edges = {}
        self._adjacency = {node: set() for node in self.nodes}
        for u, v, length in (
            ("AMB", "JCT", 100.0),
            ("JCT", "CALL", 100.0),  # road segment with sensor alert
            ("AMB", "A", 120.0),
            ("A", "B", 150.0),
            ("B", "CALL", 150.0),
        ):
            self._edges[(u, v)] = {"length": length}
            self._edges[(v, u)] = {"length": length}
            self._adjacency[u].add(v)
            self._adjacency[v].add(u)

    def neighbors(self, node):
        return iter(self._adjacency[node])

    def get_edge_data(self, u, v):
        return self._edges.get((u, v))

    def number_of_nodes(self):
        return len(self.nodes)


PRIOR_BLOCKED = 0.20
SENSOR_SENSITIVITY = 0.80
SENSOR_FALSE_POSITIVE_RATE = 0.10
RISK_EDGE = ("JCT", "CALL")


@dataclass(frozen=True)
class RouteDemoResult:
    sensor_alert: bool
    initial_route: RouteResult
    bayes: BayesianUpdate
    mdp: MDPDecision
    selected_route: RouteResult


def run_route_demo(sensor_alert: bool) -> RouteDemoResult:
    """Run the three algorithms as one small end-to-end decision pipeline."""
    graph = DemoRoadGraph()
    initial_route = astar_route(graph, "AMB", "CALL")
    if not initial_route.path:
        raise RuntimeError("The demo graph has no initial route to the call")

    bayes = update_road_closure_probability(
        prior_blocked=PRIOR_BLOCKED,
        sensor_alert=bool(sensor_alert),
        sensitivity=SENSOR_SENSITIVITY,
        false_positive_rate=SENSOR_FALSE_POSITIVE_RATE,
    )
    mdp = decide_route(bayes.posterior_blocked)

    if mdp.action == "reroute":
        blocked_edges = {RISK_EDGE, (RISK_EDGE[1], RISK_EDGE[0])}
        selected_route = astar_route(
            graph, "AMB", "CALL", blocked_edges=blocked_edges
        )
    else:
        selected_route = initial_route

    if not selected_route.path:
        raise RuntimeError("The selected action left no route to the call")
    return RouteDemoResult(
        sensor_alert=bool(sensor_alert),
        initial_route=initial_route,
        bayes=bayes,
        mdp=mdp,
        selected_route=selected_route,
    )
