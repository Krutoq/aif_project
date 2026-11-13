"""Run the course Final Project demo: Bayes -> one-step MDP -> A* route.

This deterministic, self-contained command-line demonstration uses only the
Python standard library. The desktop application remains available via
`python app.py` after installing requirements and setting up its map data.
"""

import argparse

from ai.road_risk import update_road_closure_probability
from ai.route_mdp import decide_route
from ai.routing import astar_route


class DemoRoadGraph:
    """Minimal graph interface consumed by the project's A* implementation."""

    def __init__(self):
        # Coordinates are close together so straight-line distance is an
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
            ("JCT", "CALL", 100.0),  # sensor-risk road segment
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


def run_scenario(sensor_alert: bool) -> None:
    graph = DemoRoadGraph()
    initial_route = astar_route(graph, "AMB", "CALL")
    if not initial_route.path:
        raise RuntimeError("The demo graph has no initial route to the call")

    update = update_road_closure_probability(
        prior_blocked=PRIOR_BLOCKED,
        sensor_alert=sensor_alert,
        sensitivity=SENSOR_SENSITIVITY,
        false_positive_rate=SENSOR_FALSE_POSITIVE_RATE,
    )
    decision = decide_route(update.posterior_blocked)

    if decision.action == "reroute":
        blocked_edges = {RISK_EDGE, (RISK_EDGE[1], RISK_EDGE[0])}
        selected_route = astar_route(
            graph, "AMB", "CALL", blocked_edges=blocked_edges
        )
    else:
        selected_route = initial_route

    if not selected_route.path:
        raise RuntimeError("The selected action left no route to the call")

    signal_name = "ALERT" if sensor_alert else "NO ALERT"
    print(f"\n{'=' * 64}\nSCENARIO: {signal_name}")
    print("INPUT")
    print(f"  Ambulance: AMB | Call: CALL | Sensor alert: {sensor_alert}")
    print(f"  Prior P(road blocked): {PRIOR_BLOCKED:.1%}")
    print("\n1) SEARCH: A*")
    print(f"  Initial shortest route: {' -> '.join(initial_route.path)}")
    print(f"  Initial route length: {initial_route.distance_m:.0f} m")
    print("\n2) PROBABILISTIC REASONING: BAYES")
    print(f"  P(alert | blocked): {SENSOR_SENSITIVITY:.1%}")
    print(f"  P(alert | clear): {SENSOR_FALSE_POSITIVE_RATE:.1%}")
    print(f"  P(observation): {update.evidence_probability:.3f}")
    print("  Bayes: P(blocked | observation) = "
          "P(observation | blocked) * P(blocked) / P(observation)")
    print(f"  P(blocked | observation): {update.posterior_blocked:.1%}")
    print("\n3) SEQUENTIAL DECISION: ONE-STEP MDP")
    print("  Transitions (next state Clear/Blocked):")
    print("    Continue from Clear:  0.90 / 0.10; from Blocked: 0.10 / 0.90")
    print("    Reroute  from either: 0.95 / 0.05")
    print("  Rewards: Continue Clear=+10, Blocked=-60; Reroute=-15")
    print("  Bellman example: Q(Clear,Continue)=10+0.90*(0.90*5+0.10*(-25))=11.80")
    print("                   Q(Blocked,Continue)=-60+0.90*(0.10*5+0.90*(-25))=-79.80")
    print("                   Q(Reroute)=-15+0.90*(0.95*5+0.05*(-25))=-11.85")
    print("  Belief-weighted action values (hypothetical reward units):")
    for action, value in decision.belief_action_values.items():
        print(f"    {action:8s}: {value:7.2f}")
    print(f"  Selected action: {decision.action.upper()}")
    print("\nOUTPUT")
    print(f"  Route used: {' -> '.join(selected_route.path)}")
    print(f"  Route length: {selected_route.distance_m:.0f} m")
    if decision.action == "reroute":
        print("  Explanation: MDP selected reroute; A* avoids the sensor-risk road.")
    else:
        print("  Explanation: MDP selected continue; A* keeps the shortest route.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Demonstrate A*, Bayesian road-risk inference, and a small MDP."
    )
    parser.add_argument(
        "--signal", choices=("both", "alert", "clear"), default="both",
        help="which sensor scenario to run (default: both)",
    )
    args = parser.parse_args()
    scenarios = {
        "alert": (True,),
        "clear": (False,),
        "both": (False, True),
    }[args.signal]
    for sensor_alert in scenarios:
        run_scenario(sensor_alert)


if __name__ == "__main__":
    main()
