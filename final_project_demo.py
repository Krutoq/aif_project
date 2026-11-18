"""Run the course Final Project demo: Bayes -> one-step MDP -> A* route.

This deterministic, self-contained command-line demonstration uses only the
Python standard library. The desktop application remains available via
`python app.py` after installing requirements and setting up its map data.
"""

import argparse

from ai.route_demo import (
    PRIOR_BLOCKED, SENSOR_SENSITIVITY, SENSOR_FALSE_POSITIVE_RATE,
    RISK_EDGE, run_route_demo,
)


def run_scenario(sensor_alert: bool) -> None:
    result = run_route_demo(sensor_alert)
    initial_route = result.initial_route
    update = result.bayes
    decision = result.mdp
    selected_route = result.selected_route

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
