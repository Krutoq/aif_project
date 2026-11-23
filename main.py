"""Small offline console demonstration of the integrated pipeline."""

import argparse

from algorithms.astar import compare_searches
from config import BASELINE, DEFAULT_PRIOR, SMOKE_FPR, SMOKE_TPR, STRESS
from pipeline import run_pipeline


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sensor", choices=("alert", "clear"), default="alert")
    parser.add_argument("--prior", type=float, default=DEFAULT_PRIOR)
    parser.add_argument("--fpr", type=float, default=SMOKE_FPR)
    parser.add_argument("--tpr", type=float, default=SMOKE_TPR)
    args = parser.parse_args()
    result = run_pipeline(observation=args.sensor, prior=args.prior,
                          true_positive=args.tpr, false_positive=args.fpr)
    print("Mini Emergency Exit Planner (educational simulation)")
    print(f"Sensor: {args.sensor.upper()} | prior: {args.prior:.3f}")
    for index, sensor in enumerate(result.bayes, 1):
        print(f"Bayes update {index}: evidence={sensor.evidence:.4f}, posterior={sensor.posterior:.4%}")
    d = result.decision
    print(f"MDP: EU(Proceed)={d.proceed_utility:.1f}, EU(Detour)={d.detour_utility:.1f}, threshold={d.threshold:.1%} -> {d.action}")
    print(f"Route: {result.selected_exit}; cost={result.route.cost}; path={result.route.path}")
    print("A* vs BFS (baseline):")
    for row in compare_searches(BASELINE.size, BASELINE.start, BASELINE.exit_a, BASELINE.walls):
        print(f"  {row['Algorithm']}: path={row['Path length']}, expanded={row['Nodes expanded']}, runtime={row['Runtime (ms)']:.4f} ms")
    print("A* vs BFS (10 x 10 stress grid):")
    for row in compare_searches(STRESS.size, STRESS.start, STRESS.exit_a, STRESS.walls):
        print(f"  {row['Algorithm']}: path={row['Path length']}, expanded={row['Nodes expanded']}, runtime={row['Runtime (ms)']:.4f} ms")


if __name__ == "__main__":
    main()
