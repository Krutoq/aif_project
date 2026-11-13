"""A small finite-horizon MDP for continue-versus-reroute decisions.

The model uses two road states and two actions, with illustrative transition
probabilities, immediate rewards, discounting, and a one-step Bellman update.
It is a teaching model, not an operational safety policy.
"""

from dataclasses import dataclass
import math


STATES = ("clear", "blocked")
ACTIONS = ("continue", "reroute")

# P(next state | current state, action)
TRANSITIONS = {
    "continue": {
        "clear": {"clear": 0.90, "blocked": 0.10},
        "blocked": {"clear": 0.10, "blocked": 0.90},
    },
    "reroute": {
        "clear": {"clear": 0.95, "blocked": 0.05},
        "blocked": {"clear": 0.95, "blocked": 0.05},
    },
}

# Immediate rewards (hypothetical units)
REWARDS = {
    "continue": {"clear": 10.0, "blocked": -60.0},
    "reroute": {"clear": -15.0, "blocked": -15.0},
}

# One-step lookahead values for the next state (hypothetical units)
NEXT_STATE_VALUES = {"clear": 5.0, "blocked": -25.0}
DISCOUNT = 0.90


@dataclass(frozen=True)
class MDPDecision:
    probability_blocked: float
    state_action_values: dict
    belief_action_values: dict
    action: str


def _validate_model() -> None:
    if not math.isfinite(DISCOUNT) or not 0.0 <= DISCOUNT <= 1.0:
        raise ValueError("DISCOUNT must be between 0 and 1")
    for action in ACTIONS:
        for state in STATES:
            probabilities = TRANSITIONS[action][state]
            if set(probabilities) != set(STATES):
                raise ValueError(f"Transition states are incomplete for {action}/{state}")
            if any(not math.isfinite(p) or p < 0.0 or p > 1.0
                   for p in probabilities.values()):
                raise ValueError(f"Invalid transition probability for {action}/{state}")
            if not math.isclose(sum(probabilities.values()), 1.0, abs_tol=1e-9):
                raise ValueError(f"Transition probabilities must sum to 1 for {action}/{state}")


def decide_route(probability_blocked: float) -> MDPDecision:
    """Perform one Bellman update, then average state values by the Bayes belief."""
    _validate_model()
    probability_blocked = float(probability_blocked)
    if not math.isfinite(probability_blocked) or not 0.0 <= probability_blocked <= 1.0:
        raise ValueError("probability_blocked must be between 0 and 1")

    q_values = {action: {} for action in ACTIONS}
    for action in ACTIONS:
        for state in STATES:
            expected_future = sum(
                TRANSITIONS[action][state][next_state]
                * NEXT_STATE_VALUES[next_state]
                for next_state in STATES
            )
            q_values[action][state] = (
                REWARDS[action][state] + DISCOUNT * expected_future
            )

    belief_values = {
        action: (
            (1.0 - probability_blocked) * q_values[action]["clear"]
            + probability_blocked * q_values[action]["blocked"]
        )
        for action in ACTIONS
    }
    selected = max(ACTIONS, key=lambda action: belief_values[action])
    return MDPDecision(
        probability_blocked=probability_blocked,
        state_action_values=q_values,
        belief_action_values=belief_values,
        action=selected,
    )
