"""Bayesian update for a noisy road-closure alert.

This small model is intended for the Final Project demonstration. Probabilities
are illustrative inputs, not live traffic or emergency-service data.
"""

from dataclasses import dataclass
import math


def _probability(value: float, name: str) -> float:
    value = float(value)
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be a finite probability between 0 and 1")
    return value


@dataclass(frozen=True)
class BayesianUpdate:
    prior_blocked: float
    sensor_alert: bool
    likelihood_if_blocked: float
    likelihood_if_clear: float
    evidence_probability: float
    posterior_blocked: float


def update_road_closure_probability(
    prior_blocked: float,
    sensor_alert: bool,
    sensitivity: float,
    false_positive_rate: float,
) -> BayesianUpdate:
    """Return P(blocked | sensor observation) using Bayes' rule.

    `sensitivity` is P(alert | blocked); `false_positive_rate` is
    P(alert | clear). The observation is one binary sensor alert.
    """
    prior = _probability(prior_blocked, "prior_blocked")
    sensitivity = _probability(sensitivity, "sensitivity")
    false_positive_rate = _probability(
        false_positive_rate, "false_positive_rate"
    )

    if sensor_alert:
        likelihood_blocked = sensitivity
        likelihood_clear = false_positive_rate
    else:
        likelihood_blocked = 1.0 - sensitivity
        likelihood_clear = 1.0 - false_positive_rate

    evidence = (
        prior * likelihood_blocked
        + (1.0 - prior) * likelihood_clear
    )
    if evidence == 0.0:
        raise ValueError("The supplied observation has zero probability under this model")

    posterior = prior * likelihood_blocked / evidence
    return BayesianUpdate(
        prior_blocked=prior,
        sensor_alert=bool(sensor_alert),
        likelihood_if_blocked=likelihood_blocked,
        likelihood_if_clear=likelihood_clear,
        evidence_probability=evidence,
        posterior_blocked=posterior,
    )
