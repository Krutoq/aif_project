"""Bayesian updates for binary blockage sensors."""

from dataclasses import dataclass
from typing import Iterable


@dataclass
class BayesResult:
    prior: float
    observation: str
    likelihood_if_blocked: float
    likelihood_if_clear: float
    evidence: float
    posterior: float


def _probability(name: str, value: float) -> float:
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be between 0 and 1")
    return float(value)


def update(prior: float, observation: str, true_positive: float, false_positive: float) -> BayesResult:
    prior = _probability("prior", prior)
    true_positive = _probability("true_positive", true_positive)
    false_positive = _probability("false_positive", false_positive)
    observation = observation.strip().lower()
    if observation not in {"alert", "clear"}:
        raise ValueError("observation must be 'alert' or 'clear'")
    likelihood_b = true_positive if observation == "alert" else 1.0 - true_positive
    likelihood_c = false_positive if observation == "alert" else 1.0 - false_positive
    evidence = likelihood_b * prior + likelihood_c * (1.0 - prior)
    posterior = prior if evidence == 0 else likelihood_b * prior / evidence
    return BayesResult(prior, observation, likelihood_b, likelihood_c, evidence, posterior)


def sequential_update(prior: float, observations: Iterable[tuple[str, float, float]]) -> tuple[float, list[BayesResult]]:
    results, current = [], prior
    for observation, true_positive, false_positive in observations:
        result = update(current, observation, true_positive, false_positive)
        results.append(result)
        current = result.posterior
    return current, results
