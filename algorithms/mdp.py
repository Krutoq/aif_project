"""One-step expected utility policy for proceed versus detour."""

from dataclasses import dataclass


@dataclass
class Decision:
    blockage_probability: float
    proceed_utility: float
    detour_utility: float
    threshold: float
    action: str


def decide(blockage_probability: float, clear_reward: float = 100.0,
           blocked_reward: float = -100.0, detour_reward: float = 40.0) -> Decision:
    p = float(blockage_probability)
    if not 0 <= p <= 1:
        raise ValueError("blockage_probability must be between 0 and 1")
    if not all(map(lambda x: isinstance(x, (int, float)), (clear_reward, blocked_reward, detour_reward))):
        raise ValueError("rewards must be numeric")
    proceed = (1 - p) * clear_reward + p * blocked_reward
    denominator = clear_reward - blocked_reward
    threshold = (clear_reward - detour_reward) / denominator if denominator else (0.0 if clear_reward <= detour_reward else 1.0)
    # Equality is assigned to the safer detour action.
    action = "PROCEED" if proceed > detour_reward else "DETOUR"
    return Decision(p, proceed, detour_reward, threshold, action)


def sensitivity(clear_reward: float = 100.0, blocked_reward: float = -100.0,
                detour_reward: float = 40.0, points: int = 101) -> list[dict]:
    if points < 2:
        raise ValueError("points must be at least 2")
    return [{"p": i / (points - 1),
             "Proceed": (1 - i / (points - 1)) * clear_reward + (i / (points - 1)) * blocked_reward,
             "Detour": detour_reward} for i in range(points)]
