"""Small, UI-independent helpers for the interactive imagination demo."""

from dataclasses import dataclass

import numpy as np

MIN_TEMPERATURE = 0.05
MAX_TEMPERATURE = 2.0
TEMPERATURE_STEP = 0.05


def car_racing_action(
    *, left: bool, right: bool, gas: bool, brake: bool
) -> np.ndarray:
    """Convert held keys to Gymnasium CarRacing's continuous action vector."""
    steering = float(right) - float(left)
    return np.asarray([steering, float(gas), float(brake)], dtype=np.float32)


def adjust_temperature(temperature: float, delta: float) -> float:
    """Adjust and clamp the MDN sampling temperature."""
    return float(np.clip(temperature + delta, MIN_TEMPERATURE, MAX_TEMPERATURE))


@dataclass
class RolloutSelection:
    """The dataset frame and random seed needed to reproduce a rollout."""

    episode: int
    rollout_seed: int
