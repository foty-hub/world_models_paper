import numpy as np

from wm.imagination import (
    MAX_TEMPERATURE,
    MIN_TEMPERATURE,
    adjust_temperature,
    car_racing_action,
)


def test_car_racing_action_uses_gymnasium_order_and_bounds():
    action = car_racing_action(left=True, right=False, gas=True, brake=False)

    np.testing.assert_array_equal(action, [-1.0, 1.0, 0.0])
    assert action.dtype == np.float32


def test_opposing_steering_keys_cancel():
    action = car_racing_action(left=True, right=True, gas=False, brake=True)

    np.testing.assert_array_equal(action, [0.0, 0.0, 1.0])


def test_temperature_adjustment_is_bounded():
    assert adjust_temperature(MAX_TEMPERATURE, 0.5) == MAX_TEMPERATURE
    assert adjust_temperature(MIN_TEMPERATURE, -0.5) == MIN_TEMPERATURE
    assert adjust_temperature(1.0, 0.05) == 1.05
