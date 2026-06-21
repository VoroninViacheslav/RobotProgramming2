import numpy as np

from manipulator_2d.constants import ACTION_DIM
from manipulator_2d.controller.pd_controller import (
    PHASE_APPROACH,
    PHASE_TRANSPORT,
    PDController,
)


def test_compute_returns_correct_shape(full_obs):
    controller = PDController()
    action = controller.compute(full_obs)
    assert action.shape == (ACTION_DIM,)


def test_phase_approach_when_far_from_object(full_obs):
    controller = PDController()
    full_obs[18] = 0.0
    full_obs[12:14] = [0.0, 0.0]
    full_obs[14:16] = [1.0, 1.0]
    controller.compute(full_obs)
    assert controller.phase == PHASE_APPROACH


def test_phase_transport_when_grasped(full_obs):
    controller = PDController()
    full_obs[18] = 1.0
    full_obs[14:16] = [0.3, 0.1]
    full_obs[16:18] = [0.8, 0.5]
    controller.compute(full_obs)
    assert controller.phase == PHASE_TRANSPORT


def test_action_is_finite(full_obs):
    controller = PDController()
    action = controller.compute(full_obs)
    assert np.all(np.isfinite(action))
