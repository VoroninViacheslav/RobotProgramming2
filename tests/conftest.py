import numpy as np
import pytest

from manipulator_2d.simulator.env import Simulator


@pytest.fixture
def full_obs():
    return np.array(
        [
            0.0, 0.0, 0.0,       # base pos (x,y) + rot
            0.0, 0.0, 0.0,       # base vel
            0.5, 0.8, -0.5,      # joint angles
            0.0, 0.0, 0.0,       # joint vel
            0.4, 0.2,            # eef pos
            0.3, 0.1,            # obj pos
            0.6, 0.4,            # goal
            0.0,                 # grasped
        ],
        dtype=np.float32,
    )


@pytest.fixture
def simulator():
    sim = Simulator()
    yield sim
    sim.close()
