import numpy as np

from manipulator_2d.constants import ACTION_DIM, OBS_DIM


def test_reset_returns_obs_with_correct_dim(simulator):
    np.random.seed(42)
    obs = simulator.reset()
    assert obs.shape == (OBS_DIM,)


def test_get_obs_shape(simulator):
    obs = simulator.get_obs()
    assert obs.shape == (OBS_DIM,)
    assert obs.dtype == np.float32


def test_step_returns_valid_obs(simulator):
    action = np.zeros(ACTION_DIM)
    obs, reward, done, info = simulator.step(action)
    assert obs.shape == (OBS_DIM,)
    assert isinstance(done, (bool, np.bool_))
    assert "distance" in info
    assert "grasped" in info
    assert isinstance(reward, (float, np.floating))


def test_build_state_keys(simulator):
    obs = simulator.get_obs()
    state = simulator.build_state(obs)
    assert set(state.keys()) == {"obs", "eef_pos", "obj_pos", "goal", "grasped"}
    assert state["obs"].shape == (OBS_DIM,)
    assert state["eef_pos"].shape == (2,)
    assert state["obj_pos"].shape == (2,)
    assert state["goal"].shape == (2,)
    assert isinstance(state["grasped"], bool)
