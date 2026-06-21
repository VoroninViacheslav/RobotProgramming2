import numpy as np

from manipulator_2d.simulator.kinematics import ForwardKinematics


def test_jacobian_shape():
    fk = ForwardKinematics()
    jacobian = fk.jacobian_world(np.array([0.5, 0.8, -0.5]), base_rot=0.1)
    assert jacobian.shape == (2, 3)


def test_jacobian_finite_values():
    fk = ForwardKinematics()
    jacobian = fk.jacobian_world(np.zeros(3), base_rot=0.0)
    assert np.all(np.isfinite(jacobian))


def test_jacobian_depends_on_base_rotation():
    fk = ForwardKinematics()
    q = np.array([0.3, 0.5, -0.2])
    j0 = fk.jacobian_world(q, base_rot=0.0)
    j1 = fk.jacobian_world(q, base_rot=1.0)
    assert not np.allclose(j0, j1)
