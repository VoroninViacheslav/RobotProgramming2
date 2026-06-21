import numpy as np

from manipulator_2d.constants import (
    APPROACH_DISTANCE,
    ARM_REACH,
    ARM_REACH_FACTOR,
    BASE_OFFSET,
    BASE_SPEED_LIMIT,
    DEFAULT_KD,
    DEFAULT_KP,
    GOAL_DISTANCE,
    IDX_BASE_POS,
    IDX_BASE_ROT,
    IDX_BASE_VEL,
    IDX_EEF_POS,
    IDX_GOAL,
    IDX_GRASPED,
    IDX_JOINT_ANGLES,
    IDX_JOINT_VEL,
    IDX_OBJ_POS,
    MAX_FORCE,
    MAX_TORQUE,
)
from manipulator_2d.simulator.kinematics import ForwardKinematics

PHASE_APPROACH = "approach"
PHASE_GRASP = "grasp"
PHASE_TRANSPORT = "transport"
PHASE_DONE = "done"

GAIN_BY_PHASE = {
    PHASE_APPROACH: 0.8,
    PHASE_GRASP: 0.4,
    PHASE_TRANSPORT: 0.7,
    PHASE_DONE: 0.1,
}


class PDController:
    def __init__(self, kp=DEFAULT_KP, kd=DEFAULT_KD):
        self.kp = kp
        self.kd = kd
        self.fk = ForwardKinematics()
        self.phase = PHASE_APPROACH

    def _select_phase_and_target(self, eef_pos, obj_pos, goal, grasped):
        if not grasped:
            target = obj_pos
            phase = PHASE_APPROACH if np.linalg.norm(eef_pos - obj_pos) > APPROACH_DISTANCE else PHASE_GRASP
        else:
            target = goal
            phase = (
                PHASE_TRANSPORT
                if np.linalg.norm(obj_pos - goal) > GOAL_DISTANCE
                else PHASE_DONE
            )
        return phase, target, GAIN_BY_PHASE[phase]

    def _compute_torques(self, target, eef_pos, joint_angles, joint_vel, base_rot, gain_mult):
        jacobian = self.fk.jacobian_world(joint_angles, base_rot)
        eef_vel = jacobian @ joint_vel

        error = target - eef_pos
        force = self.kp * gain_mult * error - self.kd * eef_vel * 0.6

        force_norm = np.linalg.norm(force)
        if force_norm > MAX_FORCE:
            force = force / force_norm * MAX_FORCE

        torques = jacobian.T @ force
        torques -= 1.5 * joint_vel
        return np.clip(torques, -MAX_TORQUE, MAX_TORQUE)

    def _compute_base_speed(self, base_pos, base_vel, target):
        dist_base_to_target = np.linalg.norm(base_pos - target)

        if dist_base_to_target > ARM_REACH * ARM_REACH_FACTOR:
            base_target = target - np.array([BASE_OFFSET, 0.0])
            base_error = base_target - base_pos
            return np.clip(base_error * 0.5, -BASE_SPEED_LIMIT, BASE_SPEED_LIMIT)

        return np.clip(-base_vel * 0.3, -BASE_SPEED_LIMIT, BASE_SPEED_LIMIT)

    def compute(self, obs, dt=0.005):
        base_pos = obs[IDX_BASE_POS]
        base_vel = obs[IDX_BASE_VEL]
        base_rot = obs[IDX_BASE_ROT]
        joint_angles = obs[IDX_JOINT_ANGLES]
        joint_vel = obs[IDX_JOINT_VEL]
        eef_pos = obs[IDX_EEF_POS]
        obj_pos = obs[IDX_OBJ_POS]
        goal = obs[IDX_GOAL]
        grasped = obs[IDX_GRASPED] > 0.5

        self.phase, target, gain_mult = self._select_phase_and_target(
            eef_pos, obj_pos, goal, grasped
        )

        torques = self._compute_torques(
            target, eef_pos, joint_angles, joint_vel, base_rot, gain_mult
        )
        base_speed = self._compute_base_speed(base_pos, base_vel, target)

        return np.concatenate([torques, base_speed])
