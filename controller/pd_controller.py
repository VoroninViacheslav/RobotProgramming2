import numpy as np
from simulator.kinematics import ForwardKinematics


class PDController:
    def __init__(self, kp=250.0, kd=25.0):
        self.kp = kp
        self.kd = kd
        self.fk = ForwardKinematics()
        self.phase = "approach"

    def compute(self, obs, dt=0.005):
        base_pos = obs[0:2]
        base_vel = obs[3:5]
        base_rot = obs[2]
        joint_angles = obs[6:9]
        joint_vel = obs[9:12]
        eef_pos = obs[12:14]
        obj_pos = obs[14:16]
        goal = obs[16:18]
        grasped = obs[18] > 0.5

        J = self.fk.jacobian_world(joint_angles, base_rot)
        eef_vel = J @ joint_vel

        dist_eef_obj = np.linalg.norm(eef_pos - obj_pos)
        dist_obj_goal = np.linalg.norm(obj_pos - goal)

        if not grasped:
            target = obj_pos
            if dist_eef_obj > 0.1:
                self.phase = "approach"
                gain_mult = 0.8
            else:
                self.phase = "grasp"
                gain_mult = 0.4
        else:
            target = goal
            if dist_obj_goal > 0.05:
                self.phase = "transport"
                gain_mult = 0.7
            else:
                self.phase = "done"
                gain_mult = 0.1

        error = target - eef_pos
        force = self.kp * gain_mult * error - self.kd * eef_vel * 0.6

        max_force = 700.0
        force_norm = np.linalg.norm(force)
        if force_norm > max_force:
            force = force / force_norm * max_force

        torques = J.T @ force
        torques -= 1.5 * joint_vel
        torques = np.clip(torques, -200.0, 200.0)

        arm_reach = 0.45
        dist_base_to_target = np.linalg.norm(base_pos - target)

        if dist_base_to_target > arm_reach * 0.7:
            base_target = target - np.array([0.2, 0.0])
            base_error = base_target - base_pos
            base_speed = np.clip(base_error * 0.5, -0.3, 0.3)
        else:
            base_speed = -base_vel * 0.3

        return np.concatenate([torques, base_speed])