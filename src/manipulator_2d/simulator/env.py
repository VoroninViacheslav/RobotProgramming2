import logging
import numpy as np
import mujoco
from importlib.resources import files

from manipulator_2d.constants import (
    CTRL_LIMIT,
    GOAL_DISTANCE,
    GRASP_DISTANCE,
    JOINT_NAMES,
    LINK_PINCH_DISTANCE,
    MOTOR_GEARS,
    NUM_JOINTS,
)


logger = logging.getLogger(__name__)


def _default_model_path():
    return files("manipulator_2d.config") / "robot.xml"


class Simulator:
    def __init__(self, xml_path=None):
        if xml_path is None:
            xml_path = _default_model_path()

        self.model = mujoco.MjModel.from_xml_path(str(xml_path))
        self.data = mujoco.MjData(self.model)
        self.viewer = None

        self.eef_site_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SITE, "eef_site"
        )
        self.target_site_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SITE, "target_site"
        )
        self.object_site_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SITE, "object_site"
        )

        self.act_ids = {
            name: mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"{name}_motor"
            )
            for name in JOINT_NAMES
        }
        self.gears = np.array(MOTOR_GEARS)

        self.goal = np.array([0.6, 0.4])
        self.object_pos = np.array([0.4, 0.1])
        self._update_target_site()
        self.grasped = False
        self._reset_sim()

    def _update_target_site(self):
        self.model.site_pos[self.target_site_id][0] = self.goal[0]
        self.model.site_pos[self.target_site_id][1] = self.goal[1]

    def _reset_sim(self):
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[0:3] = 0.0
        self.data.qpos[3:6] = [0.5, 0.8, -0.5]
        self.data.qpos[6] = self.object_pos[0]
        self.data.qpos[7] = self.object_pos[1]
        self.data.qvel[:] = 0.0
        mujoco.mj_forward(self.model, self.data)
        self.grasped = False

    def reset(self):
        self.object_pos[0] = np.random.uniform(0.15, 0.65)
        self.object_pos[1] = np.random.uniform(-0.15, 0.45)

        for _ in range(30):
            angle = np.random.uniform(-np.pi / 2.5, np.pi / 2.5)
            radius = np.random.uniform(0.15, 0.5)
            self.goal[0] = abs(radius * np.cos(angle)) + 0.15
            self.goal[1] = radius * np.sin(angle) + 0.15
            if np.linalg.norm(self.goal - self.object_pos) > 0.25:
                break

        self.goal[0] = np.clip(self.goal[0], 0.15, 0.7)
        self.goal[1] = np.clip(self.goal[1], -0.15, 0.5)
        self._update_target_site()
        self._reset_sim()

        logger.info(
            "Новая задача: объект (%.2f, %.2f) -> цель (%.2f, %.2f)",
            self.object_pos[0],
            self.object_pos[1],
            self.goal[0],
            self.goal[1],
        )
        return self.get_obs()

    def get_obs(self):
        base_pos = self.data.qpos[0:3].copy()
        base_vel = self.data.qvel[0:3].copy()
        joint_angles = self.data.qpos[3:6].copy()
        joint_vel = self.data.qvel[3:6].copy()
        eef_pos = self.data.site(self.eef_site_id).xpos[:2].copy()
        obj_pos = self.data.site(self.object_site_id).xpos[:2].copy()
        obs = np.concatenate(
            [base_pos, base_vel, joint_angles, joint_vel, eef_pos, obj_pos, self.goal, [float(self.grasped)]]
        )
        return obs.astype(np.float32)

    def get_eef_pos(self):
        return self.data.site(self.eef_site_id).xpos[:2].copy()

    def get_object_pos(self):
        return self.data.site(self.object_site_id).xpos[:2].copy()

    def get_goal_pos(self):
        return self.goal.copy()

    def build_state(self, obs):
        return {
            "obs": obs,
            "eef_pos": self.get_eef_pos(),
            "obj_pos": self.get_object_pos(),
            "goal": self.get_goal_pos(),
            "grasped": self.grasped,
        }

    def _apply_controls(self, full_action):
        action = full_action[:NUM_JOINTS]
        base_speed = full_action[NUM_JOINTS : NUM_JOINTS + 2]

        for i, name in enumerate(JOINT_NAMES):
            ctrl_val = np.clip(action[i] / self.gears[i], -CTRL_LIMIT, CTRL_LIMIT)
            self.data.ctrl[self.act_ids[name]] = ctrl_val

        self.data.qvel[0] = np.clip(base_speed[0], -0.3, 0.3)
        self.data.qvel[1] = np.clip(base_speed[1], -0.3, 0.3)

    def _try_grasp(self):
        dist_to_obj = np.linalg.norm(self.get_eef_pos() - self.get_object_pos())
        if dist_to_obj < GRASP_DISTANCE and not self.grasped:
            self.grasped = True
            logger.info("Объект схвачен")

    def _check_pinch_release(self):
        if not self.grasped:
            return

        link2_pos = self.data.xpos[self.model.body("link2").id][:2]
        link3_pos = self.data.xpos[self.model.body("link3").id][:2]
        obj_pos = self.get_object_pos()

        if (
            np.linalg.norm(obj_pos - link2_pos) < LINK_PINCH_DISTANCE
            or np.linalg.norm(obj_pos - link3_pos) < LINK_PINCH_DISTANCE
        ):
            self.grasped = False
            logger.warning("Объект зажат между звеньями — отпускание")

    def _sync_grasped_object(self):
        if not self.grasped:
            return

        eef_pos = self.get_eef_pos()
        self.data.qpos[6] = eef_pos[0]
        self.data.qpos[7] = eef_pos[1]
        self.data.qvel[6] = 0.0
        self.data.qvel[7] = 0.0
        mujoco.mj_forward(self.model, self.data)

    def step(self, full_action):
        self._apply_controls(full_action)
        mujoco.mj_step(self.model, self.data)

        self._try_grasp()
        self._check_pinch_release()
        self._sync_grasped_object()

        dist_to_goal = np.linalg.norm(self.get_object_pos() - self.goal)
        done = dist_to_goal < GOAL_DISTANCE and self.grasped

        return self.get_obs(), -dist_to_goal, done, {
            "distance": dist_to_goal,
            "grasped": self.grasped,
            "dist_to_obj": np.linalg.norm(self.get_eef_pos() - self.get_object_pos()),
        }

    def render(self):
        try:
            import mujoco.viewer

            if self.viewer is None:
                self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
            else:
                self.viewer.sync()
        except ImportError:
            pass

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None
