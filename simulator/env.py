import numpy as np
import mujoco
from pathlib import Path


class Simulator:
    def __init__(self, xml_path=None):
        if xml_path is None:
            current_dir = Path(__file__).parent.parent
            xml_path = current_dir / "config" / "robot.xml"
        
        self.model = mujoco.MjModel.from_xml_path(str(xml_path))
        self.data = mujoco.MjData(self.model)
        self.viewer = None

        self.eef_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "eef_site")
        self.target_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "target_site")
        self.object_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "object_site")

        self.act_ids = {
            "joint1": mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "joint1_motor"),
            "joint2": mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "joint2_motor"),
            "joint3": mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "joint3_motor"),
        }
        self.gears = np.array([100.0, 80.0, 60.0])

        self.goal = np.array([0.6, 0.4])
        self.object_pos = np.array([0.4, 0.1])
        
        self.model.site_pos[self.target_site_id][0] = self.goal[0]
        self.model.site_pos[self.target_site_id][1] = self.goal[1]

        self.grasped = False
        self._reset_sim()

    def _reset_sim(self):
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[0] = 0.0
        self.data.qpos[1] = 0.0
        self.data.qpos[2] = 0.0
        self.data.qpos[3] = 0.5
        self.data.qpos[4] = 0.8
        self.data.qpos[5] = -0.5
        self.data.qpos[6] = self.object_pos[0]
        self.data.qpos[7] = self.object_pos[1]
        self.data.qvel[:] = 0.0
        mujoco.mj_forward(self.model, self.data)
        self.grasped = False

    def reset(self):
        self.object_pos[0] = np.random.uniform(0.15, 0.65)
        self.object_pos[1] = np.random.uniform(-0.15, 0.45)
        
        attempts = 0
        while attempts < 30:
            angle = np.random.uniform(-np.pi/2.5, np.pi/2.5)
            radius = np.random.uniform(0.15, 0.5)
            self.goal[0] = abs(radius * np.cos(angle)) + 0.15
            self.goal[1] = radius * np.sin(angle) + 0.15
            if np.linalg.norm(self.goal - self.object_pos) > 0.25:
                break
            attempts += 1
        
        self.goal[0] = np.clip(self.goal[0], 0.15, 0.7)
        self.goal[1] = np.clip(self.goal[1], -0.15, 0.5)
        
        self.model.site_pos[self.target_site_id][0] = self.goal[0]
        self.model.site_pos[self.target_site_id][1] = self.goal[1]
        
        self._reset_sim()
        print(f"\n[НОВАЯ ЗАДАЧА] Объект: ({self.object_pos[0]:.2f}, {self.object_pos[1]:.2f}) -> Цель: ({self.goal[0]:.2f}, {self.goal[1]:.2f})")
        return self.get_obs()

    def get_obs(self):
        base_pos = self.data.qpos[0:3].copy()
        base_vel = self.data.qvel[0:3].copy()
        joint_angles = self.data.qpos[3:6].copy()
        joint_vel = self.data.qvel[3:6].copy()
        eef_pos = self.data.site(self.eef_site_id).xpos[:2].copy()
        obj_pos = self.data.site(self.object_site_id).xpos[:2].copy()
        obs = np.concatenate([
            base_pos, base_vel, joint_angles, joint_vel,
            eef_pos, obj_pos, self.goal, [float(self.grasped)]
        ])
        return obs.astype(np.float32)

    def get_eef_pos(self):
        return self.data.site(self.eef_site_id).xpos[:2].copy()

    def get_object_pos(self):
        return self.data.site(self.object_site_id).xpos[:2].copy()

    def get_goal_pos(self):
        return self.goal.copy()

    def step(self, action):
        for i in range(3):
            ctrl_val = np.clip(action[i] / self.gears[i], -20.0, 20.0)
            self.data.ctrl[self.act_ids[["joint1", "joint2", "joint3"][i]]] = ctrl_val

        mujoco.mj_step(self.model, self.data)

        eef_pos = self.get_eef_pos()
        obj_pos = self.get_object_pos()
        dist_to_obj = np.linalg.norm(eef_pos - obj_pos)

        # Увеличиваем зону захвата с 0.12 до 0.18
        if dist_to_obj < 0.18 and not self.grasped:
            self.grasped = True
            print(f"[ЗАХВАТ] Объект схвачен!")

        # Проверка на зажатие между звеньями
        if self.grasped:
            # Получаем позиции звеньев
            link2_pos = self.data.xpos[self.model.body('link2').id][:2]
            link3_pos = self.data.xpos[self.model.body('link3').id][:2]
            obj_pos_current = self.get_object_pos()
            
            dist_to_link2 = np.linalg.norm(obj_pos_current - link2_pos)
            dist_to_link3 = np.linalg.norm(obj_pos_current - link3_pos)
            
            # Если объект зажат между звеньями - отпускаем
            if dist_to_link2 < 0.06 or dist_to_link3 < 0.06:
                self.grasped = False
                print("[ОТПУСКАНИЕ] Объект зажат между звеньями!")

        if self.grasped:
            eef_pos_current = self.get_eef_pos()
            self.data.qpos[6] = eef_pos_current[0]
            self.data.qpos[7] = eef_pos_current[1]
            self.data.qvel[6] = 0.0
            self.data.qvel[7] = 0.0
            mujoco.mj_forward(self.model, self.data)

        dist_to_goal = np.linalg.norm(self.get_object_pos() - self.goal)
        done = dist_to_goal < 0.05 and self.grasped

        return self.get_obs(), -dist_to_goal, done, {
            "distance": dist_to_goal, 
            "grasped": self.grasped,
            "dist_to_obj": np.linalg.norm(self.get_eef_pos() - self.get_object_pos())
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