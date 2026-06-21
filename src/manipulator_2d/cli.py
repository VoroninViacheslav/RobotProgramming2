"""Точки входа для запуска симулятора и контроллера."""

import logging
import time

import numpy as np

from manipulator_2d.communication.zmq_bridge import ZMQClient, ZMQServer
from manipulator_2d.constants import (
    DEFAULT_HOST,
    DEFAULT_KD,
    DEFAULT_KP,
    DEFAULT_PORT,
    LOG_EVERY_N_STEPS,
    NUM_JOINTS,
    RENDER_INIT_SLEEP,
    SIM_TIMESTEP_SLEEP,
)
from manipulator_2d.controller.pd_controller import PDController
from manipulator_2d.logging_config import setup_logging
from manipulator_2d.simulator.env import Simulator

logger = logging.getLogger(__name__)


def _format_status(step, eef_pos, obj_pos, goal, grasped, distance, base_pos=None):
    status = "grasped" if grasped else "open"
    base_part = ""
    if base_pos is not None:
        base_part = f"base=({base_pos[0]:.2f},{base_pos[1]:.2f}) "
    return (
        f"step={step} {base_part}"
        f"ee=({eef_pos[0]:.2f},{eef_pos[1]:.2f}) "
        f"obj=({obj_pos[0]:.2f},{obj_pos[1]:.2f}) "
        f"goal=({goal[0]:.2f},{goal[1]:.2f}) "
        f"grip={status} dist={distance:.3f}"
    )


def run_sim(port=DEFAULT_PORT):
    setup_logging()
    logger.info("2D манипулятор — симуляция (ZMQ server, port=%s)", port)

    sim = Simulator()
    logger.info("MuJoCo-среда создана")

    server = ZMQServer(port=port)

    obs = sim.reset()
    sim.render()
    time.sleep(RENDER_INIT_SLEEP)

    logger.info("Ожидание контроллера — запустите `poetry run control` в другом терминале")

    step = 0
    success_count = 0

    try:
        while True:
            action_data = server.exchange(sim.build_state(obs))
            if action_data is None:
                logger.warning("Нет связи с контроллером, повтор через 0.5 с")
                time.sleep(0.5)
                continue

            obs, _, done, info = sim.step(action_data["action"])
            step += 1

            if step % LOG_EVERY_N_STEPS == 0:
                logger.info(
                    _format_status(
                        step,
                        sim.get_eef_pos(),
                        sim.get_object_pos(),
                        sim.get_goal_pos(),
                        info["grasped"],
                        info["distance"],
                        base_pos=obs[0:2],
                    )
                )

            if done:
                success_count += 1
                logger.info(
                    "Успех #%d: объект доставлен на шаге %d",
                    success_count,
                    step,
                )
                obs = sim.reset()
                step = 0
                time.sleep(RENDER_INIT_SLEEP)
                continue

            sim.render()
            time.sleep(SIM_TIMESTEP_SLEEP)

    except KeyboardInterrupt:
        logger.info("Остановлено пользователем")
    finally:
        server.close()
        sim.close()
        logger.info("Симуляция завершена, успешных доставок: %d", success_count)


def run_control(host=DEFAULT_HOST, port=DEFAULT_PORT, kp=DEFAULT_KP, kd=DEFAULT_KD):
    setup_logging()
    logger.info(
        "2D манипулятор — контроллер (ZMQ client, %s:%s, kp=%s, kd=%s)",
        host,
        port,
        kp,
        kd,
    )

    controller = PDController(kp=kp, kd=kd)
    client = ZMQClient(server_ip=host, port=port)
    logger.info("Ожидание данных от симуляции")

    step = 0
    full_action = np.zeros(NUM_JOINTS + 2)

    try:
        while True:
            state = client.exchange({"action": full_action})
            if state is None:
                logger.warning("Нет данных от симуляции, повтор через 0.5 с")
                time.sleep(0.5)
                continue

            step += 1
            full_action = controller.compute(state["obs"])

            if step % LOG_EVERY_N_STEPS == 0:
                logger.info(
                    _format_status(
                        step,
                        state["eef_pos"],
                        state["obj_pos"],
                        state["goal"],
                        state["grasped"],
                        np.linalg.norm(state["obj_pos"] - state["goal"]),
                    )
                )

            time.sleep(SIM_TIMESTEP_SLEEP)

    except KeyboardInterrupt:
        logger.info("Остановлено пользователем")
    finally:
        client.close()
        logger.info("Контроллер завершён")
