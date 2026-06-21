"""Точки входа для запуска симулятора и контроллера."""

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
from manipulator_2d.simulator.env import Simulator


def _format_status(step, eef_pos, obj_pos, goal, grasped, distance, base_pos=None):
    status = "✓" if grasped else "✗"
    base_part = ""
    if base_pos is not None:
        base_part = f"База: ({base_pos[0]:.2f},{base_pos[1]:.2f}) | "
    return (
        f"Шаг {step:4d} | {base_part}"
        f"EE: ({eef_pos[0]:.2f},{eef_pos[1]:.2f}) | "
        f"Obj: ({obj_pos[0]:.2f},{obj_pos[1]:.2f}) | "
        f"Цель: ({goal[0]:.2f},{goal[1]:.2f}) | {status} | {distance:.3f}"
    )


def run_sim(port=DEFAULT_PORT):
    print("=" * 70)
    print("2D МАНИПУЛЯТОР - СИМУЛЯЦИЯ (ZMQ SERVER)")
    print("=" * 70)

    sim = Simulator()
    print("[✓] Среда создана")

    server = ZMQServer(port=port)
    print("[✓] ZMQ сервер запущен")

    obs = sim.reset()
    sim.render()
    time.sleep(RENDER_INIT_SLEEP)

    print("\n[▶] Ожидание подключения контроллера...")
    print("[▶] Запустите `poetry run control` в другом терминале\n")

    step = 0
    success_count = 0

    try:
        while True:
            action_data = server.exchange(sim.build_state(obs))
            if action_data is None:
                print("[ZMQ] Нет связи, повтор...")
                time.sleep(0.5)
                continue

            obs, _, done, info = sim.step(action_data["action"])
            step += 1

            if step % LOG_EVERY_N_STEPS == 0:
                print(
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
                print(f"\n{'=' * 50}")
                print(f"[✓ УСПЕХ #{success_count}] Объект доставлен! Шаг {step}")
                print(f"{'=' * 50}\n")
                obs = sim.reset()
                step = 0
                time.sleep(RENDER_INIT_SLEEP)
                continue

            sim.render()
            time.sleep(SIM_TIMESTEP_SLEEP)

    except KeyboardInterrupt:
        print("\n[!] Остановлено пользователем")
    finally:
        server.close()
        sim.close()
        print(f"[■] Завершено. Успешных доставок: {success_count}")


def run_control(host=DEFAULT_HOST, port=DEFAULT_PORT, kp=DEFAULT_KP, kd=DEFAULT_KD):
    print("=" * 70)
    print("2D МАНИПУЛЯТОР - КОНТРОЛЛЕР (ZMQ CLIENT)")
    print("=" * 70)

    controller = PDController(kp=kp, kd=kd)
    print("[✓] Контроллер создан")

    client = ZMQClient(server_ip=host, port=port)
    print("[✓] ZMQ клиент запущен")
    print("\n[▶] Ожидание данных от симуляции...\n")

    step = 0
    full_action = np.zeros(NUM_JOINTS + 2)

    try:
        while True:
            state = client.exchange({"action": full_action})
            if state is None:
                print("[!] Нет данных от симуляции, ждем...")
                time.sleep(0.5)
                continue

            step += 1
            full_action = controller.compute(state["obs"])

            if step % LOG_EVERY_N_STEPS == 0:
                print(
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
        print("\n[!] Остановлено пользователем")
    finally:
        client.close()
        print("[■] Завершено")
