#!/usr/bin/env python3
import sys
import os
import time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from controller.pd_controller import PDController
from communication.zmq_bridge import ZMQClient


def main():
    print("=" * 70)
    print("2D МАНИПУЛЯТОР - КОНТРОЛЛЕР (ZMQ CLIENT)")
    print("=" * 70)
    
    controller = PDController(kp=250.0, kd=25.0)
    print("[✓] Контроллер создан")
    
    client = ZMQClient(server_ip="127.0.0.1", port=5555)
    print("[✓] ZMQ клиент запущен")
    
    print("\n[▶] Ожидание данных от симуляции...\n")
    
    step = 0
    full_action = np.zeros(5)
    
    try:
        while True:
            # Отправляем управление и получаем состояние
            state = client.exchange({'action': full_action})
            
            if state is None:
                print("[!] Нет данных от симуляции, ждем...")
                time.sleep(0.5)
                continue
            
            obs = state['obs']
            step += 1
            
            # Вычисляем управление
            full_action = controller.compute(obs)
            
            if step % 20 == 0:
                eef_pos = state['eef_pos']
                obj_pos = state['obj_pos']
                goal = state['goal']
                status = "✓" if state['grasped'] else "✗"
                dist = np.linalg.norm(obj_pos - goal)
                print(
                    f"Шаг {step:4d} | "
                    f"EE: ({eef_pos[0]:.2f},{eef_pos[1]:.2f}) | "
                    f"Obj: ({obj_pos[0]:.2f},{obj_pos[1]:.2f}) | "
                    f"Цель: ({goal[0]:.2f},{goal[1]:.2f}) | {status} | {dist:.3f}"
                )
            
            time.sleep(0.001)
            
    except KeyboardInterrupt:
        print("\n[!] Остановлено пользователем")
    finally:
        client.close()
        print("[■] Завершено")


if __name__ == "__main__":
    main()