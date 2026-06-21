#!/usr/bin/env python3
import sys
import os
import time
import numpy as np
import pickle

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from simulator.env import Simulator
from communication.zmq_bridge import ZMQServer


def main():
    print("=" * 70)
    print("2D МАНИПУЛЯТОР - СИМУЛЯЦИЯ (ZMQ SERVER)")
    print("=" * 70)
    
    sim = Simulator()
    print("[✓] Среда создана")
    
    server = ZMQServer(port=5555)
    print("[✓] ZMQ сервер запущен")
    
    obs = sim.reset()
    sim.render()
    time.sleep(0.3)
    
    print("\n[▶] Ожидание подключения контроллера...")
    print("[▶] Запустите controller/run_control.py в другом терминале\n")
    
    step = 0
    success_count = 0
    
    try:
        while True:
            # Ждем первый запрос от контроллера
            print("[ZMQ] Ожидание первого соединения...")
            try:
                msg = server.socket.recv()
                action_data = pickle.loads(msg)
                print("[ZMQ] Контроллер подключен!")
            except Exception as e:
                print(f"[ZMQ] Ошибка: {e}")
                time.sleep(1.0)
                continue
            
            while True:
                # Получаем состояние
                state = {
                    'obs': obs,
                    'eef_pos': sim.get_eef_pos(),
                    'obj_pos': sim.get_object_pos(),
                    'goal': sim.get_goal_pos(),
                    'grasped': sim.grasped
                }
                
                # Отправляем состояние и ждем управление
                action_data = server.exchange(state)
                
                if action_data is None:
                    print("[ZMQ] Потеря связи. Переподключение...")
                    break
                
                # Применяем управление (full_action = [torques, base_speed])
                full_action = action_data['action']
                action = full_action[:3]
                base_speed = full_action[3:5]
                
                sim.data.qvel[0] = np.clip(base_speed[0], -0.3, 0.3)
                sim.data.qvel[1] = np.clip(base_speed[1], -0.3, 0.3)
                
                obs, reward, done, info = sim.step(action)
                step += 1
                
                if step % 20 == 0:
                    eef_pos = sim.get_eef_pos()
                    obj_pos = sim.get_object_pos()
                    goal = sim.get_goal_pos()
                    base_pos = obs[0:2]
                    status = "✓" if info['grasped'] else "✗"
                    dist = info['distance']
                    print(
                        f"Шаг {step:4d} | "
                        f"База: ({base_pos[0]:.2f},{base_pos[1]:.2f}) | "
                        f"EE: ({eef_pos[0]:.2f},{eef_pos[1]:.2f}) | "
                        f"Obj: ({obj_pos[0]:.2f},{obj_pos[1]:.2f}) | "
                        f"Цель: ({goal[0]:.2f},{goal[1]:.2f}) | {status} | {dist:.3f}"
                    )
                
                if done:
                    success_count += 1
                    print(f"\n{'=' * 50}")
                    print(f"[✓ УСПЕХ #{success_count}] Объект доставлен! Шаг {step}")
                    print(f"{'=' * 50}\n")
                    obs = sim.reset()
                    step = 0
                    time.sleep(0.3)
                    continue
                
                sim.render()
                time.sleep(0.001)
            
    except KeyboardInterrupt:
        print("\n[!] Остановлено пользователем")
    finally:
        server.close()
        sim.close()
        print(f"[■] Завершено. Успешных доставок: {success_count}")


if __name__ == "__main__":
    main()