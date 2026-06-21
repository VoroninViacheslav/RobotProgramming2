import pickle

import zmq

from manipulator_2d.constants import DEFAULT_HOST, DEFAULT_PORT, ZMQ_TIMEOUT_MS


class ZMQServer:
    def __init__(self, port=DEFAULT_PORT):
        self.port = port
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REP)
        self.socket.bind(f"tcp://*:{port}")
        self._set_timeouts()
        print(f"[ZMQ Server] Запущен на порту {port}")

    def _set_timeouts(self):
        self.socket.setsockopt(zmq.RCVTIMEO, ZMQ_TIMEOUT_MS)
        self.socket.setsockopt(zmq.SNDTIMEO, ZMQ_TIMEOUT_MS)

    def wait_for_connection(self):
        """Ожидает первое сообщение от контроллера (handshake)."""
        try:
            msg = self.socket.recv()
            return pickle.loads(msg)
        except zmq.Again:
            print("[ZMQ Server] Таймаут ожидания подключения")
            return None
        except Exception as exc:
            print(f"[ZMQ Server] Ошибка подключения: {exc}")
            return None

    def exchange(self, state):
        """Отправить состояние и получить управление."""
        try:
            self.socket.send(pickle.dumps(state))
            msg = self.socket.recv()
            return pickle.loads(msg)
        except zmq.Again:
            print("[ZMQ Server] Таймаут")
            return None
        except Exception as exc:
            print(f"[ZMQ Server] Ошибка: {exc}")
            return None

    def close(self):
        self.socket.close()
        self.context.term()


class ZMQClient:
    def __init__(self, server_ip=DEFAULT_HOST, port=DEFAULT_PORT):
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REQ)
        self.socket.connect(f"tcp://{server_ip}:{port}")
        self.socket.setsockopt(zmq.RCVTIMEO, ZMQ_TIMEOUT_MS)
        self.socket.setsockopt(zmq.SNDTIMEO, ZMQ_TIMEOUT_MS)
        print(f"[ZMQ Client] Подключен к {server_ip}:{port}")

    def exchange(self, action):
        """Отправить управление и получить состояние."""
        try:
            self.socket.send(pickle.dumps(action))
            msg = self.socket.recv()
            return pickle.loads(msg)
        except zmq.Again:
            print("[ZMQ Client] Таймаут")
            return None
        except Exception as exc:
            print(f"[ZMQ Client] Ошибка: {exc}")
            return None

    def close(self):
        self.socket.close()
        self.context.term()
