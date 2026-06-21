import zmq
import pickle


class ZMQServer:
    def __init__(self, port=5555):
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REP)
        self.socket.bind(f"tcp://*:{port}")
        self.socket.setsockopt(zmq.RCVTIMEO, 5000)
        self.socket.setsockopt(zmq.SNDTIMEO, 5000)
        print(f"[ZMQ Server] Запущен на порту {port}")

    def exchange(self, state):
        """Отправить состояние и получить управление"""
        try:
            self.socket.send(pickle.dumps(state))
            msg = self.socket.recv()
            return pickle.loads(msg)
        except zmq.Again:
            print("[ZMQ Server] Таймаут")
            return None
        except Exception as e:
            print(f"[ZMQ Server] Ошибка: {e}")
            return None

    def close(self):
        self.socket.close()
        self.context.term()


class ZMQClient:
    def __init__(self, server_ip="127.0.0.1", port=5555):
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REQ)
        self.socket.connect(f"tcp://{server_ip}:{port}")
        self.socket.setsockopt(zmq.RCVTIMEO, 5000)
        self.socket.setsockopt(zmq.SNDTIMEO, 5000)
        print(f"[ZMQ Client] Подключен к {server_ip}:{port}")

    def exchange(self, action):
        """Отправить управление и получить состояние"""
        try:
            self.socket.send(pickle.dumps(action))
            msg = self.socket.recv()
            return pickle.loads(msg)
        except zmq.Again:
            print("[ZMQ Client] Таймаут")
            return None
        except Exception as e:
            print(f"[ZMQ Client] Ошибка: {e}")
            return None

    def close(self):
        self.socket.close()
        self.context.term()