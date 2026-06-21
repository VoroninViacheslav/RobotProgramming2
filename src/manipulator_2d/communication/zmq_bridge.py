import pickle

import zmq

from manipulator_2d.constants import DEFAULT_HOST, DEFAULT_PORT, ZMQ_TIMEOUT_MS


def _set_timeouts(socket):
    socket.setsockopt(zmq.RCVTIMEO, ZMQ_TIMEOUT_MS)
    socket.setsockopt(zmq.SNDTIMEO, ZMQ_TIMEOUT_MS)


def _recv_send(socket, reply, label):
    try:
        request = pickle.loads(socket.recv())
        socket.send(pickle.dumps(reply))
        return request
    except zmq.Again:
        print(f"[ZMQ {label}] Таймаут")
        return None
    except Exception as exc:
        print(f"[ZMQ {label}] Ошибка: {exc}")
        return None


def _send_recv(socket, request, label):
    try:
        socket.send(pickle.dumps(request))
        return pickle.loads(socket.recv())
    except zmq.Again:
        print(f"[ZMQ {label}] Таймаут")
        return None
    except Exception as exc:
        print(f"[ZMQ {label}] Ошибка: {exc}")
        return None


class ZMQServer:
    """REP-сокет: получает action, отправляет state."""

    def __init__(self, port=DEFAULT_PORT):
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REP)
        self.socket.bind(f"tcp://*:{port}")
        _set_timeouts(self.socket)
        print(f"[ZMQ Server] Запущен на порту {port}")

    def exchange(self, state):
        return _recv_send(self.socket, state, "Server")

    def close(self):
        self.socket.close()
        self.context.term()


class ZMQClient:
    """REQ-сокет: отправляет action, получает state."""

    def __init__(self, server_ip=DEFAULT_HOST, port=DEFAULT_PORT):
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REQ)
        self.socket.connect(f"tcp://{server_ip}:{port}")
        _set_timeouts(self.socket)
        print(f"[ZMQ Client] Подключен к {server_ip}:{port}")

    def exchange(self, action):
        return _send_recv(self.socket, action, "Client")

    def close(self):
        self.socket.close()
        self.context.term()
