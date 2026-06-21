import logging
import pickle

import zmq

from manipulator_2d.constants import DEFAULT_HOST, DEFAULT_PORT, ZMQ_TIMEOUT_MS

logger = logging.getLogger(__name__)


def _set_timeouts(socket):
    socket.setsockopt(zmq.RCVTIMEO, ZMQ_TIMEOUT_MS)
    socket.setsockopt(zmq.SNDTIMEO, ZMQ_TIMEOUT_MS)


def _recv_send(socket, reply, role):
    try:
        request = pickle.loads(socket.recv())
        socket.send(pickle.dumps(reply))
        return request
    except zmq.Again:
        logger.warning("Таймаут обмена (%s)", role)
        return None
    except Exception:
        logger.exception("Ошибка обмена (%s)", role)
        return None


def _send_recv(socket, request, role):
    try:
        socket.send(pickle.dumps(request))
        return pickle.loads(socket.recv())
    except zmq.Again:
        logger.warning("Таймаут обмена (%s)", role)
        return None
    except Exception:
        logger.exception("Ошибка обмена (%s)", role)
        return None


class ZMQServer:
    """REP-сокет: получает action, отправляет state."""

    def __init__(self, port=DEFAULT_PORT):
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REP)
        self.socket.bind(f"tcp://*:{port}")
        _set_timeouts(self.socket)
        logger.info("Сервер запущен на порту %s", port)

    def exchange(self, state):
        return _recv_send(self.socket, state, "server")

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
        logger.info("Клиент подключён к %s:%s", server_ip, port)

    def exchange(self, action):
        return _send_recv(self.socket, action, "client")

    def close(self):
        self.socket.close()
        self.context.term()
