import threading
import time

import numpy as np

from manipulator_2d.communication.zmq_bridge import ZMQClient, ZMQServer


def _find_free_port():
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("", 0))
        return sock.getsockname()[1]


def test_zmq_handshake_and_exchange():
    port = _find_free_port()
    results = {}

    def run_server():
        server = ZMQServer(port=port)
        try:
            results["handshake"] = server.wait_for_connection()
            results["reply"] = server.exchange(
                {
                    "obs": np.zeros(19, dtype=np.float32),
                    "eef_pos": np.zeros(2),
                    "obj_pos": np.zeros(2),
                    "goal": np.zeros(2),
                    "grasped": False,
                }
            )
        finally:
            server.close()

    thread = threading.Thread(target=run_server, daemon=True)
    thread.start()
    time.sleep(0.2)

    client = ZMQClient(port=port)
    try:
        action = {"action": np.array([1.0, 2.0, 3.0, 0.1, 0.2])}
        results["state"] = client.exchange(action)
    finally:
        client.close()

    thread.join(timeout=2)

    assert "action" in results["handshake"]
    assert results["handshake"]["action"].shape == (5,)
    assert results["state"] is not None
    assert results["state"]["obs"].shape == (19,)
