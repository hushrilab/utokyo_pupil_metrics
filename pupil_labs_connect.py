import time
import zmq
import msgpack
from contextlib import contextmanager

PUPIL_ADDR = "127.0.0.1"
REQ_PORT = 50020  # Pupil Remote (REQ/REP)

class PupilCoreClient:
    def __init__(self, addr=PUPIL_ADDR, req_port=REQ_PORT):
        self.ctx = zmq.Context.instance()
        # REQ socket to query ports and issue synchronous commands
        self.req = self.ctx.socket(zmq.REQ)
        self.req.connect(f"tcp://{addr}:{req_port}")

        # discover PUB/SUB ports
        self.req.send_string("SUB_PORT")
        self.sub_port = self.req.recv_string()

        self.req.send_string("PUB_PORT")
        self.pub_port = self.req.recv_string()

        # SUB for data streams (e.g., "gaze.")
        self.sub = self.ctx.socket(zmq.SUB)
        self.sub.connect(f"tcp://{addr}:{self.sub_port}")

        # PUB for notifications/commands (msgpack-encoded)
        self.pub = self.ctx.socket(zmq.PUB)
        self.pub.connect(f"tcp://{addr}:{self.pub_port}")

        # give sockets a moment
        time.sleep(0.1)

    def subscribe(self, topic_prefix: str):
        """Subscribe to topics like 'gaze.' or 'pupil.' or 'notifications'."""
        self.sub.setsockopt_string(zmq.SUBSCRIBE, topic_prefix)

    def notify(self, subject: str, **kwargs):
        """
        Send a notify.* message as msgpack via PUB.
        Example: notify('recording.should_start')
        """
        payload = {
            "subject": subject,
            "timestamp": time.time(),
            **kwargs,
        }
        # Topic must be 'notify.' for notifications
        topic = "notify."
        self.pub.send_string(topic, flags=zmq.SNDMORE)
        self.pub.send(msgpack.dumps(payload, use_bin_type=True))

    def recv_gaze(self, timeout_ms=1000):
        """Receive one gaze message (topic, payload_dict)."""
        poller = zmq.Poller()
        poller.register(self.sub, zmq.POLLIN)
        socks = dict(poller.poll(timeout_ms))
        if socks.get(self.sub) == zmq.POLLIN:
            topic = self.sub.recv_string()
            payload = msgpack.loads(self.sub.recv(), raw=False)
            return topic, payload
        return None, None

    def close(self):
        self.sub.close(0)
        self.pub.close(0)
        self.req.close(0)
        # Do not terminate context.instance() globally—others might be using it.

@contextmanager
def pupil_core(addr=PUPIL_ADDR, req_port=REQ_PORT):
    client = PupilCoreClient(addr, req_port)
    try:
        yield client
    finally:
        client.close()

if __name__ == "__main__":
    # Example usage
    with pupil_core() as pc:
        # subscribe to gaze stream
        pc.subscribe("gaze.")  # other useful topics: 'pupil.', 'notify.'

        # OPTIONAL: start a recording (stored by Pupil Capture)
        # You can pass a custom session name:
        pc.notify("recording.should_start", session_name="demo_test")

        print("Collecting gaze for ~5 seconds...")
        t0 = time.time()
        while time.time() - t0 < 5:
            topic, payload = pc.recv_gaze(timeout_ms=1000)
            if payload:
                # payload keys typically include: 'gaze_point_3d', 'gaze_point_2d', 'confidence', 'timestamp'
                gp2d = payload.get("gaze_point_2d")
                conf = payload.get("confidence")
                ts = payload.get("timestamp")
                print(f"{topic} | ts={ts:.3f} | gaze2d={gp2d} | conf={conf}")

        # OPTIONAL: stop recording
        pc.notify("recording.should_stop")

        print("Done.")