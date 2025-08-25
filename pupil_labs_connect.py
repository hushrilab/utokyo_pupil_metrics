"""
Pupil Core live pupil-diameter plot + keyboard-controlled recording

Keys:
  r -> start recording
  s -> stop recording
  q / ESC -> quit

Requirements:
- Run Pupil Capture and ensure the device is recognized.
- Enable the 'Pupil Remote' plugin (default REQ port: 50020).
"""

import time
import threading
import collections
from datetime import datetime

import zmq
import msgpack
import matplotlib.pyplot as plt

# ---------- Config ----------
PUPIL_ADDR = "127.0.0.1"
REQ_PORT = 50020           # Pupil Remote REQ/REP
CONF_THRESH = 0.6          # filter low-confidence pupil samples
WINDOW_SECONDS = 20        # plot rolling window length
UPDATE_INTERVAL = 0.05     # seconds between plot updates (~20 Hz)

# ---------- Pupil Core Client ----------
class PupilCoreClient:
    def __init__(self, addr=PUPIL_ADDR, req_port=REQ_PORT):
        self.ctx = zmq.Context.instance()

        # REQ socket — for querying ports
        self.req = self.ctx.socket(zmq.REQ)
        self.req.connect(f"tcp://{addr}:{req_port}")

        # discover sub/pub ports
        self.req.send_string("SUB_PORT")
        self.sub_port = self.req.recv_string()

        self.req.send_string("PUB_PORT")
        self.pub_port = self.req.recv_string()

        # SUB — for data streams
        self.sub = self.ctx.socket(zmq.SUB)
        self.sub.connect(f"tcp://{addr}:{self.sub_port}")
        # Subscribe to pupil stream (diameter is in these messages)
        self.sub.setsockopt_string(zmq.SUBSCRIBE, "pupil.")

        # PUB — for notifications (e.g., recording control)
        self.pub = self.ctx.socket(zmq.PUB)
        self.pub.connect(f"tcp://{addr}:{self.pub_port}")

        # Give sockets a moment to connect
        time.sleep(0.1)

        # Receiver thread state
        self._stop_flag = threading.Event()
        self._thread = None

    def start_receiver(self, callback):
        """Start background thread; callback(topic:str, payload:dict)."""
        def _run():
            poller = zmq.Poller()
            poller.register(self.sub, zmq.POLLIN)
            while not self._stop_flag.is_set():
                socks = dict(poller.poll(100))  # 100 ms
                if socks.get(self.sub) == zmq.POLLIN:
                    topic = self.sub.recv_string()
                    payload = msgpack.loads(self.sub.recv(), raw=False)
                    callback(topic, payload)
        self._thread = threading.Thread(target=_run, daemon=True)
        self._thread.start()

    def stop_receiver(self):
        self._stop_flag.set()
        if self._thread:
            self._thread.join(timeout=1.0)

    def notify(self, subject: str, **kwargs):
        """Send Pupil notification via PUB."""
        payload = {"subject": subject, "timestamp": time.time(), **kwargs}
        self.pub.send_string("notify.", flags=zmq.SNDMORE)
        self.pub.send(msgpack.dumps(payload, use_bin_type=True))

    def close(self):
        self.stop_receiver()
        self.sub.close(0)
        self.pub.close(0)
        self.req.close(0)

# ---------- Live Plot / Main ----------
def main():
    pc = PupilCoreClient()

    # Deques for rolling window
    ts0 = None
    times = collections.deque(maxlen=10_000)
    diam = collections.deque(maxlen=10_000)

    recording = {"on": False, "label": ""}

    def handle_sample(topic, payload):
        nonlocal ts0
        # Typical keys include: timestamp, confidence, diameter, diameter_3d, id, method, etc.
        conf = payload.get("confidence", 0.0)
        d = payload.get("diameter", None)  # pixel or mm depending on method; often pixels for 2D
        t = payload.get("timestamp", None)
        if t is None or d is None or conf is None:
            return
        if conf < CONF_THRESH:
            return
        if ts0 is None:
            ts0 = t
        times.append(t - ts0)
        diam.append(d)

    pc.start_receiver(handle_sample)

    # Setup Matplotlib live plot
    plt.ion()
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ln, = ax.plot([], [], lw=1.5)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Pupil diameter")
    ax.set_title("Pupil diameter (live) — press 'r' to record, 's' to stop, 'q' to quit")

    # Status text (top-right)
    status_text = ax.text(
        0.98, 0.92, "IDLE",
        transform=ax.transAxes, ha="right", va="center", fontsize=11,
        bbox=dict(boxstyle="round,pad=0.2", fc="none", ec="gray")
    )

    # Keyboard handlers
    def on_key(event):
        key = (event.key or "").lower()
        if key in ("q", "escape"):
            plt.close(fig)
        elif key == "r":
            # Start a recording
            if not recording["on"]:
                label = f"pupil-diam-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
                pc.notify("recording.should_start", session_name=label)
                recording["on"] = True
                recording["label"] = label
        elif key == "s":
            # Stop a recording
            if recording["on"]:
                pc.notify("recording.should_stop")
                recording["on"] = False

    fig.canvas.mpl_connect("key_press_event", on_key)

    # Live update loop
    try:
        last_draw = 0.0
        while plt.fignum_exists(fig.number):
            now = time.time()
            # Throttle updates
            if now - last_draw >= UPDATE_INTERVAL:
                # Limit x to rolling window
                if times:
                    tmax = times[-1]
                    tmin = max(0.0, tmax - WINDOW_SECONDS)
                    # Find starting index in deque (linear scan is fine for this window length)
                    # For efficiency, convert to list slice once per draw:
                    t_list = list(times)
                    d_list = list(diam)
                    # keep only within window
                    start_idx = 0
                    for i in range(len(t_list) - 1, -1, -1):
                        if t_list[i] < tmin:
                            start_idx = i + 1
                            break
                    t_view = t_list[start_idx:]
                    d_view = d_list[start_idx:]

                    ln.set_data(t_view, d_view)
                    if d_view:
                        ax.set_xlim(max(0, tmin), max(1.0, tmax))
                        ymin = min(d_view)
                        ymax = max(d_view)
                        if ymin == ymax:
                            ymin -= 0.5
                            ymax += 0.5
                        ax.set_ylim(ymin, ymax)

                # Update status
                if recording["on"]:
                    status_text.set_text(f"REC ● {recording['label']}")
                    status_text.set_bbox(dict(boxstyle="round,pad=0.2", fc="#ffcccc", ec="#cc0000"))
                else:
                    status_text.set_text("IDLE")
                    status_text.set_bbox(dict(boxstyle="round,pad=0.2", fc="none", ec="gray"))

                fig.canvas.draw()
                fig.canvas.flush_events()
                last_draw = now

            time.sleep(0.005)
    finally:
        # Ensure recording is stopped on exit
        if recording["on"]:
            pc.notify("recording.should_stop")
        pc.close()

if __name__ == "__main__":
    main()