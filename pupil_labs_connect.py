#!/usr/bin/env python3
"""
Pupil Core realtime pupil-diameter plot (no export).

Controls in the plot window:
  r -> start recording
  t -> stop recording
  q / ESC -> quit

Requirements:
- Pupil Capture running, device detected
- Pupil Remote enabled (default REQ port: 50020)

Example:
  python pupil_core_plot.py --session-name mytest
"""
import time
import threading
import collections
import argparse

import zmq
import msgpack
import matplotlib.pyplot as plt

# ---------- Args ----------
parser = argparse.ArgumentParser(description="Realtime pupil diameter plot from Pupil Core")
parser.add_argument("--addr", default="127.0.0.1", help="Pupil Capture host (default: 127.0.0.1)")
parser.add_argument("--req-port", type=int, default=50020, help="Pupil Remote REQ port (default: 50020)")
parser.add_argument("--field", choices=["diameter", "diameter_3d"], default="diameter",
                    help="Which field to plot (default: diameter in px; use diameter_3d for mm if 3D pupil is enabled)")
parser.add_argument("--conf-thresh", type=float, default=0.6, help="Confidence threshold (default: 0.6)")
parser.add_argument("--window", type=float, default=60.0, help="Live plot rolling window in seconds (default: 20)")
parser.add_argument("--session-name", type=str, default=None, help="Custom session name for recordings")
args = parser.parse_args()

# ---------- Config ----------
PUPIL_ADDR = args.addr
REQ_PORT = args.req_port
CONF_THRESH = args.conf_thresh
WINDOW_SECONDS = args.window
FIELD_NAME = args.field

# ---------- Pupil Core Client ----------
class PupilCoreClient:
    def __init__(self, addr=PUPIL_ADDR, req_port=REQ_PORT):
        self.ctx = zmq.Context.instance()

        # REQ socket to query ports
        self.req = self.ctx.socket(zmq.REQ)
        self.req.connect(f"tcp://{addr}:{req_port}")

        # discover SUB/PUB ports
        self.req.send_string("SUB_PORT")
        self.sub_port = self.req.recv_string()

        self.req.send_string("PUB_PORT")
        self.pub_port = self.req.recv_string()

        # SUB for data
        self.sub = self.ctx.socket(zmq.SUB)
        self.sub.connect(f"tcp://{addr}:{self.sub_port}")
        self.sub.setsockopt_string(zmq.SUBSCRIBE, "pupil.")

        # PUB for notifications (recording control, etc.)
        self.pub = self.ctx.socket(zmq.PUB)
        self.pub.connect(f"tcp://{addr}:{self.pub_port}")

        time.sleep(0.1)

        self._stop_flag = threading.Event()
        self._thread = None

    def start_receiver(self, callback):
        def _run():
            poller = zmq.Poller()
            poller.register(self.sub, zmq.POLLIN)
            while not self._stop_flag.is_set():
                socks = dict(poller.poll(50))  # ms
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
        payload = {"subject": subject, "timestamp": time.time(), **kwargs}
        self.pub.send_string("notify.", flags=zmq.SNDMORE)
        self.pub.send(msgpack.dumps(payload, use_bin_type=True))

    def close(self):
        self.stop_receiver()
        self.sub.close(0)
        self.pub.close(0)
        self.req.close(0)

# ---------- Live Plot ----------
def main():
    pc = PupilCoreClient()

    # Live rolling buffers
    ts0 = None
    live_t = collections.deque(maxlen=12_000)
    live_d = collections.deque(maxlen=12_000)

    # Recording state
    rec_active = False
    rec_label = None

    def handle_sample(topic, payload):
        nonlocal ts0
        conf = payload.get("confidence", 0.0)
        d = payload.get(FIELD_NAME)
        dev_ts = payload.get("timestamp")
        if dev_ts is None or d is None or conf is None:
            return
        if conf < CONF_THRESH:
            return

        if ts0 is None:
            ts0 = dev_ts
        live_t.append(dev_ts - ts0)
        live_d.append(d)

    pc.start_receiver(handle_sample)

    # Matplotlib UI
    plt.ion()
    fig, ax = plt.subplots(figsize=(9, 4.5))
    line, = ax.plot([], [], lw=1.5)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Pupil diameter" + (" (mm)" if FIELD_NAME == "diameter_3d" else " (px)"))
    ax.set_title("Pupil diameter (live) — r: start, t: stop, q: quit")

    status = ax.text(
        0.98, 0.92, "IDLE",
        transform=ax.transAxes, ha="right", va="center", fontsize=11,
        bbox=dict(boxstyle="round,pad=0.2", fc="none", ec="gray")
    )

    def on_key(event):
        nonlocal rec_active, rec_label
        k = (event.key or "").lower()
        if k in ("q", "escape"):
            plt.close(fig)
        elif k == "r":
            if not rec_active:
                rec_active = True
                rec_label = args.session_name or time.strftime("%Y%m%d-%H%M%S")
                pc.notify("recording.should_start", session_name=f"pupil-diam-{rec_label}")
                print(f"[REC START] {rec_label}")
        elif k == "t":  # stop recording
            if rec_active:
                pc.notify("recording.should_stop")
                rec_active = False
                print(f"[REC STOP] {rec_label}")

    cid = fig.canvas.mpl_connect("key_press_event", on_key)

    try:
        last_draw = 0.0
        while plt.fignum_exists(fig.number):
            now = time.time()
            if now - last_draw >= 0.05:
                if live_t:
                    tmax = live_t[-1]
                    tmin = max(0.0, tmax - WINDOW_SECONDS)
                    t_list = list(live_t)
                    d_list = list(live_d)
                    # slice to rolling window
                    start_idx = 0
                    for i in range(len(t_list) - 1, -1, -1):
                        if t_list[i] < tmin:
                            start_idx = i + 1
                            break
                    tv = t_list[start_idx:]
                    dv = d_list[start_idx:]

                    line.set_data(tv, dv)
                    if dv:
                        ax.set_xlim(max(0, tmin), max(1.0, tmax))
                        ymin, ymax = min(dv), max(dv)
                        if ymin == ymax:
                            ymin -= 0.5
                            ymax += 0.5
                        ax.set_ylim(ymin, ymax)

                # status badge
                if rec_active:
                    status.set_text(f"REC ● {rec_label}")
                    status.set_bbox(dict(boxstyle="round,pad=0.2", fc="#ffcccc", ec="#cc0000"))
                else:
                    status.set_text("IDLE")
                    status.set_bbox(dict(boxstyle="round,pad=0.2", fc="none", ec="gray"))

                fig.canvas.draw()
                fig.canvas.flush_events()
                last_draw = now

            # keep GUI responsive
            plt.pause(0.001)
    finally:
        if rec_active:
            pc.notify("recording.should_stop")
        fig.canvas.mpl_disconnect(cid)
        pc.close()

if __name__ == "__main__":
    main()