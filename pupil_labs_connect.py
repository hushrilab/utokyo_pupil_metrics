#!/usr/bin/env python3
"""
Pupil Core realtime pupil-diameter plot with latched alert coloring.

Controls:
  r -> start recording
  t -> stop recording
  q / ESC -> quit

Metrics:
- avg_2s: mean diameter over [t-2, t]
- prev_avg_2s: mean diameter over [t-4, t-2]
- pct_change: 100 * (avg_2s - prev_avg_2s) / prev_avg_2s
- pct_change_avg_1s: avg of pct_change over the last 1s

Coloring (latched):
- Plot is BLUE until the first time pct_change > ALERT_THRESH
- From that detection time onward, the plot is RED (no flipping back)
"""

import time
import threading
import collections
import argparse
from statistics import mean

import zmq
import msgpack
import matplotlib.pyplot as plt

# ---------- Args ----------
parser = argparse.ArgumentParser(description="Realtime pupil diameter plot + latched alert coloring (Pupil Core)")
parser.add_argument("--addr", default="127.0.0.1")
parser.add_argument("--req-port", type=int, default=50020)
parser.add_argument("--field", choices=["diameter", "diameter_3d"], default="diameter")
parser.add_argument("--conf-thresh", type=float, default=0.6)
parser.add_argument("--window", type=float, default=20.0)
parser.add_argument("--session-name", type=str, default=None)
parser.add_argument("--metrics", dest="metrics", action="store_true", default=True)
parser.add_argument("--no-metrics", dest="metrics", action="store_false")
parser.add_argument("--metrics-print", action="store_true")
parser.add_argument("--alert-thresh", type=float, default=15.0, help="Percent change trigger for red (default 15.0)")
args = parser.parse_args()

# ---------- Config ----------
PUPIL_ADDR = args.addr
REQ_PORT = args.req_port
CONF_THRESH = args.conf_thresh
WINDOW_SECONDS = args.window
FIELD_NAME = args.field
UPDATE_INTERVAL = 0.05

AVG_WIN = 2.0
DELTA_AVG_WIN = 1.0
ALERT_THRESH = args.alert_thresh  # percent

# ---------- Pupil Core Client ----------
class PupilCoreClient:
    def __init__(self, addr=PUPIL_ADDR, req_port=REQ_PORT):
        self.ctx = zmq.Context.instance()
        self.req = self.ctx.socket(zmq.REQ)
        self.req.connect(f"tcp://{addr}:{req_port}")

        self.req.send_string("SUB_PORT"); self.sub_port = self.req.recv_string()
        self.req.send_string("PUB_PORT"); self.pub_port = self.req.recv_string()

        self.sub = self.ctx.socket(zmq.SUB)
        self.sub.connect(f"tcp://{addr}:{self.sub_port}")
        self.sub.setsockopt_string(zmq.SUBSCRIBE, "pupil.")

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
                socks = dict(poller.poll(50))
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
        self.sub.close(0); self.pub.close(0); self.req.close(0)

# ---------- Helpers ----------
def window_slice(times_list, start_t, end_t):
    i0, j0 = None, None
    for i in range(len(times_list)):
        if times_list[i] > start_t:
            i0 = i; break
    for k in range(len(times_list)-1, -1, -1):
        if times_list[k] <= end_t:
            j0 = k; break
    if i0 is None or j0 is None or i0 > j0:
        return None
    return i0, j0

def compute_avg(values, i0, j0):
    if i0 is None:
        return None
    seg = values[i0:j0+1]
    return mean(seg) if seg else None

# ---------- Live Plot + Metrics ----------
def main():
    pc = PupilCoreClient()

    ts0 = None
    live_t = collections.deque(maxlen=60_000)
    live_d = collections.deque(maxlen=60_000)
    delta_hist = collections.deque(maxlen=10_000)

    rec_active = False
    rec_label = None

    # Latching state
    alert_trigger_time = None  # time (relative seconds) when threshold was first exceeded

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

    plt.ion()
    fig, ax = plt.subplots(figsize=(10, 5))
    # Two overlayed lines: blue for before trigger, red for after trigger
    blue_line, = ax.plot([], [], lw=1.5, color="blue")
    red_line,  = ax.plot([], [], lw=1.5, color="red")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Pupil diameter" + (" (mm)" if FIELD_NAME == "diameter_3d" else " (px)"))

    status = ax.text(0.02, 0.95, "IDLE",
                     transform=ax.transAxes, ha="left", va="center", fontsize=11,
                     bbox=dict(boxstyle="round,pad=0.2", fc="none", ec="gray"))
    metrics_text = ax.text(0.98, 0.05, "",
                           transform=ax.transAxes, ha="right", va="bottom", fontsize=10,
                           bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))

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
        elif k == "t":
            if rec_active:
                pc.notify("recording.should_stop")
                rec_active = False
                print(f"[REC STOP] {rec_label}")

    fig.canvas.mpl_connect("key_press_event", on_key)

    last_draw = 0.0
    last_print = 0.0

    try:
        while plt.fignum_exists(fig.number):
            now = time.time()
            if now - last_draw >= UPDATE_INTERVAL:
                if live_t:
                    tmax = live_t[-1]
                    tmin = max(0.0, tmax - WINDOW_SECONDS)
                    tv, dv = list(live_t), list(live_d)
                    # Slice rolling window
                    start_idx = 0
                    for i in range(len(tv)-1, -1, -1):
                        if tv[i] < tmin:
                            start_idx = i+1
                            break
                    tvw, dvw = tv[start_idx:], dv[start_idx:]

                    # ----- Metrics / trigger detection -----
                    metrics_str = ""
                    if args.metrics and tv:
                        t_now = tv[-1]
                        cur_win = window_slice(tv, t_now-AVG_WIN, t_now)
                        prev_win = window_slice(tv, t_now-2*AVG_WIN, t_now-AVG_WIN)
                        avg2 = compute_avg(dv, *(cur_win or (None,None)))
                        prev2 = compute_avg(dv, *(prev_win or (None,None)))
                        pct, pct_avg_1s = None, None
                        if avg2 is not None and prev2 is not None and prev2 != 0:
                            pct = 100.0*(avg2-prev2)/prev2
                            delta_hist.append((t_now, pct))
                            # Smooth 1s
                            while delta_hist and delta_hist[0][0] < t_now-DELTA_AVG_WIN:
                                delta_hist.popleft()
                            if delta_hist:
                                pct_avg_1s = mean(val for _, val in delta_hist)
                            # Latch trigger if threshold exceeded and not latched yet
                            if alert_trigger_time is None and pct > ALERT_THRESH:
                                alert_trigger_time = t_now
                                print(f"[ALERT LATCHED] Δ%={pct:.2f}% at t={t_now:.2f}s (threshold {ALERT_THRESH}%)")
                        def fmt(x, d=2): return f"{x:.{d}f}" if x is not None else "NA"
                        metrics_str = (f"t={t_now:.2f}s\n"
                                       f"avg_2s={fmt(avg2)}  prev_2s={fmt(prev2)}\n"
                                       f"Δ%={fmt(pct)}%  avg Δ%(1s)={fmt(pct_avg_1s)}%")
                        metrics_text.set_text(metrics_str)
                    else:
                        metrics_text.set_text("")

                    # ----- Update lines with latched coloring -----
                    if alert_trigger_time is None:
                        # No trigger yet: all blue, red empty
                        blue_line.set_data(tvw, dvw)
                        red_line.set_data([], [])
                    else:
                        # Split window at trigger time
                        # Find first index in window where t >= trigger
                        split_idx = 0
                        while split_idx < len(tvw) and tvw[split_idx] < alert_trigger_time:
                            split_idx += 1
                        # Blue up to split_idx (exclusive), red from split_idx onward
                        blue_line.set_data(tvw[:split_idx], dvw[:split_idx])
                        red_line.set_data(tvw[split_idx:], dvw[split_idx:])

                    # Axes limits
                    if dvw:
                        ax.set_xlim(max(0, tmin), max(1.0, tmax))
                        ymin, ymax = min(dvw), max(dvw)
                        if ymin == ymax: ymin, ymax = ymin-0.5, ymax+0.5
                        ax.set_ylim(ymin, ymax)

                # Status
                status.set_text(f"REC ● {rec_label}" if rec_active else "IDLE")
                status.set_bbox(dict(boxstyle="round,pad=0.2",
                                     fc="#ffcccc" if rec_active else "none",
                                     ec="#cc0000" if rec_active else "gray"))
                fig.canvas.draw()
                fig.canvas.flush_events()
                last_draw = now

                if args.metrics and args.metrics_print and metrics_text.get_text() and (now - last_print >= 0.5):
                    print(metrics_text.get_text().replace("\n", " | "))
                    last_print = now

            plt.pause(0.001)
    finally:
        if rec_active:
            pc.notify("recording.should_stop")
        pc.close()

if __name__ == "__main__":
    main()
