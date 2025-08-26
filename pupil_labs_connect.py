#!/usr/bin/env python3
"""
Realtime Pupil Core plot using PCPS/APCPS metrics.

Modes:
- Default (manual onset): press "o" to mark stimulus onset. Baseline = 1 s before onset.
- Continuous baseline (--continuous-baseline): baseline = rolling mean of previous 1 s.

Hotkeys:
- r : start recording (uses --session-name if provided)
- t : stop recording
- o : mark stimulus onset (when not using --continuous-baseline)
- q / Esc : quit
"""

import argparse, time, threading, collections, atexit
from typing import Optional

import zmq, msgpack, matplotlib.pyplot as plt
from metrics_pupil import PCPSCalculator

# ---------- CLI ----------
parser = argparse.ArgumentParser()
parser.add_argument("--addr", default="127.0.0.1")
parser.add_argument("--req-port", type=int, default=50020)
parser.add_argument("--field", choices=["diameter", "diameter_3d"], default="diameter")
parser.add_argument("--conf-thresh", type=float, default=0.8)
parser.add_argument("--window", type=float, default=60.0)
parser.add_argument("--session-name", type=str, default=None,
                    help="Session name to use when starting a Pupil recording.")
parser.add_argument("--alert-up", type=float, default=15.0)
parser.add_argument("--alert-down", type=float, default=None)
parser.add_argument("--continuous-baseline", action="store_true",
                    help="Use rolling 1 s baseline instead of manual onset")
parser.add_argument("--forward-apcps", action="store_true",
                    help="With continuous baseline, use next 2.5 s instead of last 2.5 s")
parser.add_argument("--auto-start", action="store_true",
                    help="Start Pupil recording immediately on connect.")
parser.add_argument("--auto-stop", action="store_true",
                    help="Stop Pupil recording on exit.")
args = parser.parse_args()

# ---------- Pupil Core Client ----------
class PupilCoreClient:
    def __init__(self, addr=args.addr, req_port=args.req_port):
        self.ctx = zmq.Context.instance()
        self.req = self.ctx.socket(zmq.REQ); self.req.connect(f"tcp://{addr}:{req_port}")

        # Ask for ports
        self.req.send_string("SUB_PORT"); self.sub_port = self.req.recv_string()
        self.req.send_string("PUB_PORT"); self.pub_port = self.req.recv_string()

        # SUB socket for data
        self.sub = self.ctx.socket(zmq.SUB); self.sub.connect(f"tcp://{addr}:{self.sub_port}")
        self.sub.setsockopt_string(zmq.SUBSCRIBE, "pupil.")

        # PUB socket for notifications/annotations
        self.pub = self.ctx.socket(zmq.PUB); self.pub.connect(f"tcp://{addr}:{self.pub_port}")

        time.sleep(0.1)
        self._stop_flag = threading.Event(); self._thread = None

    def start_receiver(self, callback):
        def _run():
            poller = zmq.Poller(); poller.register(self.sub, zmq.POLLIN)
            while not self._stop_flag.is_set():
                socks = dict(poller.poll(50))
                if socks.get(self.sub) == zmq.POLLIN:
                    topic = self.sub.recv_string()
                    payload = msgpack.loads(self.sub.recv(), raw=False)
                    callback(topic, payload)
        self._thread = threading.Thread(target=_run, daemon=True); self._thread.start()

    def stop_receiver(self):
        self._stop_flag.set()
        if self._thread: self._thread.join(timeout=1.0)

    def notify(self, subject: str, **kwargs):
        payload = {"subject": subject, "timestamp": time.time(), **kwargs}
        self.pub.send_string("notify.", flags=zmq.SNDMORE)
        self.pub.send(msgpack.dumps(payload, use_bin_type=True))

    # ---- Recording helpers ----
    def start_recording(self, session_name: Optional[str] = None):
        if session_name:
            self.notify("recording.should_start", session_name=session_name)
        else:
            self.notify("recording.should_start")
        print("[PUPIL] Recording START requested", f"(session='{session_name}')" if session_name else "")

    def stop_recording(self):
        self.notify("recording.should_stop")
        print("[PUPIL] Recording STOP requested")

    def close(self):
        self.stop_receiver()
        self.sub.close(0); self.pub.close(0); self.req.close(0)

# ---------- Main ----------
def main():
    ts0 = None
    t_buf = collections.deque(maxlen=60000)
    y_buf = collections.deque(maxlen=60000)

    calc = PCPSCalculator(
        baseline_win=1.0,
        apcps_win=2.5,
        alert_up=args.alert_up,
        alert_down=args.alert_down,
        continuous_baseline=args.continuous_baseline,
        #forward_apcps=args.forward_apcps
    )

    pc = PupilCoreClient()

    # Track recording state for UI label
    recording = False

    # Ensure optional auto-stop on exit
    def _cleanup():
        nonlocal recording
        if args.auto_stop and recording:
            try:
                pc.stop_recording()
            except Exception as e:
                print(f"[PUPIL] Auto-stop failed: {e}")
        try:
            pc.close()
        except Exception:
            pass
    atexit.register(_cleanup)

    def on_sample(topic, payload):
        nonlocal ts0
        conf = payload.get("confidence", 0.0)
        y = payload.get(args.field); ts = payload.get("timestamp")
        if ts is None or y is None or conf < args.conf_thresh: return
        if ts0 is None: ts0 = ts
        t_buf.append(ts - ts0); y_buf.append(y)

    pc.start_receiver(on_sample)

    # Optional auto-start
    if args.auto_start:
        pc.start_recording(args.session_name)
        recording = True

    plt.ion()
    fig, ax = plt.subplots(figsize=(10, 5))
    blue_line, = ax.plot([], [], lw=1.5, color="blue")
    red_line,  = ax.plot([], [], lw=1.5, color="red")

    # Metrics and REC status labels
    metrics_text = ax.text(
        0.98, 0.05, "", transform=ax.transAxes,
        ha="right", va="bottom", fontsize=10,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8)
    )
    rec_text = ax.text(
        0.02, 0.95, "", transform=ax.transAxes,
        ha="left", va="top", fontsize=12, color="red",
        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="gray", alpha=0.9)
    )

    def update_rec_label():
        rec_text.set_text("REC ●" if recording else "IDLE")

    def on_key(event):
        nonlocal recording
        k = (event.key or "").lower()
        if k in ("q","escape"):
            plt.close(fig)
        elif k == "o" and not args.continuous_baseline:
            if t_buf:
                t_now = t_buf[-1]
                calc.set_onset(t_now, list(t_buf), list(y_buf))
                # Optional: write an annotation into the Pupil timeline
                pc.notify("annotation", label="onset", duration=0.0)
                print(f"[ONSET] t={t_now:.2f}s, baseline={calc.baseline_value}")
        elif k == "r":
            pc.start_recording(args.session_name)
            recording = True
            update_rec_label()
        elif k == "t":
            pc.stop_recording()
            recording = False
            update_rec_label()

    fig.canvas.mpl_connect("key_press_event", on_key)

    last_draw = 0.0
    update_rec_label()

    while plt.fignum_exists(fig.number):
        now = time.time()
        if now - last_draw >= 0.05 and t_buf:
            tmax = t_buf[-1]; tmin = max(0.0, tmax - args.window)
            tv, yv = list(t_buf), list(y_buf)
            m = calc.update(tv, yv)

            def fmt(x,d=2): return f"{x:.{d}f}" if x is not None else "NA"
            metrics_text.set_text(
                f"t={fmt(m.t_now)}s\nbaseline={fmt(m.baseline,3)}\n"
                f"PCPS={fmt(m.pcps_now)}%  APCPS={fmt(m.apcps)}% ({m.apcps_window_covered:.1f}s)\n"
                f"state={m.state}"
            )

            # Slice window
            start_idx = next((i for i,t in enumerate(tv) if t>=tmin),0)
            tvw, yvw = tv[start_idx:], yv[start_idx:]
            blue_y, red_y = calc.mask_window_segments(tvw, yvw, tmin, tmax)
            blue_line.set_data(tvw, blue_y); red_line.set_data(tvw, red_y)

            if yvw:
                ax.set_xlim(max(0,tmin), max(1.0,tmax))
                ymin,ymax=min(yvw),max(yvw)
                if ymin==ymax: ymin,ymax=ymin-0.5,ymax+0.5
                ax.set_ylim(ymin,ymax)

            # draw
            fig.canvas.draw(); fig.canvas.flush_events(); last_draw = now
        plt.pause(0.001)

if __name__=="__main__":
    main()
