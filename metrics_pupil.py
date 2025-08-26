#!/usr/bin/env python3
"""
PCPS/APCPS metrics with:
  - confidence-based exclusion
  - low-pass smoothing (first-order IIR, supports irregular sampling)
  - realtime (incremental) sliding APCPS in update()/step_update()
  - offline, vectorized series computation in apcps_series(..., mode=["sliding","forward"])
  - OPTIONAL fixed reference baseline (reference_baseline) used instead of rolling 1 s

API
---
PCPSCalculator(
    baseline_win=1.0,
    apcps_win=2.5,
    alert_up=15.0,
    alert_down=None,          # defaults to alert_up
    continuous_baseline=False,
    conf_thresh=None,         # drop samples with confidence < conf_thresh (if confidences supplied)
    lp_cutoff_hz=None,        # low-pass cutoff [Hz] before computing metrics
    reference_baseline=None   # FIXED baseline value; if set (non-None, non-zero) overrides rolling baseline
)

Notes
-----
- If reference_baseline is provided (and != 0), all PCPS/APCPS use that constant for normalization.
- PCPS(t) = 100 * (y(t) - baseline) / baseline
- update()/step_update(): sliding APCPS (trailing window). Forward APCPS is for offline series only.
"""

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple
import math
import numpy as np
import collections


# ---------------------- helpers ----------------------
def window_slice(times: Sequence[float], start_t: float, end_t: float) -> Optional[Tuple[int, int]]:
    """Return [i0, j0] with start_t < t <= end_t. None if no sample."""
    if len(times) == 0:
        return None
    i0 = next((i for i, t in enumerate(times) if t > start_t), None)
    j0 = next((i for i in range(len(times) - 1, -1, -1) if times[i] <= end_t), None)
    if i0 is None or j0 is None or i0 > j0:
        return None
    return i0, j0


def mean_span(values: Sequence[float], span: Optional[Tuple[int, int]]) -> Optional[float]:
    if span is None:
        return None
    i0, j0 = span
    if i0 > j0:
        return None
    seg = values[i0 : j0 + 1]
    return float(np.mean(seg)) if len(seg) > 0 else None


def _as_np(a) -> np.ndarray:
    return np.asarray(a, dtype=float)


def _prep_mask(times: np.ndarray, values: np.ndarray,
               confidences: Optional[np.ndarray],
               conf_thresh: Optional[float]) -> np.ndarray:
    """Finite & (optional) confidence filter."""
    valid = np.isfinite(times) & np.isfinite(values)
    if confidences is not None:
        valid &= np.isfinite(confidences)
        if conf_thresh is not None:
            valid &= confidences >= conf_thresh
    return valid


def _lowpass_exp(times: np.ndarray, values: np.ndarray, cutoff_hz: Optional[float]) -> np.ndarray:
    """First-order IIR LPF for irregular sampling. Returns filtered copy."""
    if cutoff_hz is None or cutoff_hz <= 0.0 or len(values) == 0:
        return values.copy()
    y = np.empty_like(values, dtype=float)
    y[0] = values[0]
    tau = 1.0 / (2.0 * math.pi * cutoff_hz)
    for i in range(1, len(values)):
        dt = max(times[i] - times[i - 1], 0.0)
        if not np.isfinite(dt) or dt <= 0.0:
            y[i] = y[i - 1]
            continue
        alpha = 1.0 - math.exp(-dt / tau)
        y[i] = y[i - 1] + alpha * (values[i] - y[i - 1])
    return y


def _rolling_mean_prev(t: np.ndarray, y: np.ndarray, win: float) -> np.ndarray:
    """Mean over (t[i]-win, t[i]] per i. O(n) two-pointer."""
    n = len(t)
    out = np.full(n, np.nan, float)
    j, acc = 0, 0.0
    for i in range(n):
        acc += y[i]
        while j <= i and t[j] <= t[i] - win:
            acc -= y[j]
            j += 1
        cnt = i - j + 1
        if cnt > 0:
            out[i] = acc / cnt
    return out


def _forward_mean_next(t: np.ndarray, y: np.ndarray, win: float) -> np.ndarray:
    """Mean over [t[i], t[i]+win] per i. O(n) two-pointer."""
    n = len(t)
    out = np.full(n, np.nan, float)
    j, acc = 0, 0.0
    for i in range(n):
        if j < i:
            j = i
            acc = 0.0
        while j < n and t[j] <= t[i] + win:
            acc += y[j]
            j += 1
        cnt = j - i
        if cnt > 0:
            out[i] = acc / cnt
        acc -= y[i]
    return out


# ---------------------- data structs ----------------------
@dataclass
class Metrics:
    t_now: Optional[float] = None
    baseline: Optional[float] = None
    pcps_now: Optional[float] = None
    apcps: Optional[float] = None
    apcps_window_covered: float = 0.0
    state: str = "blue"  # 'blue'|'red'


# ---------------------- main class ----------------------
class PCPSCalculator:
    def __init__(
        self,
        baseline_win: float = 1.0,
        apcps_win: float = 2.5,
        alert_up: float = 15.0,
        alert_down: Optional[float] = None,
        continuous_baseline: bool = False,
        conf_thresh: Optional[float] = None,
        lp_cutoff_hz: Optional[float] = None,
        reference_baseline: Optional[float] = None,
    ):
        self.baseline_win = float(baseline_win)
        self.apcps_win = float(apcps_win)
        self.alert_up = float(alert_up)
        self.alert_down = float(alert_down) if alert_down is not None else float(alert_up)
        self.continuous_baseline = bool(continuous_baseline)

        self.conf_thresh = conf_thresh
        self.lp_cutoff_hz = lp_cutoff_hz
        self.reference_baseline = None if reference_baseline is None else float(reference_baseline)

        self.switches: List[Tuple[float, str]] = [(0.0, "blue")]
        self.current_state: str = "blue"

        # Manual onset (ignored if reference_baseline is set)
        self.onset_t: Optional[float] = None
        self.baseline_value: Optional[float] = None

        # Incremental buffers for realtime use
        self._times = collections.deque()
        self._values = collections.deque()
        self._confs = collections.deque()

    # -------- configuration helpers --------
    def set_confidence_threshold(self, thresh: Optional[float]):
        self.conf_thresh = thresh

    def set_lowpass_cutoff(self, cutoff_hz: Optional[float]):
        self.lp_cutoff_hz = cutoff_hz

    def set_reference_baseline(self, ref: Optional[float]):
        self.reference_baseline = None if ref is None else float(ref)

    # -------- preprocessing (filter + smooth) --------
    def _preprocess(
        self,
        times: Sequence[float],
        values: Sequence[float],
        confidences: Optional[Sequence[float]] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        t = _as_np(times)
        y = _as_np(values)
        c = _as_np(confidences) if confidences is not None else None

        mask = _prep_mask(t, y, c, self.conf_thresh)
        if not mask.any():
            return np.array([], dtype=float), np.array([], dtype=float)

        t2 = t[mask]
        y2 = y[mask]

        y_smooth = _lowpass_exp(t2, y2, self.lp_cutoff_hz) if self.lp_cutoff_hz else y2.copy()
        return t2, y_smooth

    # -------- manual onset (ignored when reference_baseline set) --------
    def set_onset(
        self,
        t_now: float,
        times: Sequence[float],
        values: Sequence[float],
        confidences: Optional[Sequence[float]] = None,
    ) -> None:
        if self.reference_baseline is not None:
            # fixed reference makes onset irrelevant
            return
        if self.continuous_baseline:
            return
        self.onset_t = t_now
        t_proc, y_proc = self._preprocess(times, values, confidences)
        if len(t_proc) == 0:
            self.baseline_value = None
            return
        span = window_slice(t_proc, t_now - self.baseline_win, t_now)
        self.baseline_value = mean_span(y_proc, span)

    # -------- incremental (sliding APCPS only) --------
    def step_update(self, t: float, y: float, conf: Optional[float] = None) -> Metrics:
        self._times.append(t)
        self._values.append(y)
        self._confs.append(conf)

        # prune old samples to keep computation O(1)
        window_span = max(self.baseline_win, self.apcps_win) + 0.5
        while self._times and (t - self._times[0] > window_span):
            self._times.popleft(); self._values.popleft(); self._confs.popleft()

        return self.update()  # will use internal buffers

    def update(
        self,
        times: Optional[Sequence[float]] = None,
        values: Optional[Sequence[float]] = None,
        confidences: Optional[Sequence[float]] = None,
    ) -> Metrics:
        """Compute instantaneous PCPS and sliding APCPS (trailing window). If reference_baseline is set,
        use that constant instead of rolling/manual baselines."""
        m = Metrics(state=self.current_state)

        if times is None or values is None:
            t_src, y_src, c_src = list(self._times), list(self._values), list(self._confs)
        else:
            t_src, y_src = list(times), list(values)
            c_src = list(confidences) if confidences is not None else None

        t_proc, y_proc = self._preprocess(t_src, y_src, c_src)
        if len(t_proc) == 0:
            return m

        t_now = t_proc[-1]
        m.t_now = t_now

        baseline = None
        pcps_now = None
        apcps = None
        apcps_covered = 0.0

        # Fixed reference baseline takes precedence
        if self.reference_baseline is not None and self.reference_baseline != 0.0:
            baseline = float(self.reference_baseline)
            pcps_now = 100.0 * (y_proc[-1] - baseline) / baseline
            ap_span = window_slice(t_proc, t_now - self.apcps_win, t_now)
            if ap_span is not None:
                i0, j0 = ap_span
                vals = y_proc[i0 : j0 + 1]
                if len(vals) > 0:
                    pcps_vals = (vals - baseline) / baseline * 100.0
                    apcps = float(np.mean(pcps_vals))
                    apcps_covered = min(self.apcps_win, t_proc[j0] - t_proc[i0])

        else:
            if self.continuous_baseline:
                span = window_slice(t_proc, t_now - self.baseline_win, t_now)
                baseline = mean_span(y_proc, span)
                if baseline and baseline != 0.0:
                    pcps_now = 100.0 * (y_proc[-1] - baseline) / baseline
                    ap_span = window_slice(t_proc, t_now - self.apcps_win, t_now)
                    if ap_span is not None:
                        i0, j0 = ap_span
                        vals = y_proc[i0 : j0 + 1]
                        if len(vals) > 0:
                            pcps_vals = (vals - baseline) / baseline * 100.0
                            apcps = float(np.mean(pcps_vals))
                            apcps_covered = min(self.apcps_win, t_proc[j0] - t_proc[i0])
            else:
                baseline = self.baseline_value
                if baseline and baseline != 0.0 and self.onset_t is not None and t_now >= self.onset_t:
                    pcps_now = 100.0* (y_proc[-1] - baseline) / baseline
                    ap_span = window_slice(t_proc, self.onset_t, self.onset_t + self.apcps_win)
                    if ap_span is not None:
                        i0, j0 = ap_span
                        vals = y_proc[i0 : j0 + 1]
                        if len(vals) > 0:
                            pcps_vals = (vals - baseline) / baseline * 100.0
                            apcps = float(np.mean(pcps_vals))
                            apcps_covered = min(self.apcps_win, t_proc[j0] - self.onset_t)

        m.baseline = baseline
        m.pcps_now = pcps_now
        m.apcps = apcps
        m.apcps_window_covered = apcps_covered

        # hysteresis based on APCPS if available, else PCPS
        trigger = apcps if apcps is not None else pcps_now
        if trigger is not None:
            if self.current_state == "blue" and trigger > self.alert_up:
                self.current_state = "red"
                self.switches.append((t_now, "red"))
            elif self.current_state == "red" and trigger < -self.alert_down:
                self.current_state = "blue"
                self.switches.append((t_now, "blue"))

        m.state = self.current_state
        return m

    # -------- offline series (vectorized, fast) --------
    def apcps_series(
        self,
        times: Sequence[float],
        values: Sequence[float],
        confidences: Optional[Sequence[float]] = None,
        mode: str = "sliding",   # "sliding" (last apcps_win) or "forward" (next apcps_win)
        baseline_win: Optional[float] = None,
        apcps_win: Optional[float] = None,
    ):
        """
        Offline APCPS time series:
          baseline(t) = rolling mean over previous baseline_win, or fixed reference_baseline if provided
          APCPS(t)    = mean PCPS over last/next apcps_win per 'mode'
        Returns: (t_proc, y_proc, apcps)
        """
        bw = float(baseline_win) if baseline_win is not None else self.baseline_win
        aw = float(apcps_win) if apcps_win is not None else self.apcps_win

        t_proc, y_proc = self._preprocess(times, values, confidences)
        if len(t_proc) == 0:
            return np.asarray([]), np.asarray([]), np.asarray([])

        # Baseline series
        if self.reference_baseline is not None and self.reference_baseline != 0.0:
            base = np.full_like(y_proc, float(self.reference_baseline), dtype=float)
        else:
            base = _rolling_mean_prev(t_proc, y_proc, bw)

        # Window mean (trailing or forward)
        if mode == "forward":
            mean_win = _forward_mean_next(t_proc, y_proc, aw)
        else:
            mean_win = _rolling_mean_prev(t_proc, y_proc, aw)  # trailing

        # APCPS
        apcps = np.full_like(mean_win, np.nan, dtype=float)
        mask = np.isfinite(base) & np.isfinite(mean_win) & (base != 0.0)
        apcps[mask] = 100.0 * (mean_win[mask] - base[mask]) / base[mask]

        return t_proc, y_proc, apcps

    # -------- plotting helper (unchanged) --------
    def mask_window_segments(self, t_window, y_window, tmin, tmax):
        """Return blue_y, red_y arrays per switch state over [tmin, tmax]."""
        while len(self.switches) > 1 and self.switches[1][0] < tmin:
            self.switches.pop(0)

        n = len(t_window)
        blue_y = np.full(n, np.nan, float)
        red_y = np.full(n, np.nan, float)
        if n == 0:
            return blue_y, red_y

        for idx, (t_sw, state) in enumerate(self.switches):
            seg_start = max(tmin, t_sw)
            seg_end = tmax if idx + 1 >= len(self.switches) else self.switches[idx + 1][0]
            if seg_end <= tmin or seg_start >= tmax:
                continue
            s = next((i for i, tt in enumerate(t_window) if tt >= seg_start), n)
            e = next((i for i, tt in enumerate(t_window) if tt >= seg_end), n)
            if s < e:
                if state == "blue":
                    blue_y[s:e] = y_window[s:e]
                else:
                    red_y[s:e] = y_window[s:e]
        return blue_y, red_y
