#!/usr/bin/env python3
"""
Computation & state utilities for realtime/offline pupil plotting (PCPS/APCPS),
with optional confidence-based exclusion and low-pass smoothing.

Features
--------
- Baseline & PCPS/APCPS as previously defined:
  • Manual onset mode:
      baseline = mean over the 1 s immediately BEFORE onset
      APCPS    = mean PCPS over the first 2.5 s AFTER onset
  • Continuous baseline mode:
      baseline(t) = rolling mean over the last 1 s
      APCPS:
         - sliding (default): mean PCPS over [t-2.5, t]
         - forward: mean PCPS over (t, t+2.5]

- NEW: Confidence-based exclusion (if confidences are provided to update/set_onset)
  Exclude samples where confidence < conf_thresh.

- NEW: Low-pass smoothing (first-order IIR / exponential)
  Applies to the Y-signal BEFORE baseline/PCPS/APCPS calculations.
  Handles IRREGULAR sampling using per-sample dt:
      tau = 1 / (2π * cutoff_hz)
      alpha_t = 1 - exp(-dt / tau)
      y_filt[n] = y_filt[n-1] + alpha_t * (y[n] - y_filt[n-1])

Public API
----------
- class PCPSCalculator(...):
    __init__(..., conf_thresh: Optional[float] = None,
                  lp_cutoff_hz: Optional[float] = None)

    set_onset(t_now, times, values, confidences=None)
    update(times, values, confidences=None) -> Metrics
    mask_window_segments(t_window, y_window, tmin, tmax) -> (blue_y, red_y)

- dataclass Metrics: t_now, baseline, pcps_now, apcps, apcps_window_covered, state
"""

from dataclasses import dataclass
from statistics import mean
from typing import List, Optional, Sequence, Tuple
import math
import numpy as np


# ---------- helpers ----------
def window_slice(times: Sequence[float], start_t: float, end_t: float) -> Optional[Tuple[int, int]]:
    """Return [i0, j0] indices with start_t < t <= end_t (None if empty)."""
    if len(times) == 0:
        return None
    # times assumed non-decreasing
    i0 = next((i for i, t in enumerate(times) if t > start_t), None)
    j0 = next((i for i in range(len(times) - 1, -1, -1) if times[i] <= end_t), None)
    if i0 is None or j0 is None or i0 > j0:
        return None
    return i0, j0


def mean_span(values: Sequence[float], span: Optional[Tuple[int, int]]) -> Optional[float]:
    """Mean of values[i0..j0] or None."""
    if span is None:
        return None
    i0, j0 = span
    if i0 > j0:
        return None
    seg = values[i0:j0 + 1]
    return mean(seg) if len(seg) > 0 else None


def _as_np(a) -> np.ndarray:
    return np.asarray(a, dtype=float)


def _prep_mask(times: np.ndarray, values: np.ndarray,
               confidences: Optional[np.ndarray],
               conf_thresh: Optional[float]) -> np.ndarray:
    """Build a boolean mask selecting valid samples after NaN and confidence filtering."""
    valid = np.isfinite(times) & np.isfinite(values)
    if confidences is not None:
        valid &= np.isfinite(confidences)
        if conf_thresh is not None:
            valid &= confidences >= conf_thresh
    return valid


def _lowpass_exp(times: np.ndarray, values: np.ndarray, cutoff_hz: float) -> np.ndarray:
    """First-order IIR low-pass for IRREGULAR sampling. Returns filtered values (same length)."""
    if cutoff_hz is None or cutoff_hz <= 0 or len(values) == 0:
        return values.copy()
    y = np.empty_like(values, dtype=float)
    y[0] = values[0]
    tau = 1.0 / (2.0 * math.pi * cutoff_hz)  # seconds
    for i in range(1, len(values)):
        dt = max(times[i] - times[i - 1], 0.0)
        if not np.isfinite(dt) or dt <= 0:
            # No time advance; carry forward
            y[i] = y[i - 1]
            continue
        alpha = 1.0 - math.exp(-dt / tau)
        y[i] = y[i - 1] + alpha * (values[i] - y[i - 1])
    return y


# ---------- data structures ----------
@dataclass
class Metrics:
    t_now: Optional[float] = None
    baseline: Optional[float] = None
    pcps_now: Optional[float] = None
    apcps: Optional[float] = None
    apcps_window_covered: float = 0.0
    state: str = "blue"  # 'blue' or 'red'


class PCPSCalculator:
    def __init__(
        self,
        baseline_win: float = 1.0,
        apcps_win: float = 2.5,
        alert_up: float = 15.0,
        alert_down: Optional[float] = None,
        continuous_baseline: bool = False,
        forward_apcps: bool = False,
        # NEW:
        conf_thresh: Optional[float] = None,
        lp_cutoff_hz: Optional[float] = None,
    ):
        self.baseline_win = float(baseline_win)
        self.apcps_win = float(apcps_win)
        self.alert_up = float(alert_up)
        self.alert_down = float(alert_down) if alert_down is not None else float(alert_up)
        self.continuous_baseline = bool(continuous_baseline)
        self.forward_apcps = bool(forward_apcps)

        # new options
        self.conf_thresh = conf_thresh
        self.lp_cutoff_hz = lp_cutoff_hz

        # color switches: list of (t_switch, 'blue'|'red'); start blue at t=0
        self.switches: List[Tuple[float, str]] = [(0.0, "blue")]
        self.current_state: str = "blue"

        # manual onset mode state
        self.onset_t: Optional[float] = None
        self.baseline_value: Optional[float] = None

    # --- configuration setters (optional convenience) ---
    def set_confidence_threshold(self, thresh: Optional[float]):
        self.conf_thresh = thresh

    def set_lowpass_cutoff(self, cutoff_hz: Optional[float]):
        self.lp_cutoff_hz = cutoff_hz

    # --- preprocessing pipeline ---
    def _preprocess(
        self,
        times: Sequence[float],
        values: Sequence[float],
        confidences: Optional[Sequence[float]] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply confidence filtering and low-pass smoothing. Returns (t_proc, y_proc).
        If no samples remain, returns empty arrays.
        """
        t = _as_np(times)
        y = _as_np(values)
        c = _as_np(confidences) if confidences is not None else None

        mask = _prep_mask(t, y, c, self.conf_thresh)
        if not mask.any():
            return np.array([], dtype=float), np.array([], dtype=float)

        t2 = t[mask]
        y2 = y[mask]

        # Ensure non-decreasing time (most exports are already sorted)
        # If needed, we could stable-sort by t2, but assume sorted for speed.

        # Low-pass smoothing on remaining samples
        y_smooth = _lowpass_exp(t2, y2, self.lp_cutoff_hz) if self.lp_cutoff_hz else y2.copy()
        return t2, y_smooth

    # --- manual onset baseline ---
    def set_onset(
        self,
        t_now: float,
        times: Sequence[float],
        values: Sequence[float],
        confidences: Optional[Sequence[float]] = None,
    ) -> None:
        """
        Mark stimulus onset at t_now and compute baseline from the 1 s immediately before,
        after applying confidence filtering and low-pass smoothing.
        """
        if self.continuous_baseline:
            return
        self.onset_t = t_now

        t_proc, y_proc = self._preprocess(times, values, confidences)
        if len(t_proc) == 0:
            self.baseline_value = None
            return

        span = window_slice(t_proc, t_now - self.baseline_win, t_now)
        self.baseline_value = mean_span(y_proc, span)

    # --- main update ---
    def update(
        self,
        times: Sequence[float],
        values: Sequence[float],
        confidences: Optional[Sequence[float]] = None,
    ) -> Metrics:
        """
        Compute PCPS/APCPS at the latest time using preprocessed (filtered + smoothed) data
        and update hysteresis color state. Returns a Metrics snapshot.
        """
        m = Metrics(state=self.current_state)

        t_proc, y_proc = self._preprocess(times, values, confidences)
        if len(t_proc) == 0:
            return m

        t_now = t_proc[-1]
        m.t_now = t_now

        baseline = None
        pcps_now = None
        apcps = None
        apcps_covered = 0.0

        if self.continuous_baseline:
            # Rolling baseline over last 1 s
            span = window_slice(t_proc, t_now - self.baseline_win, t_now)
            baseline = mean_span(y_proc, span)

            if baseline and baseline != 0:
                # instantaneous PCPS
                pcps_now = 100.0 * (y_proc[-1] - baseline) / baseline

                # APCPS window
                if self.forward_apcps:
                    ap_span = window_slice(t_proc, t_now, t_now + self.apcps_win)
                else:
                    ap_span = window_slice(t_proc, t_now - self.apcps_win, t_now)

                if ap_span is not None:
                    i0, j0 = ap_span
                    vals = y_proc[i0:j0 + 1]
                    if len(vals) > 0:
                        pcps_vals = [100.0 * (v - baseline) / baseline for v in vals]
                        apcps = mean(pcps_vals)
                        apcps_covered = min(self.apcps_win, t_proc[j0] - t_proc[i0])

        else:
            # Manual onset mode
            baseline = self.baseline_value
            if baseline and baseline != 0 and self.onset_t is not None and t_now >= self.onset_t:
                pcps_now = 100.0 * (y_proc[-1] - baseline) / baseline
                ap_span = window_slice(t_proc, self.onset_t, self.onset_t + self.apcps_win)
                if ap_span is not None:
                    i0, j0 = ap_span
                    vals = y_proc[i0:j0 + 1]
                    if len(vals) > 0:
                        pcps_vals = [100.0 * (v - baseline) / baseline for v in vals]
                        apcps = mean(pcps_vals)
                        apcps_covered = min(self.apcps_win, t_proc[j0] - self.onset_t)

        m.baseline = baseline
        m.pcps_now = pcps_now
        m.apcps = apcps
        m.apcps_window_covered = apcps_covered

        # Hysteresis color switching (based on APCPS if available, else PCPS)
        trigger_val = apcps if apcps is not None else pcps_now
        if trigger_val is not None:
            if self.current_state == "blue" and trigger_val > self.alert_up:
                self.current_state = "red"
                self.switches.append((t_now, "red"))
            elif self.current_state == "red" and trigger_val < -self.alert_down:
                self.current_state = "blue"
                self.switches.append((t_now, "blue"))

        m.state = self.current_state
        return m

    # --- plotting helper (unchanged) ---
    def mask_window_segments(self, t_window, y_window, tmin, tmax):
        """
        Given windowed time/value arrays, return two masked arrays (blue_y, red_y)
        following the switch list within [tmin, tmax].
        """
        # prune old switches (keep at least one)
        while len(self.switches) > 1 and self.switches[1][0] < tmin:
            self.switches.pop(0)

        tvw = list(t_window)
        yvw = list(y_window)
        n = len(tvw)
        blue_y = np.full(n, np.nan, float)
        red_y = np.full(n, np.nan, float)
        if n == 0:
            return blue_y, red_y

        for idx, (t_sw, state) in enumerate(self.switches):
            seg_start = max(tmin, t_sw)
            seg_end = tmax
            if idx + 1 < len(self.switches):
                seg_end = self.switches[idx + 1][0]
            if seg_end <= tmin or seg_start >= tmax:
                continue
            s = next((i for i, t in enumerate(tvw) if t >= seg_start), n)
            e = next((i for i, t in enumerate(tvw) if t >= seg_end), n)
            if s < e:
                if state == "blue":
                    blue_y[s:e] = yvw[s:e]
                else:
                    red_y[s:e] = yvw[s:e]
        return blue_y, red_y
