#!/usr/bin/env python3
"""
Plot pupil diameter from a Pupil Capture export with:
  • confidence-colored trace (red/blue/green),
  • APCPS shading only where above/below thresholds,
  • optional confidence filtering and low-pass smoothing from metrics_core.

CSV path:
  ~/recordings/<SESSION_NAME>/000/exports/000/pupil_positions.csv
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from metrics_pupil import PCPSCalculator  # <-- updated class with conf + LP filtering

plt.rcParams.update({
    "font.size": 14,
    "axes.titlesize": 16,
    "axes.labelsize": 14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12
})

def normalize_to_seconds(t: np.ndarray) -> np.ndarray:
    t = np.asarray(pd.to_numeric(t, errors="coerce"))
    t0 = np.nanmin(t)
    dt = t - t0
    max_abs = np.nanmax(np.abs(dt))
    if max_abs > 1e12:       # ns
        dt = dt / 1e9
    elif max_abs > 1e6:      # µs
        dt = dt / 1e6
    return dt

def main():
    ap = argparse.ArgumentParser(description="Plot pupil diameter + APCPS threshold shading")
    ap.add_argument("session_name", type=str, help="Session name under ~/recordings/")
    ap.add_argument("--field", choices=["diameter", "diameter_3d"], default="diameter")
    # Confidence bands for trace coloring
    ap.add_argument("--thresh-low", type=float, default=0.6)
    ap.add_argument("--thresh-high", type=float, default=0.8)
    # APCPS thresholds
    ap.add_argument("--alert-up", type=float, default=15.0)
    ap.add_argument("--alert-down", type=float, default=None)
    # Baseline mode
    ap.add_argument("--continuous-baseline", action="store_true")
    ap.add_argument("--onset", type=float, default=None,
                    help="Manual onset time in seconds (baseline = 1 s before onset). "
                         "If omitted, continuous mode is used by default.")
    # NEW: filtering options
    ap.add_argument("--conf-thresh", type=float, default=None,
                    help="Drop samples with confidence below this value (default: keep all)")
    ap.add_argument("--lp-cutoff-hz", type=float, default=None,
                    help="Apply low-pass filter with cutoff freq in Hz (default: none)")
    ap.add_argument("--use-filtered", action="store_true",
                    help="Plot filtered data (confidence+LP) instead of raw")
    args = ap.parse_args()

    # ---- Load CSV ----
    csv_path = Path.home() / "recordings" / args.session_name / "000" / "exports" / "000" / "pupil_positions.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path, low_memory=False)
    for col in ["pupil_timestamp", "confidence", args.field]:
        if col not in df.columns:
            raise KeyError(f"Missing column '{col}' in {csv_path}")

    df = df[["pupil_timestamp", "confidence", args.field]].dropna()
    if df.empty:
        raise ValueError("No valid data after dropping NaNs.")

    # ---- Arrays ----
    t = normalize_to_seconds(df["pupil_timestamp"].to_numpy())
    y = pd.to_numeric(df[args.field], errors="coerce").to_numpy()
    c = pd.to_numeric(df["confidence"], errors="coerce").to_numpy()

    # ---- Metrics calc (for filtering + APCPS) ----
        # ---- Metrics calc (for filtering + APCPS) ----
    use_continuous = args.continuous_baseline or (args.onset is None)
    calc = PCPSCalculator(
        baseline_win=1.0,
        apcps_win=2.5,
        alert_up=args.alert_up,
        alert_down=args.alert_down,
        continuous_baseline=use_continuous,
        forward_apcps=False,
        conf_thresh=args.conf_thresh,
        lp_cutoff_hz=args.lp_cutoff_hz
    )

    # preprocess once to get filtered signal (for plotting if requested)
    t_proc, y_proc = calc._preprocess(t, y, confidences=c)

    if args.use_filtered:
        # Drop samples (no interpolation!)
        t_plot, y_plot = t_proc, y_proc
        # Also drop confidence values for consistency
        mask = np.isin(t, t_proc)
        c_plot = c[mask]
    else:
        t_plot, y_plot, c_plot = t, y, c

    # ---- Compute APCPS series ----
    apcps_vals = np.full_like(t_plot, np.nan, dtype=float)
    for i in range(len(t_plot)):
        m = calc.update(list(t_plot[:i+1]), list(y_plot[:i+1]), confidences=c_plot[:i+1] if len(c_plot)==len(t_plot) else None)
        apcps_vals[i] = m.apcps if m.apcps is not None else np.nan

    # ---- Plot ----
    fig, ax = plt.subplots(figsize=(12, 6))
    unit = "mm" if args.field == "diameter_3d" else "px"

    # Confidence masks for coloring
    low_band  = c_plot < args.thresh_low
    mid_band  = (c_plot >= args.thresh_low) & (c_plot < args.thresh_high)
    high_band = c_plot >= args.thresh_high

    ax.plot(t_plot, np.where(low_band,  y_plot, np.nan), lw=1.5, color="red",   label=f"conf < {args.thresh_low}")
    ax.plot(t_plot, np.where(mid_band,  y_plot, np.nan), lw=1.5, color="blue",  label=f"{args.thresh_low} ≤ conf < {args.thresh_high}")
    ax.plot(t_plot, np.where(high_band, y_plot, np.nan), lw=1.5, color="green", label=f"conf ≥ {args.thresh_high}")
    ax.set_xlabel("Time since start (s)")
    ax.set_ylabel(f"Pupil diameter ({unit})")

    # APCPS shading (only > thresholds)
    ax2 = ax.twinx()
    ax2.set_ylabel("APCPS (%)")
    up = float(args.alert_up)
    down = float(args.alert_down) if args.alert_down is not None else up

    ap = apcps_vals.copy()
    high_mask = (~np.isnan(ap)) & (ap >  up)
    low_mask  = (~np.isnan(ap)) & (ap < -down)

    ax2.fill_between(t_plot, 0, np.where(high_mask, ap, np.nan), color="red",  alpha=0.3, label=f"APCPS > +{up}%")
    ax2.fill_between(t_plot, 0, np.where(low_mask,  ap, np.nan), color="blue", alpha=0.3, label=f"APCPS < -{down}%")

    mode_txt = "continuous 1s baseline" if use_continuous else f"manual onset @ {args.onset:.2f}s"
    filt_txt = " (filtered)" if args.use_filtered else " (raw)"
    ax.set_title(f"Pupil diameter{filt_txt} + APCPS crossings — session: {args.session_name} | {mode_txt}")

    ax.legend(loc="upper left")
    ax2.legend(loc="upper right")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
