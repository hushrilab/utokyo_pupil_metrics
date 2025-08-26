#!/usr/bin/env python3
"""
Plot pupil diameter from a Pupil Capture export with:
  • confidence-colored trace (red/blue/green),
  • APCPS shading only where above/below thresholds,
  • optional confidence filtering and low-pass smoothing (from metrics_pupil),
  • optional FIXED reference baseline for percent changes,
  • fast offline APCPS (sliding or forward) computed by metrics_pupil.apcps_series()

CSV path template:
  ~/recordings/<SESSION_NAME>/<SESSION_NUMBER>/exports/<SESSION_NUMBER>/pupil_positions.csv
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from metrics_pupil import PCPSCalculator

plt.rcParams.update({
    "font.size": 14,
    "axes.titlesize": 16,
    "axes.labelsize": 14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12
})

def normalize_to_seconds(t: np.ndarray) -> np.ndarray:
    """Normalize timestamps to seconds starting at 0 (handles ns/µs/s)."""
    t = np.asarray(pd.to_numeric(t, errors="coerce"))
    t0 = np.nanmin(t)
    dt = t - t0
    max_abs = np.nanmax(np.abs(dt))
    if max_abs > 1e12:   # ns
        dt = dt / 1e9
    elif max_abs > 1e6:  # µs
        dt = dt / 1e6
    return dt

def main():
    ap = argparse.ArgumentParser(description="Plot pupil diameter + APCPS threshold shading")
    ap.add_argument("session_name", type=str, help="Session name under ~/recordings/")
    ap.add_argument("--session-number", type=str, default="000",
                    help="Recording session number folder (default: 000)")
    ap.add_argument("--field", choices=["diameter", "diameter_3d"], default="diameter_3d")
    # Confidence coloring thresholds
    ap.add_argument("--thresh-low", type=float, default=0.6)
    ap.add_argument("--thresh-high", type=float, default=0.8)
    # APCPS thresholds (for shading only)
    ap.add_argument("--alert-up", type=float, default=15.0)
    ap.add_argument("--alert-down", type=float, default=None)
    # APCPS window mode (offline)
    ap.add_argument("--forward-apcps", action="store_true",
                    help="Use forward 2.5 s APCPS (next window). Otherwise sliding (last 2.5 s).")
    # Filtering (applied inside metrics_pupil)
    ap.add_argument("--conf-thresh", type=float, default=None,
                    help="Drop samples with confidence below this value (if confidence column available).")
    ap.add_argument("--lp-cutoff-hz", type=float, default=None,
                    help="Low-pass cutoff in Hz before computing metrics.")
    ap.add_argument("--use-filtered", action="store_true",
                    help="Plot the filtered (confidence + low-pass) signal instead of raw.")
    # NEW: fixed reference baseline (overrides rolling baseline)
    ap.add_argument("--ref-baseline", type=float, default=None,
                    help="Fixed baseline value for percent changes (same units as the signal). If set, overrides the previous-1s rolling baseline. Must be non-zero.")
    args = ap.parse_args()

    # ---- Load CSV ----
    csv_path = Path.home() / "recordings" / args.session_name / args.session_number / "exports" / args.session_number / "pupil_positions.csv"
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
    t_raw = normalize_to_seconds(df["pupil_timestamp"].to_numpy())
    y_raw = pd.to_numeric(df[args.field], errors="coerce").to_numpy()
    c_raw = pd.to_numeric(df["confidence"], errors="coerce").to_numpy()

    # Validate reference baseline (if given)
    if args.ref_baseline is not None and args.ref_baseline == 0.0:
        raise ValueError("--ref-baseline must be non-zero.")

    # ---- Calculator (offline config; rolling baseline is internal, unless ref is set) ----
    calc = PCPSCalculator(
        baseline_win=1.0,
        apcps_win=2.5,
        alert_up=args.alert_up,
        alert_down=args.alert_down,
        continuous_baseline=True,            # offline series uses rolling baseline unless ref is provided
        conf_thresh=args.conf_thresh,
        lp_cutoff_hz=args.lp_cutoff_hz,
        reference_baseline=args.ref_baseline
    )

    # ---- Preprocess once (filter + smooth) for plotting if requested ----
    t_proc, y_proc = calc._preprocess(t_raw, y_raw, confidences=c_raw)

    if args.use_filtered:
        t_plot, y_plot = t_proc, y_proc
        # align confidence to filtered indices (no interpolation)
        mask = np.isin(t_raw, t_proc)
        c_plot = c_raw[mask]
    else:
        t_plot, y_plot, c_plot = t_raw, y_raw, c_raw

    # ---- Compute APCPS series via metrics (fast, offline) ----
    mode = "forward" if args.forward_apcps else "sliding"
    t_ap, y_ap, apcps_vals = calc.apcps_series(t_plot, y_plot, confidences=c_plot, mode=mode)

    # ---- Plot ----
    fig, ax = plt.subplots(figsize=(12, 6))
    unit = "mm" if args.field == "diameter_3d" else "px"

    # Confidence masks on the plotted data
    low_band  = c_plot < args.thresh_low
    mid_band  = (c_plot >= args.thresh_low) & (c_plot < args.thresh_high)
    high_band = c_plot >= args.thresh_high

    ax.plot(t_plot, np.where(low_band,  y_plot, np.nan), lw=1.5, color="red",   label=f"conf < {args.thresh_low}")
    ax.plot(t_plot, np.where(mid_band,  y_plot, np.nan), lw=1.5, color="blue",  label=f"{args.thresh_low} ≤ conf < {args.thresh_high}")
    ax.plot(t_plot, np.where(high_band, y_plot, np.nan), lw=1.5, color="green", label=f"conf ≥ {args.thresh_high}")
    ax.set_xlabel("Time since start (s)")
    ax.set_ylabel(f"Pupil diameter ({unit})")

    # Shading ONLY where APCPS crosses thresholds
    ax2 = ax.twinx()
    ax2.set_ylabel("APCPS (%)")
    up = float(args.alert_up)
    down = float(args.alert_down) if args.alert_down is not None else up

    ap = apcps_vals.copy()
    high_mask = (~np.isnan(ap)) & (ap >  up)
    low_mask  = (~np.isnan(ap)) & (ap < -down)

    ax2.fill_between(t_ap, 0, np.where(high_mask, ap, np.nan),
                     color="red", alpha=0.30, label=f"APCPS > +{up}%")
    ax2.fill_between(t_ap, 0, np.where(low_mask,  ap, np.nan),
                     color="blue", alpha=0.30, label=f"APCPS < -{down}%")

    mode_txt = "forward APCPS (next 2.5 s)" if args.forward_apcps else "sliding APCPS (last 2.5 s)"
    base_txt = f"reference baseline = {args.ref_baseline:g} {unit}" if args.ref_baseline is not None else "continuous 1 s baseline"
    filt_txt = " (filtered)" if args.use_filtered else " (raw)"
    ax.set_title(f"Pupil diameter{filt_txt} + APCPS crossings — session: {args.session_name} | {mode_txt}, {base_txt}")

    ax.legend(loc="upper left")
    ax2.legend(loc="upper right")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
