#!/usr/bin/env python3
"""
Plot pupil diameter from a Pupil Capture export, color-coded by confidence.

Input CSV:
  ~/recordings/<SESSION_NAME>/000/exports/000/pupil_positions.csv
Required columns:
  pupil_timestamp, confidence, diameter (or diameter_3d)

Color bands (fixed for visualization):
  confidence < 0.6         -> red
  0.6 <= confidence < 0.8  -> blue
  confidence >= 0.8        -> green

Options:
  --field diameter_3d        Plot 3D diameter in mm (if present)
  --filter-lowconf           Drop all samples with confidence < FILTER_THRESH
  --filter-thresh FLOAT      Threshold used for filtering (default: 0.6)
  --interp-lowconf           Interpolate across filtered gaps for a continuous trace
  --interp-max-gap SECONDS   Max gap to fill by interpolation (default: 2.0 s)
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---- Global font settings ----
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
    if max_abs > 1e12:       # nanoseconds
        dt = dt / 1e9
    elif max_abs > 1e6:      # microseconds
        dt = dt / 1e6
    return dt

def interpolate_gaps(t: np.ndarray, y: np.ndarray, keep_mask: np.ndarray, max_gap_s: float) -> np.ndarray:
    """
    Linearly interpolate y over t only inside gaps between kept samples and only
    if the gap duration <= max_gap_s. Returns array with NaNs where not filled.
    """
    y_nan = y.astype(float).copy()
    y_nan[~keep_mask] = np.nan
    s = pd.Series(y_nan, index=t)
    y_interp = s.interpolate(method="index", limit_area="inside").to_numpy()

    valid_idx = np.where(keep_mask)[0]
    if valid_idx.size >= 2:
        for i in range(valid_idx.size - 1):
            a, b = valid_idx[i], valid_idx[i + 1]
            if (t[b] - t[a]) > max_gap_s:
                y_interp[a+1:b] = np.nan

    y_interp[keep_mask] = y[keep_mask]
    return y_interp

def main():
    ap = argparse.ArgumentParser(description="Plot pupil diameter color-coded by confidence.")
    ap.add_argument("session_name", type=str, help="Session folder under ~/recordings/")
    ap.add_argument("--field", choices=["diameter", "diameter_3d"], default="diameter",
                    help="Y-axis column (default: diameter; use diameter_3d for mm)")
    ap.add_argument("--filter-lowconf", action="store_true",
                    help="Filter out all samples with confidence below --filter-thresh")
    ap.add_argument("--filter-thresh", type=float, default=0.6,
                    help="Confidence threshold used for filtering (default: 0.6)")
    ap.add_argument("--interp-lowconf", action="store_true",
                    help="Interpolate across filtered gaps for a continuous trace")
    ap.add_argument("--interp-max-gap", type=float, default=2.0,
                    help="Max gap (s) to fill via interpolation (default: 2.0)")
    args = ap.parse_args()

    csv_path = Path.home() / "recordings" / args.session_name / "000" / "exports" / "000" / "pupil_positions.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path, low_memory=False)

    for col in ["pupil_timestamp", "confidence", args.field]:
        if col not in df.columns:
            raise KeyError(f"Missing required column '{col}'. Available: {list(df.columns)}")

    df = df[["pupil_timestamp", "confidence", args.field]].dropna()
    if df.empty:
        raise ValueError("No valid data after dropping NaNs.")

    t = normalize_to_seconds(df["pupil_timestamp"].to_numpy())
    y = pd.to_numeric(df[args.field], errors="coerce").to_numpy()
    c = pd.to_numeric(df["confidence"], errors="coerce").to_numpy()

    # Fixed bands for coloring
    low_band  = c < 0.6
    mid_band  = (c >= 0.6) & (c < 0.8)
    high_band = c >= 0.8

    plt.figure(figsize=(12, 6))

    if args.filter_lowconf:
        # Filter using user-specified threshold
        keep_mask = c >= args.filter_thresh
        t_f, y_f, c_f = t[keep_mask], y[keep_mask], c[keep_mask]

        # Optional interpolation across removed samples (uses original timebase)
        if args.interp_lowconf:
            y_interp = interpolate_gaps(t, y, keep_mask, args.interp_max_gap)
            plt.plot(t, y_interp, lw=1.2, color="0.5", alpha=0.9,
                     label=f"interpolated (≤ {args.interp_max_gap:.1f}s gaps)")

        # Recompute bands on filtered data only for coloring
        mid_f  = (c_f >= 0.6) & (c_f < 0.8)
        high_f = c_f >= 0.8

        # Draw the filtered colored lines (note: low band <0.6 not shown after filtering)
        y_mid  = np.where(mid_f,  y_f, np.nan)
        y_high = np.where(high_f, y_f, np.nan)
        plt.plot(t_f, y_mid,  lw=1.5, color="blue",  label="0.6 ≤ confidence < 0.8")
        plt.plot(t_f, y_high, lw=1.5, color="green", label="confidence ≥ 0.8")
    else:
        # No filtering: draw all three bands
        y_low  = np.where(low_band,  y, np.nan)
        y_mid  = np.where(mid_band,  y, np.nan)
        y_high = np.where(high_band, y, np.nan)
        plt.plot(t, y_low,  lw=1.5, color="red",   label="confidence < 0.6")
        plt.plot(t, y_mid,  lw=1.5, color="blue",  label="0.6 ≤ confidence < 0.8")
        plt.plot(t, y_high, lw=1.5, color="green", label="confidence ≥ 0.8")

    unit = "mm" if args.field == "diameter_3d" else "px"
    plt.xlabel("Time since start (s)")
    plt.ylabel(f"Pupil diameter ({unit})")
    filt_txt = f" (filtered @<{args.filter_thresh})" if args.filter_lowconf else ""
    plt.title(f"Pupil diameter — session: {args.session_name}{filt_txt}")
    plt.legend(loc="best")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()