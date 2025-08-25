#!/usr/bin/env python3
"""
Plot pupil diameter from a Pupil Capture export, color-coded by confidence.

Input CSV:
  ~/recordings/<SESSION_NAME>/000/exports/000/pupil_positions.csv
Required columns:
  pupil_timestamp, confidence, diameter (or diameter_3d)

Colors:
  confidence < 0.6         -> blue
  0.6 <= confidence < 0.8  -> red
  confidence >= 0.8        -> green
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---- Global font settings ----
plt.rcParams.update({
    "font.size": 14,        # base font size
    "axes.titlesize": 16,   # title
    "axes.labelsize": 14,   # x/y labels
    "xtick.labelsize": 12,  # x tick labels
    "ytick.labelsize": 12,  # y tick labels
    "legend.fontsize": 12   # legend
})

def normalize_to_seconds(t: np.ndarray) -> np.ndarray:
    """Normalize timestamps to seconds starting at 0 (handles ns/µs/s)."""
    t = np.asarray(pd.to_numeric(t, errors="coerce"))
    t0 = np.nanmin(t)
    dt = t - t0
    max_abs = np.nanmax(np.abs(dt))
    if max_abs > 1e12:    # nanoseconds
        dt = dt / 1e9
    elif max_abs > 1e6:   # microseconds
        dt = dt / 1e6
    return dt

def main():
    ap = argparse.ArgumentParser(description="Plot pupil diameter color-coded by confidence.")
    ap.add_argument("session_name", type=str, help="Session folder under ~/recordings/")
    ap.add_argument("--field", choices=["diameter", "diameter_3d"], default="diameter",
                    help="Y-axis column (default: diameter; use diameter_3d for mm)")
    args = ap.parse_args()

    csv_path = Path.home() / "recordings" / args.session_name / "000" / "exports" / "000" / "pupil_positions.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path, low_memory=False)

    # Sanity checks
    for col in ["pupil_timestamp", "confidence", args.field]:
        if col not in df.columns:
            raise KeyError(f"Missing required column '{col}'. Available: {list(df.columns)}")

    # Keep needed cols and drop NaNs
    df = df[["pupil_timestamp", "confidence", args.field]].dropna()
    if df.empty:
        raise ValueError("No valid data after dropping NaNs.")

    # Time (s from 0)
    t = normalize_to_seconds(df["pupil_timestamp"].to_numpy())
    y = pd.to_numeric(df[args.field], errors="coerce").to_numpy()
    c = pd.to_numeric(df["confidence"], errors="coerce").to_numpy()

    # Confidence masks
    low = c < 0.6
    mid = (c >= 0.6) & (c < 0.8)
    high = c >= 0.8

    # Masked series (plot as continuous per band)
    y_low  = np.where(low,  y, np.nan)
    y_mid  = np.where(mid,  y, np.nan)
    y_high = np.where(high, y, np.nan)

    # Plot
    plt.figure(figsize=(12, 6))
    plt.plot(t, y_low,  lw=1.5, color="blue",  label="confidence < 0.6")
    plt.plot(t, y_mid,  lw=1.5, color="red",   label="0.6 ≤ confidence < 0.8")
    plt.plot(t, y_high, lw=1.5, color="green", label="confidence ≥ 0.8")

    unit = "mm" if args.field == "diameter_3d" else "px"
    plt.xlabel("Time [s]")
    plt.ylabel(f"Pupil diameter ({unit})")
    plt.title(f"Pupil diameter — session: {args.session_name}")
    plt.legend(loc="best")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()