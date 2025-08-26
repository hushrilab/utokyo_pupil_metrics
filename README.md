# Pupil Labs Pupil Diameter Analysis & Metrics

This repository provides tools for connecting to a [Pupil Labs](https://pupil-labs.com/) eye tracker in realtime and for offline analysis of exported data.  
It implements **PCPS** (Percent Change in Pupil Size) and **APCPS** (Average PCPS) metrics with options for:

- Realtime connection and recording
- Realtime sliding APCPS visualization
- Offline plotting of pupil diameter traces with **confidence-based coloring**
- APCPS threshold shading (sliding or forward windows)
- Confidence filtering and low-pass smoothing
- Reference baselines (numeric or derived from another session/file)

---

## Repository Structure

```
.
├── metrics_pupil.py       # Core metrics computation (PCPS/APCPS, filters, baselines)
├── pupil_labs_connect.py  # Realtime script: connect to eye tracker, live plot, record
├── plot_data.py           # Offline plotting script: CSV export analysis
├── requirements.txt       # Python dependencies (see below)
└── README.md              # This file
```

---

## Installation

Clone the repo and install dependencies (Python 3.8+ recommended):

```bash
git clone https://github.com/<your-username>/pupil-metrics.git
cd pupil-metrics
pip install -r requirements.txt
```

### Requirements

The repository depends on the following Python packages:

```
numpy
pandas
matplotlib
pyzmq
msgpack
```

---

## Usage

### 1. Realtime connection (`pupil_labs_connect.py`)

Connects to a Pupil Labs device (USB) and streams pupil diameter with **sliding APCPS** analysis.  
Recording can be started/stopped via keyboard input.

```bash
python pupil_labs_connect.py --addr 127.0.0.1 --lp-cutoff-hz 0.5 --conf-thresh 0.6
```

Key bindings:
- `o` = mark onset (if not using continuous baseline)
- `t` = stop recording
- `q` = quit

Optional arguments:
- `--conf-thresh <val>`: drop samples below this confidence
- `--lp-cutoff-hz <val>`: apply low-pass filter before metrics
- `--ref-baseline <val>`: use a fixed reference baseline instead of rolling 1 s

### 2. Offline plotting (`plot_data.py`)

Analyzes exported `pupil_positions.csv` from Pupil Capture.  
Generates a figure with:
- Colored trace: red/blue/green for low/mid/high confidence
- Shaded areas: APCPS excursions above/below thresholds
- Optional baseline line (fixed or from another session/file)

Example:

```bash
# Sliding APCPS (default), rolling baseline
python plot_data.py mysession --session-number 000

# Forward APCPS (next 2.5 s window), filtered trace
python plot_data.py mysession --session-number 000 --forward-apcps --use-filtered --conf-thresh 0.6 --lp-cutoff-hz 0.5

# Use a fixed numeric reference baseline
python plot_data.py mysession --session-number 000 --ref-baseline 60

# Compute baseline from another file, using confidence ≥0.8 and optional low-pass
python plot_data.py mysession --session-number 000   --ref-baseline-file ~/recordings/ref_sess/000/exports/000/pupil_positions.csv   --lp-cutoff-hz 0.5
```

---

## Metrics

- **PCPS**:  
  \[
  \text{PCPS}(t) = 100 \times \frac{y(t) - \text{baseline}}{\text{baseline}}
  \]

- **APCPS (sliding)**:  
  Average PCPS over the **last** `apcps_win` seconds (causal, works online).

- **APCPS (forward)**:  
  Average PCPS over the **next** `apcps_win` seconds (offline-only, for post-baseline analysis).

Baselines:
- **Continuous rolling baseline**: mean pupil size over the last 1 s
- **Manual onset baseline**: fixed from 1 s before onset (realtime only)
- **Reference baseline**: fixed numeric value or computed mean from another dataset

---

## Example Figures

- Pupil diameter trace, color-coded by confidence
- APCPS excursions shaded red (increase > +threshold) or blue (decrease < -threshold)
- Optional horizontal dashed line = reference baseline

---

## Notes

- `pupil_labs_connect.py` is designed for **online visualization**, always using sliding APCPS.  
- `plot_data.py` is optimized for **offline batch analysis** and supports both sliding and forward APCPS.  
- When using `--ref-baseline-file`, only samples with `confidence ≥ 0.8` are used, and the same low-pass filter (`--lp-cutoff-hz`) is applied before computing the baseline.

---

## License

MIT License. See [LICENSE](LICENSE) for details.
