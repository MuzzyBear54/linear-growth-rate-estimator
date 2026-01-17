# linear-growth-rate-estimator

Automated detection of linear (exponential) instability growth phases and growth rates from spectral mode amplitudes using rolling log-linear fits, SNR/R² gating, and plateau selection.

> This repository is meant to be a **small, practical tool** for extracting growth rates from noisy simulation diagnostics (especially PIC field outputs), while robustly identifying the linear regime and avoiding early-time noise and late-time saturation.

## What it does

Given a time sequence of 1D field snapshots `E(x, t)` (e.g., `Ex` from Smilei):

- Tracks the amplitude of a target spectral mode (FFT bin or continuous-k projection)
- Computes growth rate estimates from `ln(|A(t)|)` via rolling / windowed linear fits
- Automatically selects a **linear-growth plateau** using quality gates:
  - high fit quality (R² threshold)
  - sufficient growth (minimum e-folds)
  - sufficient SNR vs. early-time noise floor
  - avoids saturation (rejects windows near global maxima)
- Reports a final growth rate with cross-checks and conservative uncertainty:
  - neighbor FFT bins (m±1)
  - small k-shifts (Δk ~ π/L)
  - windowing dependence (Hann vs rectangular)
  - bin interpolation sensitivity

## Quick start

1) Put `estimate_growth_rate.py` in the repo root.

2) Create a virtual environment and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate  # (Windows: .venv\Scripts\activate)
pip install -r requirements.txt
```

3) Run on a Smilei output folder (example):

```bash
python estimate_growth_rate.py --sim ./full_sim_output --field Ex --diag 0
```

> The CLI above is the *intended interface*. If your script currently uses hardcoded parameters, see **Configuration** below to migrate toward flags or a config file.

## Inputs and assumptions

- 1D spatial domain with uniform spacing (FFT-friendly)
- Field diagnostic provides consistent `E(x)` arrays over time
- There exists a time interval where a single mode amplitude behaves approximately as:
  - `A(t) ~ exp(γ t)`  (linear instability phase)

## Configuration

The tool typically needs:

- **Simulation path** (e.g., `./full_sim_output`)
- **Field name** (e.g., `Ex`)
- **Field diagnostic index** (e.g., `0`)
- Target spectral mode choice:
  - analytic `k*` (if applicable), or
  - peak bin from spectrum, or
  - user-provided `k_target`

If your workflow uses physical units, prefer reading run parameters from the simulation metadata/namelist where possible, rather than hardcoding.

## Outputs

- Growth rate `γ` in s⁻¹ (or in code units if you choose)
- Selected linear regime time interval
- Diagnostic plots (optional): `|A(t)|`, `ln|A(t)|` with fit overlays, rolling γ estimates
- CSV/JSON summary for batch runs (recommended extension)

## Typical use cases

- Two-stream and beam instabilities (electrostatic / electromagnetic)
- Parameter scans where you need stable, automated extraction of γ
- Regression testing: detect “this run’s growth rate changed” after code/parameter changes

## Roadmap (suggested)

- [ ] Add a proper CLI with `argparse` (flags for density, drift, L, etc.)
- [ ] Add `--output` directory and write `summary.json`
- [ ] Add plotting toggles (`--no-plots`, `--save-plots`)
- [ ] Add batch mode: accept a list of run folders

## Citation

If you use this tool in a paper or report, please cite it as software:

See [`CITATION.cff`](./CITATION.cff).

## License

MIT — see [`LICENSE`](./LICENSE).
