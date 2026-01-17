#!/usr/bin/env python3
"""
Check_growth_FFT.py
-------------------

Estimate the two‑stream growth rate γ from the time evolution of a **single
Fourier mode** (the analytic k*), across multiple time windows. For each
window, fit ln|E_k*|(t) with a line to get γ and print its R².

This version **does not** compute per‑cell fits or average R² across windows.
It only performs the FFT‑based method at the **analytic k***.

Configuration
-------------
N_T          = 1200  # number of temporal samples
FIELD_NAME   = "Ex"  # which field component to analyse (e.g. Ex or Ez)
V0           = 1.5e5 # drift speed magnitude [m/s] for each beam (±V0)
omega_ref    = 1.78e9# reference angular frequency [rad/s] (electron ω_p for total density)
L_ref        = c/ω_ref; domain length is L_phys = 0.16 * L_ref
"""
import math
from math import pi, sqrt
import numpy as np
import happi

# ------------------------ user parameters ------------------------ #
c    = 2.99792458e8           # [m/s]       speed of light
e    = 1.602176634e-19        # [C]         elementary charge
eps0 = 8.854187817e-12        # [F/m]       vacuum permittivity
me   = 9.10938356e-31         # [kg]        electron mass
n_density = 1e15              # [m⁻³]        density of plasma in MKS
N_T        = 1200             # number of temporal samples
FIELD_NAME = "Ex"             # Ex or Ez
V0         = c * 0.0005       # ~150 km/s
omega_ref  = math.sqrt(n_density * e**2 / (eps0 * me)) # [rad/s] (electron plasma freq for total density)
omega_beam = math.sqrt(n_density * e**2 / (2 * eps0 * me)) # [rad/s] (per-beam electron plasma freq)
L_ref      = c / omega_ref
k_star_phys = omega_beam / (math.sqrt(2.0) * V0)  
# account for actual physical box length (0.16 × L_ref)
L_phys     = 0.16 * L_ref
# ----------------------------------------------------------------- #

# ---------- helpers ---------- #
def collapse_to_1d(field_data):
    """Collapse Smilei's patch list into a 1‑D numpy array."""
    if isinstance(field_data, list):
        arr = np.concatenate([np.array(p).squeeze() for p in field_data])
    else:
        arr = np.array(field_data).squeeze()
    return arr

def normalize_label(low, high):
    """Normalize window range as tuple of floats rounded to 2 decimals."""
    return (round(low, 2), round(high, 2))

def format_label(low, high):
    """Format window label as string with two decimals."""
    return f"{low:.2f}–{high:.2f}"

# --- CONTINUOUS-k PROJECTION HELPERS ---
def hann_window(N: int) -> np.ndarray:
    n = np.arange(N)
    return 0.5 * (1.0 - np.cos(2.0*np.pi*n/N))

def contDFT_amplitude(arr: np.ndarray, k_phys: float, L_phys: float, window: str = "hann") -> complex:
    """
    Continuous-k projection at physical wavenumber k_phys (rad/m).
    x_j = j * (L_phys/N). Window: 'hann' or 'rect'.
    Returns an (unnormalized) complex amplitude; the constant factor
    cancels when fitting ln|E|(t) for a growth rate.
    """
    a = np.asarray(arr).ravel()
    N = a.size
    dx = L_phys / N
    x = dx*np.arange(N)
    w = hann_window(N) if window == "hann" else np.ones(N)
    return np.sum(w * a * np.exp(-1j*k_phys*x)) * dx

# ---------- open simulation & gather timesteps ---------- #
S = happi.Open("./full_sim_output")                                     # adjust if your output dir differs
field_diag = S.Field(diagNumber=0, field=FIELD_NAME)
all_steps  = np.array(field_diag.getAvailableTimesteps(), dtype=float)
if all_steps.size == 0:
    raise RuntimeError(f"No timesteps found for field '{FIELD_NAME}'")

# choose N_T evenly spaced steps
idx_time   = np.linspace(0, all_steps.size - 1, N_T, dtype=int)
sel_steps  = all_steps[idx_time]

# build |FFT| vs time for all k
spec_time = []
space_time = []
for step in sel_steps:
    arr = collapse_to_1d(field_diag.getData(timestep=step))
    space_time.append(arr)
    spec = np.abs(np.fft.rfft(arr))                           # magnitude of Fourier modes
    spec_time.append(spec)

spec_time = np.array(spec_time)                               # shape (N_T, N_modes)
space_time = np.array(space_time, dtype=object)
N_modes   = spec_time.shape[1]

# physical times [s]
dt_code   = S.namelist.Main.timestep
omega_si  = S.namelist.Main.reference_angular_frequency_SI
Times_s   = sel_steps * (dt_code / omega_si)

# ---------- analytic k* and corresponding FFT index ---------- #
# For symmetric cold two‑beam, with ω_ref = ω_p(total) and per‑beam ω_p = ω_ref/√2:
#    k* = ω_p(beam) / (√2 V0) = (ω_ref / √2) / (√2 V0) = ω_ref / (2 V0)
# If you prefer the ω_p/√2 form (per‑beam), set k_star_phys = omega_ref/(sqrt(2)*V0)
# Choose the expression you want below.

# Map to FFT bin index: k_m = 2π m / L_phys  →  m* = k* L_phys / (2π)
k_star_frac = k_star_phys * L_phys / (2.0 * pi)
k_star_idx  = int(round(k_star_frac))

if k_star_idx >= N_modes:
    raise RuntimeError(
        f"Analytic k* index {k_star_idx} is out of FFT range (N_modes={N_modes}).\n"
        f"Try reducing N_T, checking L_phys, or verifying ω_ref and V0.")

k_bin_phys  = 2.0 * pi * k_star_idx / L_phys
lambda_star = 2.0 * pi / k_bin_phys

print("\nAnalytic k* details")
print(f"k* (phys, target)     = {k_star_phys:.3e} rad/m")
print(f"k* (FFT bin, used)    = {k_bin_phys:.3e} rad/m  (index = {k_star_idx}, fractional = {k_star_frac:.2f})")
print(f"λ* (from used bin)    = {lambda_star:.3e} m")

# Primary time series: continuous‑k projection at analytic k* (Hann window)
amp_main = [contDFT_amplitude(collapse_to_1d(a), k_star_phys, L_phys, window="hann")
            for a in space_time]
amp_main = np.abs(np.asarray(amp_main, dtype=complex))
mask  = amp_main > 0
if mask.sum() < 8:
    raise RuntimeError("Not enough positive-amplitude samples for k* to fit a growth rate.")

t_all  = Times_s[mask]
ln_all = np.log(amp_main[mask])

# ---------- window generation ---------- #
window_sizes = [0.02, 0.03, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 1.00]

candidate_windows = []
seen_windows = set()

for size in window_sizes:
    start = 0.0
    while start + size <= 1.0 + 1e-9:
        low = round(start, 2)
        high = round(start + size, 2)
        key = normalize_label(low, high)
        if key not in seen_windows:
            label = format_label(key[0], key[1])
            candidate_windows.append((key[0], key[1], label))
            seen_windows.add(key)
        start += 0.01  # slide by 0.01 to allow overlapping windows

# ---------- evaluate γ and R² for each window using the analytic k* ---------- #
results = []
for frac_lo, frac_hi, label in candidate_windows:
    n = t_all.size
    i0 = int(frac_lo * n)
    i1 = max(i0 + 3, int(frac_hi * n))
    if i1 > n or i1 <= i0 + 2:
        continue
    t_sub  = t_all[i0:i1]
    ln_sub = ln_all[i0:i1]
    coeffs = np.polyfit(t_sub, ln_sub, 1)
    slope, intercept = coeffs
    ln_fit = slope * t_sub + intercept
    ss_res = np.sum((ln_sub - ln_fit)**2)
    ss_tot = np.sum((ln_sub - ln_sub.mean())**2)
    R2     = 1.0 - ss_res/ss_tot if ss_tot > 0 else np.nan
    results.append((label, frac_lo, frac_hi, slope, R2, t_sub.size))

# sort by window start first, then width (optional: or by R² descending)
results_sorted = sorted(results, key=lambda r: (r[1], r[2]-r[1]))

print("\nProjection @ analytic k*: growth‑rate estimates by time window")
print(f"{'window':<14} {'γ [s⁻¹]':>12} {'R²':>6} {'N':>4}")
for label, lo, hi, gamma, R2, npts in results_sorted:
    gstr = f"{gamma: .3e}" if np.isfinite(gamma) else "NaN"
    rstr = f"{R2: .2f}"    if np.isfinite(R2)    else "NaN"
    print(f"{label:<14} {gstr:>12} {rstr:>6} {npts:4d}")


# Also print a ranking by R² (top 12) for convenience
ranked = sorted(results, key=lambda r: (-np.nan_to_num(r[4], nan=-1.0), r[1]))
print("\nTop windows by R² (analytic k*)")
for i,(label, lo, hi, gamma, R2, npts) in enumerate(ranked[:100], 1):
    print(f"{i:2d}. {label:<14} γ={gamma:.3e} s⁻¹  R²={R2:.2f}  N={npts}")

# ---------- Robust linear-stage estimator (rolling fit) ---------- #
# Parameters (chosen conservatively for paper-level reporting)
roll_frac   = 0.12   # rolling window size as a fraction of available samples
min_points  = 60     # minimum points per rolling fit (guards against tiny windows)
R2_min      = 0.98   # goodness-of-fit gate
efold_min   = 2.0    # require at least this many e-folds in ln|E| across window
snr_min     = 10.0   # median(|E|) in window must exceed snr_min * noise floor
sat_top_max = 0.30   # top amplitude within window must be below this fraction of global max

n_total = t_all.size
if n_total < max(min_points, 10):
    raise RuntimeError("Not enough time samples for rolling estimator.")

# Noise floor: median over the first 5% of samples (>= 5 samples)
head_n     = max(5, int(0.05 * n_total))
A_noise    = np.median(np.exp(ln_all[:head_n]))
A_global_max = np.max(np.exp(ln_all))

# Rolling window length in samples
win_len = max(int(round(roll_frac * n_total)), min_points)
win_len = min(win_len, n_total - 2)  # keep at least 2 points outside
half    = win_len // 2

print("\n[Rolling estimator] configuration:")
print(f"  window length           = {win_len} samples (~{win_len/n_total:.2%} of selected timeline)")
print(f"  gates: R² \u2265 {R2_min}, e-folds \u2265 {efold_min}, SNR \u2265 {snr_min}, top/amax \u2264 {sat_top_max}")
print(f"  noise floor A_noise     = {A_noise:.3e} (median of first {head_n} samples)")
print(f"  global max amplitude    = {A_global_max:.3e}")

roll_records = []  # (i0, i1, t_center, gamma, R2, SE_gamma, dln, snr, top_frac)

eps = 1e-30
for center in range(half, n_total - half):
    i0 = center - half
    i1 = i0 + win_len
    t_sub  = t_all[i0:i1]
    y_sub  = ln_all[i0:i1]

    # Fit y = a t + b
    a, b   = np.polyfit(t_sub, y_sub, 1)
    y_fit  = a * t_sub + b
    res    = y_sub - y_fit
    ss_res = float(np.sum(res**2))
    ss_tot = float(np.sum((y_sub - y_sub.mean())**2))
    R2     = 1.0 - ss_res/ss_tot if ss_tot > 0 else np.nan

    # Standard error of slope (OLS)
    denom = float(np.sum((t_sub - t_sub.mean())**2))
    sigma2 = ss_res / max(len(t_sub) - 2, 1)
    SE_a = np.sqrt(sigma2 / max(denom, eps))

    # Window diagnostics for gating
    A_med    = float(np.median(np.exp(y_sub)))
    snr      = A_med / max(A_noise, eps)
    dln      = float(np.max(y_sub) - np.min(y_sub))
    top_frac = float(np.max(np.exp(y_sub)) / max(A_global_max, eps))

    roll_records.append((i0, i1, t_sub[len(t_sub)//2], a, R2, SE_a, dln, snr, top_frac))

roll_records = np.array(roll_records, dtype=float)

# Apply gates to keep plausible linear-stage windows
passed = (
    (roll_records[:,4] >= R2_min) &                # R²
    (roll_records[:,6] >= efold_min) &             # e-fold coverage
    (roll_records[:,7] >= snr_min) &               # SNR above noise
    (roll_records[:,8] <= sat_top_max)             # below saturation
)

n_pass = int(np.sum(passed))
print(f"[Rolling estimator] windows evaluated = {roll_records.shape[0]}")
print(f"[Rolling estimator] windows passing gates = {n_pass}")

if n_pass < 5:
    print("[Rolling estimator] WARNING: Few windows passed gates; relaxing e-folds to 1.5 and retrying…")
    passed = (
        (roll_records[:,4] >= R2_min) &
        (roll_records[:,6] >= 1.5) &
        (roll_records[:,7] >= snr_min) &
        (roll_records[:,8] <= sat_top_max)
    )

# Identify plateau: longest contiguous run of passed windows with low CV of gamma
plateau_best = None  # (start_idx, end_idx, cv, length)

# Find contiguous index ranges of 'passed'
idxs = np.where(passed)[0]
segments = []
if idxs.size > 0:
    start = idxs[0]
    prev  = idxs[0]
    for k in idxs[1:]:
        if k == prev + 1:
            prev = k
        else:
            segments.append((start, prev))
            start = k
            prev  = k
    segments.append((start, prev))

cv_threshold = 0.15
min_segment  = 5

for s,e in segments:
    seg = roll_records[s:e+1]
    gammas = seg[:,3]
    if gammas.size < min_segment:
        continue
    mean_g = float(np.mean(gammas))
    std_g  = float(np.std(gammas, ddof=1)) if gammas.size > 1 else 0.0
    cv     = abs(std_g / mean_g) if mean_g != 0 else np.inf
    # Prefer segments with cv <= threshold; among them pick the longest
    if cv <= cv_threshold:
        if plateau_best is None or (e - s) > (plateau_best[1] - plateau_best[0]):
            plateau_best = (s,e,cv,(e-s+1))

# Fallback: if none meet CV threshold, pick the segment with **lowest** CV
if plateau_best is None and segments:
    best = None
    best_cv = np.inf
    for s,e in segments:
        seg = roll_records[s:e+1]
        gammas = seg[:,3]
        if gammas.size < min_segment:
            continue
        mean_g = float(np.mean(gammas))
        std_g  = float(np.std(gammas, ddof=1)) if gammas.size > 1 else 0.0
        cv     = abs(std_g / mean_g) if mean_g != 0 else np.inf
        if cv < best_cv:
            best_cv = cv
            best = (s,e,cv,(e-s+1))
    plateau_best = best

if plateau_best is None:
    print("[Rolling estimator] ERROR: Could not identify a linear-stage plateau. Consider adjusting gates.")
else:
    s,e,cv,seglen = plateau_best
    seg = roll_records[int(s):int(e)+1]
    # Aggregate plateau information
    i0_min = int(np.min(seg[:,0]))
    i1_max = int(np.max(seg[:,1]))
    t_low, t_high = t_all[i0_min], t_all[i1_max-1]
    frac_low = i0_min / n_total
    frac_high= i1_max / n_total

    gammas = seg[:,3]
    R2s    = seg[:,4]
    SEs    = seg[:,5]
    weights = R2s / (SEs**2 + 1e-30)
    gamma_weighted = float(np.sum(weights * gammas) / np.sum(weights))
    SE_weighted    = float(np.sqrt(1.0 / np.sum(weights)))
    spread_gamma   = float(np.std(gammas, ddof=1)) if gammas.size > 1 else 0.0

    print("\n[Rolling estimator] Selected plateau:")
    print(f"  center-window count     = {int(seglen)} (contiguous)")
    print(f"  time range              = [{t_low:.3e}, {t_high:.3e}] s  (frac {frac_low:.2f}–{frac_high:.2f} of selected interval)")
    print(f"  gamma mean (weighted)   = {gamma_weighted:.3e} s^-1")
    print(f"  statistical SE          = {SE_weighted:.3e} s^-1")
    print(f"  window-to-window spread = {spread_gamma:.3e} s^-1  (cv ~ {cv:.2f})")

    # Refit on the full plateau span for reference
    t_seg  = t_all[i0_min:i1_max]
    y_seg  = ln_all[i0_min:i1_max]
    a_all, b_all = np.polyfit(t_seg, y_seg, 1)

    # Energy (|E|^2) cross-check on the same span (expect slope ~ 2*gamma)
    yE_seg = 2.0 * y_seg
    aE, bE = np.polyfit(t_seg, yE_seg, 1)
    gamma_energy = 0.5 * aE

    # k-bin sensitivity: redo fit at m*±1 on the same time span
    def fit_bin(bin_idx):
        if bin_idx < 0 or bin_idx >= N_modes:
            return np.nan
        amp = spec_time[:, int(bin_idx)]
        mask = (amp > 0)
        # align with plateau time-span indices
        mask_seg = np.zeros_like(mask, dtype=bool)
        mask_seg[i0_min:i1_max] = True
        use = mask & mask_seg
        if np.count_nonzero(use) < 10:
            return np.nan
        t_b  = Times_s[use]
        y_b  = np.log(amp[use])
        aa, bb = np.polyfit(t_b, y_b, 1)
        return float(aa)

    gamma_m  = float(a_all)
    gamma_m1 = fit_bin(k_star_idx - 1)
    gamma_p1 = fit_bin(k_star_idx + 1)

    k_list   = [v for v in [gamma_m, gamma_m1, gamma_p1] if np.isfinite(v)]
    if len(k_list) >= 2:
        k_spread = float(np.std(k_list, ddof=1))
    else:
        k_spread = np.nan

    print("\n[Cross-checks]")
    print(f"  plateau refit (same span)    gamma = {gamma_m:.3e} s^-1")
    print(f"  energy-slope check           gamma_E = {gamma_energy:.3e} s^-1 (expected ~gamma)")

    if np.isfinite(gamma_m1):
        print(f"  neighbor bin m*-1 slope      = {gamma_m1:.3e} s^-1")
    else:
        print("  neighbor bin m*-1 slope      = NaN (insufficient data)")
    if np.isfinite(gamma_p1):
        print(f"  neighbor bin m*+1 slope      = {gamma_p1:.3e} s^-1")
    else:
        print("  neighbor bin m*+1 slope      = NaN (insufficient data)")

    # --- Systematics components ---
    # (1) k-sensitivity: continuous-k at k* ± π/L_phys on the same time span
    dk = np.pi / L_phys
    def fit_cont_k(kphys: float) -> float:
        amps = [contDFT_amplitude(collapse_to_1d(a), kphys, L_phys, window="hann")
                for a in space_time]
        amps = np.abs(np.asarray(amps, dtype=complex))
        use = (amps > 0)
        mask_seg = np.zeros_like(use, dtype=bool)
        mask_seg[i0_min:i1_max] = True
        sel = use & mask_seg
        tt = Times_s[sel]
        yy = np.log(amps[sel])
        aa, _ = np.polyfit(tt, yy, 1)
        return float(aa)

    try:
        gamma_km = fit_cont_k(k_star_phys - dk)
        gamma_kp = fit_cont_k(k_star_phys + dk)
        k_sens_spread = float(np.std([gamma_km, gamma_m, gamma_kp], ddof=1))
    except Exception:
        gamma_km = np.nan
        gamma_kp = np.nan
        k_sens_spread = np.nan

    # (2) Interpolation bias in k-space: compare 3-bin vs 5-bin quadratic around m0
    def gamma_interp_k(nbins: int) -> float:
        half = nbins // 2
        ln_series = []
        for it in range(spec_time.shape[0]):
            m0 = int(np.floor(k_star_frac))
            idxs = np.arange(m0 - half, m0 + half + 1)
            idxs = idxs[(idxs >= 0) & (idxs < N_modes)]
            if idxs.size < 3:
                continue
            k_vec = 2.0 * np.pi * idxs / L_phys
            y_vec = np.log(np.maximum(spec_time[it, idxs], 1e-300))
            c2, c1, c0 = np.polyfit(k_vec, y_vec, 2)
            ln_at_k = c2*k_star_phys**2 + c1*k_star_phys + c0
            ln_series.append(ln_at_k)
        ln_series = np.asarray(ln_series)
        tt = Times_s[:ln_series.size]
        aa, _ = np.polyfit(tt, ln_series, 1)
        return float(aa)

    try:
        gamma_interp3 = gamma_interp_k(3)
        gamma_interp5 = gamma_interp_k(5)
        interp_bias = float(abs(gamma_interp3 - gamma_interp5))
    except Exception:
        gamma_interp3 = np.nan
        gamma_interp5 = np.nan
        interp_bias = np.nan

    # (3) Windowing dependence: contDFT with rectangular vs Hann on the same plateau span
    try:
        amps_rect = [contDFT_amplitude(collapse_to_1d(a), k_star_phys, L_phys, window="rect")
                     for a in space_time]
        amps_rect = np.abs(np.asarray(amps_rect, dtype=complex))
        use = (amps_rect > 0)
        mask_seg = np.zeros_like(use, dtype=bool)
        mask_seg[i0_min:i1_max] = True
        sel = use & mask_seg
        tt_r = Times_s[sel]
        yy_r = np.log(amps_rect[sel])
        a_rect, _ = np.polyfit(tt_r, yy_r, 1)
        window_dep = float(abs(a_rect - gamma_m))
    except Exception:
        window_dep = np.nan

    # Combine systematics conservatively
    sys_candidates = [spread_gamma]
    if np.isfinite(k_sens_spread): sys_candidates.append(k_sens_spread)
    if np.isfinite(interp_bias):   sys_candidates.append(interp_bias)
    if np.isfinite(window_dep):    sys_candidates.append(window_dep)
    sys_component = float(np.max(sys_candidates)) if sys_candidates else float('nan')

    # Report details
    if np.isfinite(gamma_km) and np.isfinite(gamma_kp):
        print(f"  k-sensitivity (k*±π/L):     {gamma_km:.3e}, {gamma_m:.3e}, {gamma_kp:.3e} s^-1  → σ ≈ {k_sens_spread:.3e}")
    print(f"  interp bias (3-bin vs 5-bin): Δγ ≈ {interp_bias:.3e} s^-1")
    print(f"  windowing dependence        : Δγ ≈ {window_dep:.3e} s^-1 (rect vs Hann)")
    print(f"  plateau spread              : σ ≈ {spread_gamma:.3e} s^-1")

    print("\n[Result]")
    print(f"  gamma = {gamma_weighted:.3e} s^-1  ± {SE_weighted:.3e} (stat)  ± {sys_component:.3e} (sys)")
    print("  (sys = max{plateau spread, k-sensitivity, k-interp bias, windowing dependence})")

# ---------- Sanity check: time axis & spacing, to see it matches the simulation ---------- #
if Times_s.size >= 2:
    dt_sample = Times_s[1] - Times_s[0]
    dt_sim    = S.namelist.Main.timestep / S.namelist.Main.reference_angular_frequency_SI
    print("\nTime‑axis sanity check")
    print(f"First sampled time       = {Times_s[0]:.3e}  s")
    print(f"Last  sampled time       = {Times_s[-1]:.3e}  s")
    print(f"Δt between sampled frames= {dt_sample:.3e}  s")
    print(f"Simulation timestep Δt   = {dt_sim:.3e}  s")
    print(f"(sample spacing) / (Δt)  = {dt_sample / dt_sim:.1f}  steps")
