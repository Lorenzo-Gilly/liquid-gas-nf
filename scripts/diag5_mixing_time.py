"""Diagnostic 5: Mixing time vs temperature.

For each gas-init chain at each temperature, measures the first time the chain's
U/N enters the band [liq_mean - THRESHOLD, ∞).  Calls this the mixing time.

Reads:
    experiments/diag1/T*_data.npz   (T=0.36, 0.45 from Diagnostic 1)
    experiments/diag4/T*_data.npz   (all temperatures from Diagnostic 4)
Files present in only one source are used from that source.

Output:
    experiments/diag5/mixing_time.png   — τ (K moves) vs T*, scatter + median

Usage:
    python scripts/diag5_mixing_time.py
    python scripts/diag5_mixing_time.py --threshold 0.15
"""

import argparse
import glob
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


DIAG1_DIR  = 'experiments/diag1'
DIAG4_DIR  = 'experiments/diag4'
OUT_DIR    = 'experiments/diag5'
THRESHOLD  = 0.10   # U/N units: gas chain is "mixed" when it reaches liq_mean - threshold


def load_all_npz(diag1_dir, diag4_dir):
    """Load all available NPZ files, preferring diag4 for any temperature in both."""
    files = {}
    for d in [diag1_dir, diag4_dir]:
        for path in sorted(glob.glob(os.path.join(d, 'T*_data.npz'))):
            T_str = os.path.basename(path).replace('_data.npz', '')
            files[T_str] = path   # diag4 overwrites diag1 for same T

    datasets = {}
    for T_str, path in sorted(files.items()):
        data = np.load(path)
        T = float(data['T_star'])
        datasets[T] = data
        print(f'  Loaded T*={T:.2f} from {path}')
    return datasets


def mixing_time(energies_gas, energies_liq, sample_times, threshold):
    """Return mixing time (in moves) for each gas-init chain.

    Gas U/N starts less negative than liquid U/N (higher energy).
    "Mixed" = gas chain energy drops to within threshold of liq_mean.
    For chain c: first sample_time k where energies_gas[c, k] <= liq_mean + threshold.
    Returns np.inf if never reached.
    """
    liq_mean = energies_liq[:, -20:].mean()   # mean over chains and last 20 samples
    target   = liq_mean + threshold            # gas must drop TO this level
    n_chains, n_samples = energies_gas.shape
    times = []
    for c in range(n_chains):
        reached = np.where(energies_gas[c] <= target)[0]
        if len(reached) == 0:
            times.append(np.inf)
        else:
            times.append(float(sample_times[reached[0]]))
    return np.array(times)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--diag1-dir',  default=DIAG1_DIR)
    parser.add_argument('--diag4-dir',  default=DIAG4_DIR)
    parser.add_argument('--threshold',  type=float, default=THRESHOLD,
                        help='U/N gap below liq_mean that counts as "mixed"')
    parser.add_argument('--out-dir',    default=OUT_DIR)
    args = parser.parse_args()

    print('Diagnostic 5: Mixing time vs temperature')
    print(f'  threshold = liq_mean − {args.threshold}')

    datasets = load_all_npz(args.diag1_dir, args.diag4_dir)
    if not datasets:
        raise RuntimeError(
            'No NPZ files found.  Run diag1 and/or diag4 first.')

    temperatures = sorted(datasets.keys())
    all_times    = {}   # T → (n_chains,) array of mixing times in moves
    liq_means    = {}
    n_total_dict = {}

    for T, data in datasets.items():
        E_gas  = data['energies_gas']
        E_liq  = data['energies_liq']
        times  = data['sample_times']
        tau    = mixing_time(E_gas, E_liq, times, args.threshold)
        all_times[T]    = tau
        liq_means[T]    = E_liq[:, -20:].mean()
        n_total_dict[T] = int(times[-1])

        finite = tau[np.isfinite(tau)]
        print(f'  T*={T:.2f}: liq_mean={liq_means[T]:.3f}  '
              f'target={liq_means[T]+args.threshold:.3f}  '
              f'chains mixed: {len(finite)}/{len(tau)}  '
              + (f'median τ={np.median(finite)/1000:.0f}K' if len(finite) else 'NONE'))

    # ── figure ────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    T_arr      = np.array(temperatures)
    inf_symbol = 2.1 * max(n_total_dict.values()) / 1000   # y-position for ∞ marker

    ax = axes[0]
    for T in temperatures:
        tau = all_times[T] / 1000  # convert to K-moves
        finite = tau[np.isfinite(tau)]
        inf_ct = np.sum(np.isinf(tau))

        # scatter all finite
        jitter = np.random.default_rng(0).uniform(-0.003, 0.003, len(finite))
        ax.scatter(np.full(len(finite), T) + jitter, finite,
                   s=40, color='steelblue', alpha=0.7, zorder=3)
        # mark infinite as triangles at top
        if inf_ct:
            ax.scatter(np.full(inf_ct, T), np.full(inf_ct, inf_symbol * 0.97),
                       s=60, marker='^', color='firebrick', alpha=0.8, zorder=3)

    # median line (only from temperatures with at least one finite)
    medians = []
    T_med   = []
    for T in temperatures:
        tau = all_times[T] / 1000
        finite = tau[np.isfinite(tau)]
        if len(finite):
            medians.append(np.median(finite))
            T_med.append(T)
    if T_med:
        ax.plot(T_med, medians, 'k-o', ms=5, lw=1.5, zorder=4, label='median')

    ax.axhline(inf_symbol, color='firebrick', ls=':', lw=1, alpha=0.6)
    ax.text(T_arr[-1] + 0.005, inf_symbol, '∞ (no mix)', va='center',
            fontsize=7, color='firebrick')
    ax.set_xlabel('T*'); ax.set_ylabel('Mixing time (×10³ moves)')
    ax.set_title(f'Mixing time vs T*  (ρ*={0.30:.2f}, threshold={args.threshold})')
    ax.legend(fontsize=8)

    # ── panel 2: U/N trajectories summary ─────────────────────────────────────
    ax2 = axes[1]
    cmap = plt.cm.coolwarm
    for i, T in enumerate(temperatures):
        data  = datasets[T]
        E_gas = data['energies_gas']
        times = data['sample_times'] / 1000
        color = cmap(i / max(len(temperatures) - 1, 1))
        for c in range(E_gas.shape[0]):
            ax2.plot(times, E_gas[c], color=color, alpha=0.4, lw=0.8)
        liq_m = liq_means[T]
        ax2.axhline(liq_m, color=color, ls='--', lw=1.0, alpha=0.6,
                    label=f'T*={T:.2f} liq={liq_m:.2f}')

    ax2.set_xlabel('Moves (×10³)'); ax2.set_ylabel('U/N (gas-init chains)')
    ax2.set_title('Gas-init energy evolution by temperature\n'
                  '(dashed = liquid-init mean at each T)')
    ax2.legend(fontsize=6, loc='lower right')

    fig.suptitle(
        f'Mixing time analysis  N=128  ρ*=0.30\n'
        f'"Mixed" = gas chain U/N ≥ liq_mean − {args.threshold}',
        fontsize=10)
    fig.tight_layout()

    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, 'mixing_time.png')
    fig.savefig(out_path, dpi=130, bbox_inches='tight')
    plt.close(fig)
    print(f'\nSaved: {out_path}')


if __name__ == '__main__':
    main()
