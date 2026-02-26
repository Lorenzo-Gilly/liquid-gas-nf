"""Diagnostic 3: Order parameter histograms from equilibrated chains.

Reads Diagnostic 1 output and plots histograms of U/N and largest_cluster_fraction
using only the last LAST_FRAC of the run (default: last 40% = last 200K of 500K).

Overlays liquid-init (crimson) and gas-init (steel) to show whether:
  - Distributions overlap → fully equilibrated to the same state
  - Distributions separated → genuine phase trapping
  - Any bimodality within one init type → rare spontaneous transitions seen

Usage:
    python scripts/diag3_op_histogram.py --temperature 0.36
    python scripts/diag3_op_histogram.py --temperature 0.45 --last-frac 0.5
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde


DIAG1_DIR  = 'experiments/diag1'
OUT_DIR    = 'experiments/diag3'
LAST_FRAC  = 0.40   # use last 40% of run as production window


def ashman_d(a, b):
    """Ashman's D bimodality coefficient between two 1-D samples."""
    mu1, mu2 = np.mean(a), np.mean(b)
    s1,  s2  = np.std(a),  np.std(b)
    return abs(mu1 - mu2) / np.sqrt(0.5 * (s1**2 + s2**2) + 1e-12)


def kde_plot(ax, data_liq, data_gas, xlabel, title):
    """Histogram + KDE overlay for liq (crimson) vs gas (steel)."""
    bins = np.linspace(min(data_liq.min(), data_gas.min()),
                       max(data_liq.max(), data_gas.max()), 30)
    ax.hist(data_liq, bins=bins, density=True, alpha=0.35, color='crimson',
            label='liq-init')
    ax.hist(data_gas, bins=bins, density=True, alpha=0.35, color='steelblue',
            label='gas-init')

    for data, color in [(data_liq, 'crimson'), (data_gas, 'steelblue')]:
        if len(np.unique(data)) > 3:
            xs = np.linspace(bins[0], bins[-1], 200)
            ax.plot(xs, gaussian_kde(data)(xs), color=color, lw=2)

    d = ashman_d(data_liq, data_gas)
    ax.set_xlabel(xlabel); ax.set_ylabel('Density')
    ax.set_title(f'{title}  (Ashman D={d:.2f})')
    ax.legend(fontsize=8)
    return d


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--temperature', type=float, required=True)
    parser.add_argument('--last-frac',   type=float, default=LAST_FRAC,
                        help='Fraction of run to use as production window')
    parser.add_argument('--diag1-dir',   default=DIAG1_DIR)
    parser.add_argument('--out-dir',     default=OUT_DIR)
    args = parser.parse_args()

    T = args.temperature
    npz_path = os.path.join(args.diag1_dir, f'T{T:.2f}_data.npz')
    if not os.path.exists(npz_path):
        raise FileNotFoundError(
            f'{npz_path} not found — run diag1 first:\n'
            f'  python scripts/diag1_long_convergence.py --temperatures {T}')

    data = np.load(npz_path)
    n_samples = data['energies_liq'].shape[1]
    cutoff = int(n_samples * (1 - args.last_frac))
    print(f'T*={T:.2f}: using samples {cutoff}–{n_samples} '
          f'({args.last_frac*100:.0f}% of run = '
          f'{int(data["sample_times"][-1] * args.last_frac)//1000}K moves)')

    # flatten over chains: (n_chains, n_prod_samples) → (n_chains * n_prod,)
    E_liq  = data['energies_liq'][:, cutoff:].ravel()
    E_gas  = data['energies_gas'][:, cutoff:].ravel()
    op_liq = data['op_liq'][:,    cutoff:].ravel()
    op_gas = data['op_gas'][:,    cutoff:].ravel()

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    d_e  = kde_plot(axes[0], E_liq,  E_gas,  'U/N',
                    f'Energy  T*={T:.2f}')
    d_op = kde_plot(axes[1], op_liq, op_gas, 'Largest cluster fraction',
                    f'Cluster OP  T*={T:.2f}')

    interpretation = []
    if d_e > 2.0 or d_op > 2.0:
        interpretation.append('STRONG separation (D>2) → phases genuinely trapped')
    elif d_e > 1.0 or d_op > 1.0:
        interpretation.append('MODERATE separation → partial trapping')
    else:
        interpretation.append('LOW separation (D<1) → distributions overlapping')

    fig.suptitle(
        f'N={int(data["N"])}  ρ*={data["rho"]:.2f}  T*={T:.2f}  '
        f'production={args.last_frac*100:.0f}% of run\n'
        + '  |  '.join(interpretation),
        fontsize=9)
    fig.tight_layout()

    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, f'T{T:.2f}_histograms.png')
    fig.savefig(out_path, dpi=130, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {out_path}')
    print(f'  Ashman D (energy)={d_e:.2f}  (cluster OP)={d_op:.2f}')


if __name__ == '__main__':
    main()
