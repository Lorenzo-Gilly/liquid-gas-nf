"""Diagnostic 2: Equilibrium snapshot grid.

Loads Diagnostic 1 output and shows 2×4 scatter plots of final configurations:
  Row 1 (crimson):   4 liquid-cluster-init chains at equilibrium
  Row 2 (steel):     4 dispersed-gas-init chains at equilibrium

If both inits have converged to the same phase, the two rows look identical.
Remaining visual difference is the genuine kinetic trapping at that temperature.

Usage:
    python scripts/diag2_equilibrium_grid.py --temperature 0.36
    python scripts/diag2_equilibrium_grid.py --temperature 0.45
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


DIAG1_DIR = 'experiments/diag1'
OUT_DIR   = 'experiments/diag2'


def scatter_ax(ax, coords, N, D, L, title, color):
    xy = np.array(coords).reshape(N, D)
    ax.scatter(xy[:, 0], xy[:, 1], s=8, c=color, alpha=0.75, linewidths=0)
    ax.set_xlim(-L/2, L/2); ax.set_ylim(-L/2, L/2)
    ax.set_aspect('equal'); ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(title, fontsize=7)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--temperature', type=float, required=True,
                        help='T* value to plot (must match a diag1 output file)')
    parser.add_argument('--diag1-dir', default=DIAG1_DIR)
    parser.add_argument('--out-dir',   default=OUT_DIR)
    args = parser.parse_args()

    T = args.temperature
    npz_path = os.path.join(args.diag1_dir, f'T{T:.2f}_data.npz')
    if not os.path.exists(npz_path):
        raise FileNotFoundError(
            f'{npz_path} not found — run diag1 first:\n'
            f'  python scripts/diag1_long_convergence.py --temperatures {T}')

    data = np.load(npz_path)
    configs_liq = data['configs_final_liq']  # (n_chains, N*D)
    configs_gas = data['configs_final_gas']
    N   = int(data['N'])
    D   = int(data['D'])
    L   = float(data['L'])
    n_chains = configs_liq.shape[0]

    # final energy and OP for subtitle
    E_liq = data['energies_liq'][:, -1]   # last sample per chain
    E_gas = data['energies_gas'][:, -1]
    op_liq = data['op_liq'][:, -1]
    op_gas = data['op_gas'][:, -1]

    fig, axes = plt.subplots(2, n_chains,
                             figsize=(n_chains * 2.2, 4.8),
                             gridspec_kw={'wspace': 0.05, 'hspace': 0.30})

    for ci in range(n_chains):
        scatter_ax(axes[0, ci], configs_liq[ci], N, D, L,
                   f'liq#{ci}\nU/N={E_liq[ci]:.2f}  OP={op_liq[ci]:.2f}',
                   'crimson')
        scatter_ax(axes[1, ci], configs_gas[ci], N, D, L,
                   f'gas#{ci}\nU/N={E_gas[ci]:.2f}  OP={op_gas[ci]:.2f}',
                   'steelblue')

    axes[0, 0].set_ylabel('liquid-init', fontsize=9, color='crimson')
    axes[1, 0].set_ylabel('gas-init',    fontsize=9, color='steelblue')

    fig.suptitle(
        f'Equilibrium configurations  T*={T:.2f}  ρ*={data["rho"]:.2f}  '
        f'N={N}  ({int(data["sample_times"][-1])//1000}K moves)\n'
        f'Identical rows → converged;  Different rows → kinetic trapping',
        fontsize=9)

    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, f'T{T:.2f}_grid.png')
    fig.savefig(out_path, dpi=130, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {out_path}')


if __name__ == '__main__':
    main()
