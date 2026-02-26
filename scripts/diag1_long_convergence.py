"""Diagnostic 1: Convergence test — do gas-init chains ever coalesce?

At T*=0.36 and T*=0.45, runs 4 liquid-init + 4 gas-init chains for 500K moves.
Records U/N and largest_cluster_fraction every CHUNK moves.
Saves snapshots every SNAP_EVERY moves.

Output per temperature T:
    experiments/diag1/T{T}_data.npz       — time series + final configs
    experiments/diag1/T{T}_timeseries.png — U/N time series
    experiments/diag1/T{T}_filmstrip.png  — gas-init snapshot filmstrip

Usage:
    python scripts/diag1_long_convergence.py
    python scripts/diag1_long_convergence.py --n-total 200000 --temperatures 0.36
"""

import argparse
import functools
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from physics import lj_energy, run_mcmc
from diagnostics import (make_liquid_init, make_gas_init,
                         calibrate_step_size, largest_cluster_fraction)

# ── defaults ──────────────────────────────────────────────────────────────────
N          = 128
D          = 2
RHO        = 0.30
N_CHAINS   = 4
CHUNK      = 5_000
N_TOTAL    = 500_000
SNAP_EVERY = 50_000
R_CUT      = 1.6733   # cluster OP cutoff from Stage A g(r); NOT the LJ cutoff
T_DEFAULTS = [0.36, 0.45]
OUT_DIR    = 'experiments/diag1'


# ── helpers ───────────────────────────────────────────────────────────────────

def make_energy_fn(N, D, L):
    return jax.jit(functools.partial(
        lj_energy, n_particles=N, dimensions=D, box_length=L, use_lrc=False))


def compute_op_batch(configs, N, D, L, r_cut):
    """Cluster fraction for each config in batch (N_chains, N*D)."""
    return np.array([
        largest_cluster_fraction(np.array(configs[i]), N, D, L, r_cut)
        for i in range(configs.shape[0])
    ])


def scatter_ax(ax, coords, N, D, L, title, color):
    xy = np.array(coords).reshape(N, D)
    ax.scatter(xy[:, 0], xy[:, 1], s=6, c=color, alpha=0.7, linewidths=0)
    ax.set_xlim(-L/2, L/2); ax.set_ylim(-L/2, L/2)
    ax.set_aspect('equal'); ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(title, fontsize=6)


# ── per-temperature run ────────────────────────────────────────────────────────

def run_temperature(T_star, args, key):
    L = float(np.sqrt(N / RHO))
    beta = 1.0 / T_star
    r_cut = args.rcut
    n_chunks = args.n_total // CHUNK
    n_snaps = args.n_total // SNAP_EVERY

    print(f'\n── T*={T_star:.2f}  ρ*={RHO:.2f}  L={L:.3f}  '
          f'{n_chunks} chunks × {CHUNK} moves  {n_snaps} snapshots ──')

    energy_fn = make_energy_fn(N, D, L)
    # warm up compilation
    _dummy = jnp.zeros((2, N * D))
    _ = energy_fn(_dummy)

    # calibrate step size at this T
    key, k1, k2 = jax.random.split(key, 3)
    calib_inits = jnp.stack([make_liquid_init(N, D, L, rng_key=k)
                              for k in jax.random.split(k1, 4)])
    step_size, rates = calibrate_step_size(calib_inits, energy_fn, beta, N, D, L, k2)
    print(f'  step_size={step_size:.4f}  acc≈{rates[step_size]:.3f}')

    # initialise configs
    key, kl, kg = jax.random.split(key, 3)
    liq_configs = jnp.stack([make_liquid_init(N, D, L, rng_key=k)
                              for k in jax.random.split(kl, N_CHAINS)])
    gas_configs = jnp.stack([make_gas_init(N, D, L, rng_key=k)
                              for k in jax.random.split(kg, N_CHAINS)])

    # storage
    sample_times   = []
    energies_liq   = []   # list of (N_CHAINS,) arrays
    energies_gas   = []
    op_liq         = []
    op_gas         = []
    snapshots_liq  = []   # collected at snap boundaries
    snapshots_gas  = []

    print('  Running ', end='', flush=True)
    for ci in range(n_chunks):
        key, kl_run, kg_run = jax.random.split(key, 3)

        liq_configs, liq_e = run_mcmc(
            liq_configs, energy_fn, beta, CHUNK, kl_run, step_size, L, N, D)
        gas_configs, gas_e = run_mcmc(
            gas_configs, energy_fn, beta, CHUNK, kg_run, step_size, L, N, D)

        moves_done = (ci + 1) * CHUNK
        sample_times.append(moves_done)
        energies_liq.append(np.array(liq_e) / N)
        energies_gas.append(np.array(gas_e) / N)
        op_liq.append(compute_op_batch(liq_configs, N, D, L, r_cut))
        op_gas.append(compute_op_batch(gas_configs, N, D, L, r_cut))

        if moves_done % SNAP_EVERY == 0:
            snapshots_liq.append(np.array(liq_configs))
            snapshots_gas.append(np.array(gas_configs))
            print(f'{moves_done//1000}k', end=' ', flush=True)

    print('done.')

    # shape: (n_samples, N_CHAINS) → transpose to (N_CHAINS, n_samples)
    energies_liq  = np.array(energies_liq).T
    energies_gas  = np.array(energies_gas).T
    op_liq        = np.array(op_liq).T
    op_gas        = np.array(op_gas).T
    sample_times  = np.array(sample_times)
    snapshots_liq = np.array(snapshots_liq).transpose(1, 0, 2)  # (chains, snaps, N*D)
    snapshots_gas = np.array(snapshots_gas).transpose(1, 0, 2)

    print(f'  Final U/N: liq={energies_liq[:, -1].mean():.3f}'
          f'  gas={energies_gas[:, -1].mean():.3f}'
          f'  ΔU/N={abs(energies_liq[:,-1].mean() - energies_gas[:,-1].mean()):.3f}')

    # ── save NPZ ──────────────────────────────────────────────────────────────
    os.makedirs(args.out_dir, exist_ok=True)
    npz_path = os.path.join(args.out_dir, f'T{T_star:.2f}_data.npz')
    np.savez(npz_path,
             sample_times      = sample_times,
             energies_liq      = energies_liq,
             energies_gas      = energies_gas,
             op_liq            = op_liq,
             op_gas            = op_gas,
             snapshots_liq     = snapshots_liq,
             snapshots_gas     = snapshots_gas,
             configs_final_liq = np.array(liq_configs),
             configs_final_gas = np.array(gas_configs),
             T_star=T_star, rho=RHO, N=N, D=D, L=L, r_cut=r_cut)
    print(f'  Saved: {npz_path}')

    # ── plot time series ───────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    x = sample_times / 1000  # k-moves

    for label, E, color in [('liq-init', energies_liq, 'crimson'),
                             ('gas-init', energies_gas, 'steelblue')]:
        for ci in range(N_CHAINS):
            axes[0].plot(x, E[ci], color=color, alpha=0.6,
                         lw=1.2, label=label if ci == 0 else None)
        for ci in range(N_CHAINS):
            axes[1].plot(x, op_liq[ci] if label == 'liq-init' else op_gas[ci],
                         color=color, alpha=0.6, lw=1.2,
                         label=label if ci == 0 else None)

    liq_mean_e = energies_liq[:, -20:].mean()
    axes[0].axhline(liq_mean_e, color='crimson', ls='--', lw=1.5, alpha=0.8,
                    label=f'liq mean={liq_mean_e:.3f}')
    axes[0].axhline(liq_mean_e - 0.10, color='grey', ls=':', lw=1,
                    label='liq mean ± 0.10')
    axes[0].axhline(liq_mean_e + 0.10, color='grey', ls=':', lw=1)
    axes[0].set_xlabel('Moves (×10³)'); axes[0].set_ylabel('U/N')
    axes[0].set_title(f'T*={T_star:.2f}  ρ*={RHO:.2f}  — Energy per particle')
    axes[0].legend(fontsize=7)

    axes[1].set_xlabel('Moves (×10³)'); axes[1].set_ylabel('Largest cluster fraction')
    axes[1].set_title(f'T*={T_star:.2f}  — Cluster order parameter')
    axes[1].legend(fontsize=7)

    fig.tight_layout()
    ts_path = os.path.join(args.out_dir, f'T{T_star:.2f}_timeseries.png')
    fig.savefig(ts_path, dpi=130, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved: {ts_path}')

    # ── filmstrip: gas-init snapshots ─────────────────────────────────────────
    n_frames = snapshots_gas.shape[1]
    fig, axes = plt.subplots(N_CHAINS, n_frames,
                             figsize=(n_frames * 1.8, N_CHAINS * 1.8),
                             gridspec_kw={'wspace': 0.05, 'hspace': 0.25})
    if N_CHAINS == 1:
        axes = axes[np.newaxis, :]
    for ci in range(N_CHAINS):
        for fi in range(n_frames):
            scatter_ax(axes[ci, fi], snapshots_gas[ci, fi], N, D, L,
                       f'{(fi+1)*SNAP_EVERY//1000}k',
                       'steelblue')
        axes[ci, 0].set_ylabel(f'chain {ci}', fontsize=7)

    fig.suptitle(f'Gas-init filmstrip  T*={T_star:.2f}  ρ*={RHO:.2f}  '
                 f'(snapshot every {SNAP_EVERY//1000}K moves)', y=1.01, fontsize=9)
    fs_path = os.path.join(args.out_dir, f'T{T_star:.2f}_filmstrip.png')
    fig.savefig(fs_path, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved: {fs_path}')

    return key


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--temperatures', nargs='+', type=float, default=T_DEFAULTS)
    parser.add_argument('--n-total',  type=int,   default=N_TOTAL)
    parser.add_argument('--rcut',     type=float, default=R_CUT)
    parser.add_argument('--seed',     type=int,   default=42)
    parser.add_argument('--out-dir',  default=OUT_DIR)
    args = parser.parse_args()

    key = jax.random.PRNGKey(args.seed)
    print(f'Diagnostic 1: Convergence test')
    print(f'  N={N}  ρ*={RHO}  T*={args.temperatures}')
    print(f'  {N_CHAINS} chains × {args.n_total//1000}K moves  '
          f'chunks={CHUNK}  r_cut={args.rcut}')

    for T_star in args.temperatures:
        key, subkey = jax.random.split(key)
        run_temperature(T_star, args, subkey)

    print('\nDiagnostic 1 complete.')
    print('Next: python scripts/diag2_equilibrium_grid.py --temperature 0.36')
    print('      python scripts/diag3_op_histogram.py --temperature 0.36')


if __name__ == '__main__':
    main()
