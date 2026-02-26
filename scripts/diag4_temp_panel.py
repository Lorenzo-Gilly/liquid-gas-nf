"""Diagnostic 4: Temperature panel with full equilibration.

Polished version of visual_coex_check.py with 500K equilibration moves.
Runs 4 liquid-init + 4 gas-init chains at each of 6 temperatures, records
U/N time series, saves scatter-plot grid.

Output:
    experiments/diag4/T{T}_data.npz      — time series + final configs
    experiments/diag4/scatter_grid.png   — main presentation figure

Usage:
    python scripts/diag4_temp_panel.py
    python scripts/diag4_temp_panel.py --n-total 200000  # faster test
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
R_CUT      = 1.6733
T_SCAN     = [0.33, 0.36, 0.39, 0.42, 0.45, 0.50]
OUT_DIR    = 'experiments/diag4'


# ── helpers ───────────────────────────────────────────────────────────────────

def make_energy_fn(N, D, L):
    return jax.jit(functools.partial(
        lj_energy, n_particles=N, dimensions=D, box_length=L, use_lrc=False))


def compute_op_batch(configs, N, D, L, r_cut):
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

def run_temperature(T_star, args, key, energy_fn, step_size, L):
    beta     = 1.0 / T_star
    r_cut    = args.rcut
    n_chunks = args.n_total // CHUNK

    print(f'  T*={T_star:.2f}', end='  ', flush=True)

    key, kl, kg = jax.random.split(key, 3)
    liq_configs = jnp.stack([make_liquid_init(N, D, L, rng_key=k)
                              for k in jax.random.split(kl, N_CHAINS)])
    gas_configs = jnp.stack([make_gas_init(N, D, L, rng_key=k)
                              for k in jax.random.split(kg, N_CHAINS)])

    sample_times = []
    energies_liq, energies_gas = [], []
    op_liq, op_gas = [], []

    for ci in range(n_chunks):
        key, kl_run, kg_run = jax.random.split(key, 3)
        liq_configs, liq_e = run_mcmc(
            liq_configs, energy_fn, beta, CHUNK, kl_run, step_size, L, N, D)
        gas_configs, gas_e = run_mcmc(
            gas_configs, energy_fn, beta, CHUNK, kg_run, step_size, L, N, D)
        sample_times.append((ci + 1) * CHUNK)
        energies_liq.append(np.array(liq_e) / N)
        energies_gas.append(np.array(gas_e) / N)
        op_liq.append(compute_op_batch(liq_configs, N, D, L, r_cut))
        op_gas.append(compute_op_batch(gas_configs, N, D, L, r_cut))

    energies_liq = np.array(energies_liq).T   # (n_chains, n_samples)
    energies_gas = np.array(energies_gas).T
    op_liq       = np.array(op_liq).T
    op_gas       = np.array(op_gas).T
    sample_times = np.array(sample_times)

    E_l = energies_liq[:, -1].mean()
    E_g = energies_gas[:, -1].mean()
    du  = abs(E_l - E_g)
    print(f'liq U/N={E_l:.3f}  gas U/N={E_g:.3f}  ΔU/N={du:.3f}')

    # save NPZ
    os.makedirs(args.out_dir, exist_ok=True)
    npz_path = os.path.join(args.out_dir, f'T{T_star:.2f}_data.npz')
    np.savez(npz_path,
             sample_times      = sample_times,
             energies_liq      = energies_liq,
             energies_gas      = energies_gas,
             op_liq            = op_liq,
             op_gas            = op_gas,
             configs_final_liq = np.array(liq_configs),
             configs_final_gas = np.array(gas_configs),
             T_star=T_star, rho=RHO, N=N, D=D, L=L, r_cut=r_cut)

    return (np.array(liq_configs), np.array(gas_configs),
            E_l, E_g, du, key)


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--temperatures', nargs='+', type=float, default=T_SCAN)
    parser.add_argument('--n-total',  type=int,   default=N_TOTAL)
    parser.add_argument('--rcut',     type=float, default=R_CUT)
    parser.add_argument('--seed',     type=int,   default=42)
    parser.add_argument('--out-dir',  default=OUT_DIR)
    args = parser.parse_args()

    L   = float(np.sqrt(N / RHO))
    key = jax.random.PRNGKey(args.seed)

    print(f'Diagnostic 4: Temperature panel  N={N}  ρ*={RHO:.2f}  L={L:.3f}')
    print(f'  T* scan: {args.temperatures}')
    print(f'  {N_CHAINS} chains × {args.n_total//1000}K moves each  '
          f'chunks={CHUNK}  r_cut={args.rcut}')

    energy_fn = make_energy_fn(N, D, L)
    _ = energy_fn(jnp.zeros((2, N * D)))  # warm up compilation

    # single step-size calibration at T*=0.39
    key, k1, k2 = jax.random.split(key, 3)
    calib = jnp.stack([make_liquid_init(N, D, L, rng_key=k)
                        for k in jax.random.split(k1, 4)])
    step_size, rates = calibrate_step_size(calib, energy_fn, 1/0.39, N, D, L, k2)
    print(f'  step_size={step_size:.4f}  acc≈{rates[step_size]:.3f}\n')

    # ── run all temperatures ───────────────────────────────────────────────────
    results = {}
    for T_star in args.temperatures:
        key, subkey = jax.random.split(key)
        liq_f, gas_f, E_l, E_g, du, subkey = run_temperature(
            T_star, args, subkey, energy_fn, step_size, L)
        results[T_star] = dict(liq=liq_f, gas=gas_f, E_l=E_l, E_g=E_g, du=du)

    # ── scatter grid figure ────────────────────────────────────────────────────
    n_T   = len(args.temperatures)
    ncols = N_CHAINS * 2 + 1
    fig, axes = plt.subplots(n_T, ncols,
                             figsize=(ncols * 1.8, n_T * 2.0),
                             gridspec_kw={'wspace': 0.05, 'hspace': 0.35})

    for row, T_star in enumerate(args.temperatures):
        r = results[T_star]
        for ci in range(N_CHAINS):
            scatter_ax(axes[row, ci], r['liq'][ci], N, D, L,
                       f'liq#{ci}', 'crimson')

        ax_div = axes[row, N_CHAINS]
        ax_div.axis('off')
        ax_div.text(0.5, 0.5,
                    f'T*={T_star:.2f}\nρ*={RHO:.2f}\nΔU/N\n{r["du"]:.3f}',
                    ha='center', va='center', fontsize=8,
                    transform=ax_div.transAxes)

        for ci in range(N_CHAINS):
            scatter_ax(axes[row, N_CHAINS + 1 + ci], r['gas'][ci], N, D, L,
                       f'gas#{ci}', 'steelblue')

    fig.text(0.25, 0.995, '◀  liquid-cluster init  ▶',
             ha='center', va='top', fontsize=10, color='crimson', fontweight='bold')
    fig.text(0.75, 0.995, '◀  dispersed gas init  ▶',
             ha='center', va='top', fontsize=10, color='steelblue', fontweight='bold')
    fig.suptitle(
        f'N={N}  ρ*={RHO:.2f}  L={L:.2f}σ  '
        f'({args.n_total//1000}K equil moves)\n'
        'Coexistence: liq stays clustered, gas stays dispersed  |  '
        'Single phase: both look the same',
        y=1.01, fontsize=9)

    grid_path = os.path.join(args.out_dir, 'scatter_grid.png')
    fig.savefig(grid_path, dpi=130, bbox_inches='tight')
    plt.close(fig)
    print(f'\nSaved: {grid_path}')
    print('Next: python scripts/diag5_mixing_time.py')


if __name__ == '__main__':
    main()
