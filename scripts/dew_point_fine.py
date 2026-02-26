"""scripts/dew_point_fine.py

Fine-grained dew-point localisation in 2D LJ at rho*=0.08.

Motivation
----------
The coarse scan (dew_point_scan.py) showed:
  T*=0.44: 3/4 chains with persistent droplet  → coexistence
  T*=0.45: 1/4 chains with persistent droplet  → transition zone
  T*=0.47: 0/4 chains with persistent droplet  → pure gas

This script resolves the transition with:
  - Finer T* spacing (0.005 steps) between 0.45 and 0.47
  - 8 chains per temperature (vs 4) for cleaner statistics
  - "Droplet probability" P_drop: fraction of (chains × production
    timepoints) where OP > DROPLET_OP_THRESHOLD = 0.15.
    P_drop ~ 1 in coexistence, P_drop ~ 0 in pure gas.
    Where it crosses 0.5 is the best single-number estimate of T_dew.

Key design choices
------------------
  - Gas init (full-box triangular lattice) for all chains — unbiased.
  - Step size reused from coarse scan: 2.0σ (47% acceptance for dilute gas).
  - Same equilibration and production lengths as coarse scan: 300K+300K.
  - Error bars on all observables: std over the 8 chains' per-chain means.

Outputs (experiments/dew_point_fine/)
--------------------------------------
  T{T}_snapshot.png    — 8-panel final config scatter
  T{T}_timeseries.png  — U/N and OP traces for all 8 chains
  T{T}_data.npz        — full time series + final configs
  summary.png          — ⟨OP⟩ ± σ, P_drop, and ⟨U/N⟩ ± σ vs T*
  results.csv          — full table
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
from scipy.optimize import curve_fit

from physics import lj_energy, run_mcmc
from diagnostics import make_gas_init, calibrate_step_size, largest_cluster_fraction


# ── constants ──────────────────────────────────────────────────────────────────
N                   = 128
D                   = 2
RHO                 = 0.08
N_CHAINS            = 8
N_EQUIL             = 300_000
N_PROD              = 300_000
CHUNK               = 5_000
R_CUT_CLUSTER       = 1.5
DROPLET_OP_THRESH   = 0.15     # OP above this → chain is "in droplet state"
STEP_SIZE           = 2.0      # pre-calibrated from coarse scan
T_SCAN              = [0.450, 0.455, 0.460, 0.463, 0.465, 0.468, 0.470]
OUT_DIR             = 'experiments/dew_point_fine'


# ── helpers ────────────────────────────────────────────────────────────────────

def make_energy_fn(L):
    return jax.jit(functools.partial(
        lj_energy, n_particles=N, dimensions=D, box_length=L, use_lrc=False))


def compute_op(configs, L):
    return np.array([
        largest_cluster_fraction(np.array(configs[i]), N, D, L, R_CUT_CLUSTER)
        for i in range(N_CHAINS)
    ])


def sigmoid(x, x0, k):
    """Logistic function for fitting P_drop vs T*."""
    return 1.0 / (1.0 + np.exp(k * (x - x0)))


# ── per-temperature run ────────────────────────────────────────────────────────

def run_temperature(T_star, L, energy_fn, key, args):
    beta           = 1.0 / T_star
    n_equil_chunks = args.n_equil // CHUNK
    n_prod_chunks  = args.n_prod  // CHUNK

    print(f'\n── T*={T_star:.3f} ──────────────────────────────', flush=True)

    key, kg = jax.random.split(key)
    configs = jnp.stack([
        make_gas_init(N, D, L, rng_key=k)
        for k in jax.random.split(kg, N_CHAINS)
    ])

    # ── equilibration ─────────────────────────────────────────────────────
    print(f'  Equil {args.n_equil//1000}K ... ', end='', flush=True)
    for _ in range(n_equil_chunks):
        key, k_run = jax.random.split(key)
        configs, _ = run_mcmc(configs, energy_fn, beta, CHUNK, k_run,
                              args.step_size, L, N, D)
    op_eq = compute_op(configs, L)
    print(f'done  (OP post-equil: {op_eq.mean():.3f} ± {op_eq.std():.3f})',
          flush=True)

    # ── production ────────────────────────────────────────────────────────
    sample_times, e_series, op_series = [], [], []

    print(f'  Prod  {args.n_prod//1000}K ... ', end='', flush=True)
    for ci in range(n_prod_chunks):
        key, k_run = jax.random.split(key)
        configs, e = run_mcmc(configs, energy_fn, beta, CHUNK, k_run,
                              args.step_size, L, N, D)
        sample_times.append((ci + 1) * CHUNK)
        e_series.append(np.array(e) / N)
        op_series.append(compute_op(configs, L))
    print('done.', flush=True)

    e_series     = np.array(e_series).T    # (N_CHAINS, n_prod)
    op_series    = np.array(op_series).T
    sample_times = np.array(sample_times)

    # per-chain means (used for error bars)
    chain_mean_U  = e_series.mean(axis=1)   # (N_CHAINS,)
    chain_mean_OP = op_series.mean(axis=1)  # (N_CHAINS,)

    # droplet probability: fraction of (chain × timepoint) pairs with OP > thresh
    p_drop = float((op_series > DROPLET_OP_THRESH).mean())

    # chains whose mean OP exceeds threshold (coarser view)
    chains_with_droplet = int((chain_mean_OP > DROPLET_OP_THRESH).sum())

    print(f'  ⟨U/N⟩ = {chain_mean_U.mean():.4f} ± {chain_mean_U.std():.4f}')
    print(f'  ⟨OP⟩  = {chain_mean_OP.mean():.4f} ± {chain_mean_OP.std():.4f}')
    print(f'  P_drop = {p_drop:.3f}   '
          f'chains_with_droplet = {chains_with_droplet}/{N_CHAINS}', flush=True)

    return dict(
        configs           = np.array(configs),
        e_series          = e_series,
        op_series         = op_series,
        sample_times      = sample_times,
        chain_mean_U      = chain_mean_U,
        chain_mean_OP     = chain_mean_OP,
        mean_U            = float(chain_mean_U.mean()),
        std_U             = float(chain_mean_U.std()),
        mean_OP           = float(chain_mean_OP.mean()),
        std_OP            = float(chain_mean_OP.std()),
        p_drop            = p_drop,
        chains_with_droplet = chains_with_droplet,
        key               = key,
    )


# ── figure helpers ─────────────────────────────────────────────────────────────

def plot_snapshot(configs, T_star, L, save_path):
    ncols = N_CHAINS
    fig, axes = plt.subplots(1, ncols, figsize=(ncols * 2.8, 2.8),
                             gridspec_kw={'wspace': 0.04})
    for ci, ax in enumerate(axes):
        pos = np.array(configs[ci]).reshape(N, D)
        ax.scatter(pos[:, 0], pos[:, 1], s=6, c='steelblue',
                   alpha=0.75, linewidths=0)
        ax.set_xlim(-L/2, L/2); ax.set_ylim(-L/2, L/2)
        ax.set_aspect('equal')
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f'c{ci}', fontsize=7)
    fig.suptitle(f'T*={T_star:.3f}   ρ*={RHO}   N={N}', fontsize=9)
    fig.savefig(save_path, dpi=110, bbox_inches='tight')
    plt.close(fig)


def plot_timeseries(e_series, op_series, sample_times, T_star, p_drop, save_path):
    fig, axes = plt.subplots(1, 2, figsize=(13, 3.8))
    x = sample_times / 1000
    cmap = plt.cm.tab10

    for ci in range(N_CHAINS):
        col = cmap(ci / 10)
        axes[0].plot(x, e_series[ci],  color=col, alpha=0.75, lw=1.0,
                     label=f'c{ci}')
        axes[1].plot(x, op_series[ci], color=col, alpha=0.75, lw=1.0,
                     label=f'c{ci}')

    axes[0].set_xlabel('Production moves (×10³)');  axes[0].set_ylabel('U/N  (ε)')
    axes[0].set_title(f'T*={T_star:.3f}  —  Energy per particle')
    axes[0].legend(fontsize=6, ncol=2)

    axes[1].axhline(DROPLET_OP_THRESH, color='firebrick', ls='--', lw=1.2,
                    label=f'droplet threshold ({DROPLET_OP_THRESH})')
    axes[1].axhline(1.0/N, color='gray', ls=':', lw=1,
                    label=f'gas floor (1/N={1/N:.3f})')
    axes[1].set_xlabel('Production moves (×10³)')
    axes[1].set_ylabel('Largest cluster fraction')
    axes[1].set_title(f'T*={T_star:.3f}  —  Cluster OP   (P_drop={p_drop:.3f})')
    axes[1].set_ylim(0, None)
    axes[1].legend(fontsize=7)

    fig.tight_layout()
    fig.savefig(save_path, dpi=110, bbox_inches='tight')
    plt.close(fig)


def plot_summary(all_T, all_U, all_std_U, all_OP, all_std_OP,
                 all_pdrop, all_cwdrop, out_dir):
    """3-panel summary: ⟨OP⟩±σ, P_drop with sigmoid fit, ⟨U/N⟩±σ."""
    T   = np.array(all_T)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    # ── panel 1: ⟨OP⟩ with per-chain error bars ───────────────────────────
    ax = axes[0]
    ax.errorbar(T, all_OP, yerr=all_std_OP,
                fmt='s-', color='steelblue', lw=2, ms=7, capsize=4,
                label='⟨OP⟩ ± 1σ (over chains)')
    ax.axhline(DROPLET_OP_THRESH, color='firebrick', ls='--', lw=1.2,
               label=f'droplet threshold {DROPLET_OP_THRESH}')
    ax.axhline(1.0/N, color='gray', ls=':', lw=1,
               label=f'gas floor 1/N={1/N:.3f}')
    ax.set_xlabel('T*', fontsize=11);  ax.set_ylabel('⟨Largest cluster fraction⟩', fontsize=10)
    ax.set_title('Cluster order parameter', fontsize=10)
    ax.legend(fontsize=8);  ax.grid(alpha=0.3)

    # ── panel 2: droplet probability + sigmoid fit ────────────────────────
    ax = axes[1]
    p = np.array(all_pdrop)
    ax.scatter(T, p, s=60, zorder=3, color='steelblue',
               label=f'P_drop  (OP > {DROPLET_OP_THRESH},  {N_CHAINS} chains)')

    # secondary axis: chains with droplet count
    ax2 = ax.twinx()
    ax2.scatter(T, all_cwdrop, s=40, marker='^', color='darkorange', zorder=3,
                label=f'N chains w/ droplet (/{N_CHAINS})')
    ax2.set_ylabel(f'Chains with droplet (/{N_CHAINS})', fontsize=9,
                   color='darkorange')
    ax2.tick_params(axis='y', labelcolor='darkorange')
    ax2.set_ylim(-0.5, N_CHAINS + 0.5)
    ax2.yaxis.set_major_locator(plt.MaxNLocator(integer=True))

    # sigmoid fit to P_drop
    try:
        popt, _ = curve_fit(sigmoid, T, p, p0=[T.mean(), 200], maxfev=5000)
        T_fine = np.linspace(T.min(), T.max(), 300)
        ax.plot(T_fine, sigmoid(T_fine, *popt), 'k--', lw=1.5,
                label=f'sigmoid fit  T_dew={popt[0]:.4f}')
        print(f'\nSigmoid fit: T_dew = {popt[0]:.4f}σ  (k={popt[1]:.1f})')
    except Exception as exc:
        print(f'Sigmoid fit failed: {exc}')

    ax.axhline(0.5, color='gray', ls=':', lw=1, label='P=0.5')
    ax.set_xlabel('T*', fontsize=11)
    ax.set_ylabel('Droplet probability  P_drop', fontsize=10)
    ax.set_title('Dew-point transition', fontsize=10)
    ax.set_ylim(-0.05, 1.05)
    # combine legends
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=7, loc='center right')
    ax.grid(alpha=0.3)

    # ── panel 3: ⟨U/N⟩ with error bars ───────────────────────────────────
    ax = axes[2]
    ax.errorbar(T, all_U, yerr=all_std_U,
                fmt='o-', color='darkorange', lw=2, ms=7, capsize=4,
                label='⟨U/N⟩ ± 1σ (over chains)')
    ax.set_xlabel('T*', fontsize=11);  ax.set_ylabel('⟨U/N⟩  (ε)', fontsize=10)
    ax.set_title('Avg potential energy per particle', fontsize=10)
    ax.legend(fontsize=8);  ax.grid(alpha=0.3)

    fig.suptitle(f'2D LJ dew-point fine scan   N={N}   ρ*={RHO}   '
                 f'r_cut_cluster={R_CUT_CLUSTER}σ   {N_CHAINS} chains/T',
                 fontsize=11)
    fig.tight_layout()
    out = os.path.join(out_dir, 'summary.png')
    fig.savefig(out, dpi=130, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {out}')


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Fine dew-point scan around T*=0.45-0.47')
    parser.add_argument('--temperatures', nargs='+', type=float, default=T_SCAN)
    parser.add_argument('--n-equil',  type=int,   default=N_EQUIL)
    parser.add_argument('--n-prod',   type=int,   default=N_PROD)
    parser.add_argument('--step-size', type=float, default=STEP_SIZE)
    parser.add_argument('--seed',     type=int,   default=7)
    parser.add_argument('--out-dir',  default=OUT_DIR)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    L   = float(np.sqrt(N / RHO))
    key = jax.random.PRNGKey(args.seed)

    print('=' * 65)
    print(f'2D LJ dew-point fine scan')
    print(f'  N={N}  ρ*={RHO}  L={L:.3f}σ')
    print(f'  T* sweep: {args.temperatures}')
    print(f'  {N_CHAINS} gas-init chains  |  step_size={args.step_size:.2f}σ')
    print(f'  {args.n_equil//1000}K equil + {args.n_prod//1000}K production')
    print(f'  Droplet threshold: OP > {DROPLET_OP_THRESH}')
    print('=' * 65)

    energy_fn = make_energy_fn(L)
    _ = energy_fn(jnp.zeros((2, N * D)))   # warm-up JIT

    all_T, all_U, all_std_U  = [], [], []
    all_OP, all_std_OP       = [], []
    all_pdrop, all_cwdrop    = [], []

    for T_star in args.temperatures:
        key, subkey = jax.random.split(key)
        r = run_temperature(T_star, L, energy_fn, key=subkey, args=args)

        plot_snapshot(
            r['configs'], T_star, L,
            os.path.join(args.out_dir, f'T{T_star:.3f}_snapshot.png'))
        plot_timeseries(
            r['e_series'], r['op_series'], r['sample_times'],
            T_star, r['p_drop'],
            os.path.join(args.out_dir, f'T{T_star:.3f}_timeseries.png'))

        np.savez(
            os.path.join(args.out_dir, f'T{T_star:.3f}_data.npz'),
            sample_times = r['sample_times'],
            e_series     = r['e_series'],
            op_series    = r['op_series'],
            configs_final= r['configs'],
            T_star=T_star, rho=RHO, N=N, D=D, L=L,
            r_cut_cluster=R_CUT_CLUSTER,
            mean_U=r['mean_U'], std_U=r['std_U'],
            mean_OP=r['mean_OP'], std_OP=r['std_OP'],
            p_drop=r['p_drop'],
            chains_with_droplet=r['chains_with_droplet'])

        all_T.append(T_star);  all_U.append(r['mean_U']);  all_std_U.append(r['std_U'])
        all_OP.append(r['mean_OP']);  all_std_OP.append(r['std_OP'])
        all_pdrop.append(r['p_drop']);  all_cwdrop.append(r['chains_with_droplet'])

    # summary figure
    plot_summary(all_T, all_U, all_std_U, all_OP, all_std_OP,
                 all_pdrop, all_cwdrop, args.out_dir)

    # printed table
    print('\n' + '=' * 78)
    print(f'  {"T*":>6}  {"⟨U/N⟩":>9}  {"±σ_U":>6}  '
          f'{"⟨OP⟩":>7}  {"±σ_OP":>6}  '
          f'{"P_drop":>7}  {"chains":>7}')
    print('  ' + '-' * 74)
    for T, U, sU, op, sop, pd, cwd in zip(
            all_T, all_U, all_std_U, all_OP, all_std_OP,
            all_pdrop, all_cwdrop):
        flag = ''
        if pd > 0.7:   flag = ' ← coexistence'
        elif pd > 0.3: flag = ' ← TRANSITION'
        else:          flag = ' ← gas'
        print(f'  {T:>6.3f}  {U:>9.4f}  {sU:>6.4f}  '
              f'{op:>7.4f}  {sop:>6.4f}  '
              f'{pd:>7.3f}  {cwd:>4}/{N_CHAINS}{flag}')
    print('=' * 78)

    # CSV
    csv_path = os.path.join(args.out_dir, 'results.csv')
    with open(csv_path, 'w') as f:
        f.write('T_star,mean_U,std_U,mean_OP,std_OP,p_drop,chains_with_droplet\n')
        for T, U, sU, op, sop, pd, cwd in zip(
                all_T, all_U, all_std_U, all_OP, all_std_OP,
                all_pdrop, all_cwdrop):
            f.write(f'{T:.3f},{U:.6f},{sU:.6f},'
                    f'{op:.6f},{sop:.6f},{pd:.4f},{cwd}\n')
    print(f'Saved: {csv_path}')
    print('Fine scan complete.')


if __name__ == '__main__':
    main()
