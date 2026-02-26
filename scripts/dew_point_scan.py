"""scripts/dew_point_scan.py

Find the gas-to-coexistence (dew-point) crossover in 2D LJ at rho*=0.08.

Physical setup
--------------
N=128 particles, 2D LJ (r_c=2.5σ shifted, no LRC), rho*=0.08, L=40σ.
All chains start from a dispersed gas initialisation (full-box triangular
lattice + small noise).  We ask: at each T*, does a liquid droplet nucleate?

Protocol per temperature
------------------------
  1. Gas init for each of N_CHAINS=4 independent replicas.
  2. Equilibration: N_EQUIL=300 000 single-particle moves (no measurements).
  3. Production:    N_PROD=300 000 moves, sampling every CHUNK=5 000 moves.
     → 60 measurement points per chain, 240 per temperature.
  4. Step size calibrated once at T*=0.43 (mid-range), reused across T*.
     Candidates extended to (0.2, 0.4, 0.6, 0.8)σ — appropriate for dilute
     gas where large displacements are rarely blocked.

Cluster analysis
----------------
  Bond criterion: r_ij < R_CUT_CLUSTER = 1.5σ (as specified).
  Order parameter (OP): fraction of N particles in the largest connected
  component, averaged over the full production window.
  Gas phase:    OP ≈ 1/N ≈ 0.008 (isolated or paired particles).
  Droplet phase: OP >> 0.1 (macroscopic cluster).

Key insight from prior work at rho*=0.30
-----------------------------------------
  T_c (at critical density ~0.35) ≈ 0.36–0.39.
  At rho*=0.08 (gas branch), the dew point is expected around T*=0.41–0.44.
  Below the dew point, nucleation from gas can be slow — 300K equil moves
  should be sufficient at these temperatures; timeseries plots will reveal
  whether nucleation occurred during production if it was missed in equil.

Outputs (experiments/dew_point/)
---------------------------------
  T{T}_snapshot.png    — final particle configs, 4 panels (one per chain)
  T{T}_timeseries.png  — U/N and OP vs production move for all chains
  T{T}_data.npz        — full time series + final configs
  summary.png          — ⟨U/N⟩ and ⟨OP⟩ vs T* (the key crossover figures)
  results.csv          — table: T*, mean U/N, mean OP, mean n_clusters

Usage
-----
  python scripts/dew_point_scan.py
  python scripts/dew_point_scan.py --n-equil 100000 --n-prod 100000  # quick test
  python scripts/dew_point_scan.py --temperatures 0.42 0.43 0.44     # subset
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
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

from physics import lj_energy, run_mcmc
from diagnostics import make_gas_init, calibrate_step_size, largest_cluster_fraction


# ── constants ──────────────────────────────────────────────────────────────────
N               = 128
D               = 2
RHO             = 0.08
N_CHAINS        = 4
N_EQUIL         = 300_000
N_PROD          = 300_000
CHUNK           = 5_000
R_CUT_CLUSTER   = 1.5       # bond cutoff for cluster OP (σ), as specified
STEP_CANDIDATES = (0.5, 1.0, 1.5, 2.0)   # wide range for dilute gas (rho*=0.08)
CALIB_T         = 0.43      # step-size calibrated at this temperature
T_SCAN          = [0.40, 0.41, 0.42, 0.43, 0.44, 0.45, 0.47]
OUT_DIR         = 'experiments/dew_point'


# ── helpers ────────────────────────────────────────────────────────────────────

def make_energy_fn(L):
    return jax.jit(functools.partial(
        lj_energy, n_particles=N, dimensions=D, box_length=L, use_lrc=False))


def _pbc_dists(coords, L):
    """All pairwise distances for one flat (N*D,) config under PBC."""
    pos = np.array(coords).reshape(N, D)
    ii, jj = np.triu_indices(N, k=1)
    dr = pos[ii] - pos[jj]
    dr -= L * np.round(dr / L)
    return np.sqrt((dr ** 2).sum(axis=-1)), ii, jj


def compute_op(configs, L):
    """Largest cluster fraction for each chain in the batch."""
    return np.array([
        largest_cluster_fraction(np.array(configs[i]), N, D, L, R_CUT_CLUSTER)
        for i in range(N_CHAINS)
    ])


def count_clusters_mean(configs, L):
    """Mean number of connected components across chains."""
    counts = []
    for i in range(N_CHAINS):
        dists, ii, jj = _pbc_dists(configs[i], L)
        bonded = dists < R_CUT_CLUSTER
        if bonded.sum() == 0:
            counts.append(N)   # all isolated
            continue
        rows = np.concatenate([ii[bonded], jj[bonded]])
        cols = np.concatenate([jj[bonded], ii[bonded]])
        adj  = csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(N, N))
        n, _ = connected_components(adj, directed=False)
        counts.append(n)
    return float(np.mean(counts))


# ── per-temperature run ────────────────────────────────────────────────────────

def run_temperature(T_star, L, energy_fn, step_size, key, args):
    beta           = 1.0 / T_star
    n_equil_chunks = args.n_equil // CHUNK
    n_prod_chunks  = args.n_prod  // CHUNK

    print(f'\n── T*={T_star:.2f} ──────────────────────────', flush=True)

    # initialise: 4 independent gas configs
    key, kg = jax.random.split(key)
    configs = jnp.stack([
        make_gas_init(N, D, L, rng_key=k)
        for k in jax.random.split(kg, N_CHAINS)
    ])

    # ── equilibration ─────────────────────────────────────────────────────
    print(f'  Equilibrating {args.n_equil//1000}K moves ... ', end='', flush=True)
    for _ in range(n_equil_chunks):
        key, k_run = jax.random.split(key)
        configs, _ = run_mcmc(configs, energy_fn, beta, CHUNK, k_run,
                              step_size, L, N, D)
    # quick check after equil — flag if a droplet already formed
    op_post_equil = compute_op(configs, L).mean()
    print(f'done.  OP after equil = {op_post_equil:.3f}', flush=True)

    # ── production ────────────────────────────────────────────────────────
    sample_times, e_series, op_series = [], [], []

    print(f'  Production {args.n_prod//1000}K moves: ', end='', flush=True)
    for ci in range(n_prod_chunks):
        key, k_run = jax.random.split(key)
        configs, e = run_mcmc(configs, energy_fn, beta, CHUNK, k_run,
                              step_size, L, N, D)
        sample_times.append((ci + 1) * CHUNK)
        e_series.append(np.array(e) / N)            # (N_CHAINS,)
        op_series.append(compute_op(configs, L))     # (N_CHAINS,)
        if (ci + 1) % 12 == 0:
            print(f'{(ci+1)*CHUNK//1000}k', end=' ', flush=True)
    print('done.', flush=True)

    # shape: (N_CHAINS, n_prod_chunks)
    e_series     = np.array(e_series).T
    op_series    = np.array(op_series).T
    sample_times = np.array(sample_times)

    mean_U  = float(e_series.mean())
    mean_OP = float(op_series.mean())
    mean_nc = count_clusters_mean(np.array(configs), L)

    print(f'  ⟨U/N⟩={mean_U:.4f}   ⟨OP⟩={mean_OP:.4f}   '
          f'⟨n_clusters⟩≈{mean_nc:.1f}', flush=True)

    return dict(
        configs      = np.array(configs),
        e_series     = e_series,
        op_series    = op_series,
        sample_times = sample_times,
        mean_U       = mean_U,
        mean_OP      = mean_OP,
        mean_nc      = mean_nc,
        key          = key,
    )


# ── figure helpers ─────────────────────────────────────────────────────────────

def plot_snapshot(configs, T_star, L, save_path):
    """4-panel scatter of final configs (one per chain)."""
    fig, axes = plt.subplots(1, N_CHAINS,
                             figsize=(N_CHAINS * 3.5, 3.5),
                             gridspec_kw={'wspace': 0.05})
    for ci, ax in enumerate(axes):
        pos = np.array(configs[ci]).reshape(N, D)
        ax.scatter(pos[:, 0], pos[:, 1], s=8, c='steelblue',
                   alpha=0.75, linewidths=0)
        ax.set_xlim(-L/2, L/2); ax.set_ylim(-L/2, L/2)
        ax.set_aspect('equal')
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f'chain {ci}', fontsize=8)
        # draw box
        from matplotlib.patches import Rectangle
        ax.add_patch(Rectangle((-L/2, -L/2), L, L,
                               lw=0.8, edgecolor='k', facecolor='none'))
    fig.suptitle(f'T*={T_star:.2f}   ρ*={RHO}   N={N}   '
                 f'(cluster r_cut={R_CUT_CLUSTER}σ)', fontsize=9)
    fig.savefig(save_path, dpi=120, bbox_inches='tight')
    plt.close(fig)


def plot_timeseries(e_series, op_series, sample_times, T_star, save_path):
    """U/N and cluster OP vs production move for all chains."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.8))
    x = sample_times / 1000   # k-moves
    colours = plt.cm.tab10(np.linspace(0, 0.4, N_CHAINS))

    for ci in range(N_CHAINS):
        axes[0].plot(x, e_series[ci],  color=colours[ci], alpha=0.8, lw=1.2,
                     label=f'chain {ci}')
        axes[1].plot(x, op_series[ci], color=colours[ci], alpha=0.8, lw=1.2,
                     label=f'chain {ci}')

    axes[0].set_xlabel('Production moves (×10³)');  axes[0].set_ylabel('U/N  (ε)')
    axes[0].set_title(f'T*={T_star:.2f}  —  Energy per particle')
    axes[0].legend(fontsize=7)

    axes[1].axhline(1.0/N, color='gray', ls=':', lw=1,
                    label=f'1/N={1/N:.3f} (isolated gas)')
    axes[1].set_xlabel('Production moves (×10³)')
    axes[1].set_ylabel('Largest cluster fraction')
    axes[1].set_title(f'T*={T_star:.2f}  —  Cluster order parameter')
    axes[1].set_ylim(0, None);  axes[1].legend(fontsize=7)

    fig.tight_layout()
    fig.savefig(save_path, dpi=120, bbox_inches='tight')
    plt.close(fig)


def plot_summary(all_T, all_U, all_OP, out_dir):
    """Two-panel summary: ⟨U/N⟩ and ⟨OP⟩ vs T*."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

    axes[0].plot(all_T, all_U, 'o-', color='darkorange', lw=2.0, ms=8)
    axes[0].set_xlabel('T*', fontsize=11);  axes[0].set_ylabel('⟨U/N⟩  (ε)', fontsize=11)
    axes[0].set_title(f'Avg potential energy per particle\nρ*={RHO}  N={N}  '
                      f'(gas init, {N_CHAINS} chains)', fontsize=9)
    axes[0].grid(alpha=0.3)

    axes[1].plot(all_T, all_OP, 's-', color='steelblue', lw=2.0, ms=8)
    axes[1].axhline(1.0/N, color='gray', ls=':', lw=1.2,
                    label=f'OP floor = 1/N = {1/N:.3f}')
    axes[1].set_xlabel('T*', fontsize=11)
    axes[1].set_ylabel('⟨Largest cluster fraction⟩', fontsize=11)
    axes[1].set_title(f'Cluster order parameter (r_cut={R_CUT_CLUSTER}σ)\n'
                      f'ρ*={RHO}  N={N}', fontsize=9)
    axes[1].set_ylim(0, None);  axes[1].legend(fontsize=8);  axes[1].grid(alpha=0.3)

    fig.suptitle(f'2D LJ dew-point scan   N={N}   ρ*={RHO}   L={np.sqrt(N/RHO):.1f}σ',
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'summary.png'), dpi=130, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {os.path.join(out_dir, "summary.png")}')


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='2D LJ dew-point scan at rho*=0.08')
    parser.add_argument('--temperatures', nargs='+', type=float, default=T_SCAN)
    parser.add_argument('--n-equil',  type=int,   default=N_EQUIL)
    parser.add_argument('--n-prod',   type=int,   default=N_PROD)
    parser.add_argument('--seed',     type=int,   default=42)
    parser.add_argument('--out-dir',  default=OUT_DIR)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    L   = float(np.sqrt(N / RHO))
    key = jax.random.PRNGKey(args.seed)

    print('=' * 65)
    print(f'2D LJ dew-point scan')
    print(f'  N={N}  D={D}  ρ*={RHO}  L={L:.3f}σ')
    print(f'  T* sweep: {args.temperatures}')
    print(f'  {N_CHAINS} gas-init chains per temperature')
    print(f'  {args.n_equil//1000}K equil + {args.n_prod//1000}K production  '
          f'(chunk={CHUNK})')
    print(f'  Cluster r_cut = {R_CUT_CLUSTER}σ   |   LJ cutoff = 2.5σ')
    print(f'  Output: {args.out_dir}')
    print('=' * 65)

    # compile energy function
    energy_fn = make_energy_fn(L)
    _ = energy_fn(jnp.zeros((2, N * D)))   # warm-up JIT

    # step-size calibration at CALIB_T using gas configs
    print(f'\nCalibrating step size at T*={CALIB_T} (gas init) ...', flush=True)
    key, k1, k2 = jax.random.split(key, 3)
    calib_configs = jnp.stack([
        make_gas_init(N, D, L, rng_key=k)
        for k in jax.random.split(k1, N_CHAINS)
    ])
    step_size, rates = calibrate_step_size(
        calib_configs, energy_fn, 1.0 / CALIB_T, N, D, L, k2,
        candidates=STEP_CANDIDATES, target_rate=0.35, n_test=2000)
    print(f'  step_size = {step_size:.3f}σ   acceptance rates: '
          + '  '.join(f'{s:.2f}→{r:.3f}' for s, r in rates.items()))

    # ── temperature loop ───────────────────────────────────────────────────
    all_T, all_U, all_OP, all_nc = [], [], [], []

    for T_star in args.temperatures:
        key, subkey = jax.random.split(key)
        r = run_temperature(T_star, L, energy_fn, step_size, subkey, args)

        # per-temperature figures
        plot_snapshot(
            r['configs'], T_star, L,
            os.path.join(args.out_dir, f'T{T_star:.2f}_snapshot.png'))
        plot_timeseries(
            r['e_series'], r['op_series'], r['sample_times'], T_star,
            os.path.join(args.out_dir, f'T{T_star:.2f}_timeseries.png'))

        # save raw data
        np.savez(
            os.path.join(args.out_dir, f'T{T_star:.2f}_data.npz'),
            sample_times = r['sample_times'],
            e_series     = r['e_series'],
            op_series    = r['op_series'],
            configs_final= r['configs'],
            T_star=T_star, rho=RHO, N=N, D=D, L=L,
            r_cut_cluster=R_CUT_CLUSTER,
            mean_U=r['mean_U'], mean_OP=r['mean_OP'], mean_nc=r['mean_nc'])

        all_T.append(T_star)
        all_U.append(r['mean_U'])
        all_OP.append(r['mean_OP'])
        all_nc.append(r['mean_nc'])

    all_T  = np.array(all_T)
    all_U  = np.array(all_U)
    all_OP = np.array(all_OP)
    all_nc = np.array(all_nc)

    # summary figure
    plot_summary(all_T, all_U, all_OP, args.out_dir)

    # printed table
    print('\n' + '=' * 60)
    print(f'  {"T*":>5}   {"⟨U/N⟩":>9}   {"⟨OP⟩":>8}   {"⟨n_clusters⟩":>13}')
    print('  ' + '-' * 56)
    for T, U, op, nc in zip(all_T, all_U, all_OP, all_nc):
        flag = ''
        if op > 0.10:
            flag = '  ← DROPLET'
        elif op > 0.03:
            flag = '  ← nucleating?'
        print(f'  {T:>5.2f}   {U:>9.4f}   {op:>8.4f}   {nc:>13.1f}{flag}')
    print('=' * 60)

    # CSV
    csv_path = os.path.join(args.out_dir, 'results.csv')
    with open(csv_path, 'w') as f:
        f.write('T_star,mean_U_per_N,mean_cluster_OP,mean_n_clusters\n')
        for T, U, op, nc in zip(all_T, all_U, all_OP, all_nc):
            f.write(f'{T:.2f},{U:.6f},{op:.6f},{nc:.2f}\n')

    print(f'\nSaved: {csv_path}')
    print('Dew-point scan complete.')


if __name__ == '__main__':
    main()
