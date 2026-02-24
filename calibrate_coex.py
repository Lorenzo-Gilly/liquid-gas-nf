"""calibrate_coex.py — Systematic thermodynamic calibration for 2D LJ coexistence.

Stages A-E from gas-liquid-tasks.txt.

Usage:
    python calibrate_coex.py [--out-dir experiments/calibration] [--seed 42]

Stage A: Determine r_cut from g(r) at T*=0.42, rho*=0.35.
Stage B: Coarse sweep over (T*, rho*) grid — classify coexistence vs single-phase.
Stage C: Fine temperature sweep at best density — find T_coex via Ashman's D.
Stage D: Verify T_base (warm single-phase reference).
Stage E: Visual gallery PDF.

Parameters adjusted from spec for robustness:
    Stage A:  60K moves  (spec: 20K)  — near-coexistence equilibration is slow.
    Stage B:  20K hot + 50K cold + 50K prod  (spec: 10K+20K+20K)
    Stage C:  30K hot + 100K cold + 200K prod  (spec: 20K+50K+100K)
    Stage D:  50K equil + 100K prod  (spec: 20K+50K)

All energies use lj_energy with use_lrc=False  [A1].
All move counts are single-particle moves [A3].  Comments note full-sweep equiv.
"""

import argparse
import csv
import functools
import json
import os
import warnings

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from physics import lj_energy, run_mcmc, fcc_lattice
from diagnostics import (
    radial_distribution_function,
    find_rcut_from_gr,
    compute_op_batch,
    potential_energy_per_particle,
    multi_panel_diagnostic,
    snapshot_plot,
    plot_op_timeseries,
    make_liquid_init,
    make_gas_init,
    calibrate_step_size,
    ashmans_d,
    mean_coordination,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_lj(N, D, L):
    """JIT-compiled LJ energy function with use_lrc=False [A1]."""
    return jax.jit(functools.partial(
        lj_energy,
        n_particles=N,
        dimensions=D,
        box_length=L,
        use_lrc=False,
    ))


def _run_chains(init_configs, energy_fn, beta, step_size,
                n_hot, n_cold, n_prod, n_save_every,
                N, D, L, key):
    """Run n_chains MCMC chains in parallel (vectorised).

    Protocol: hot equilibration (0.2*beta) → cold equilibration (beta) →
    production sampling.

    Args:
        init_configs: (n_chains, N*D) initial configurations.
        energy_fn:    callable (B, N*D) -> (B,).
        beta:         target inverse temperature.
        step_size:    Gaussian displacement σ.
        n_hot:        single-particle moves for hot equilibration.
        n_cold:       single-particle moves for cold equilibration.
        n_prod:       single-particle moves for production.
        n_save_every: save configs every this many production moves.
        N, D, L:      system parameters.
        key:          JAX PRNG key.
    Returns:
        samples: (n_samples_per_chain, n_chains, N*D) production configs.
                 n_samples_per_chain = n_prod // n_save_every
    """
    configs = jnp.array(init_configs)

    # Hot equilibration
    key, k1 = jax.random.split(key)
    configs, _ = run_mcmc(configs, energy_fn, 0.2 * beta, n_hot,
                          k1, step_size, L, N, D)

    # Cold equilibration
    key, k2 = jax.random.split(key)
    configs, _ = run_mcmc(configs, energy_fn, beta, n_cold,
                          k2, step_size, L, N, D)

    # Production: save every n_save_every moves
    n_saves = n_prod // n_save_every
    samples = []
    for _ in range(n_saves):
        key, k3 = jax.random.split(key)
        configs, _ = run_mcmc(configs, energy_fn, beta, n_save_every,
                              k3, step_size, L, N, D)
        samples.append(np.array(configs))  # (n_chains, N*D)

    return np.stack(samples, axis=0)  # (n_saves, n_chains, N*D)


def _build_init_batch(init_fn, n_chains, N, D, L, key):
    """Stack n_chains initial configurations from init_fn."""
    keys = jax.random.split(key, n_chains)
    return jnp.stack([init_fn(N, D, L, rng_key=k) for k in keys])


# ---------------------------------------------------------------------------
# Stage A — Determine r_cut
# ---------------------------------------------------------------------------

def stage_a(N, D, out_dir, key, seed):
    """Compute g(r) at T*=0.42, rho*=0.35 and find first minimum.

    Uses 60K moves total; g(r) computed from last 30K (sampled every 300).
    [A3]: 60K moves = 60000/32 = 1875 full sweeps.
    """
    T_star = 0.42
    rho_star = 0.35
    beta = 1.0 / T_star
    L = float(np.sqrt(N / rho_star))
    step_size = 0.2          # default; not calibrated here (Stage B does that)

    print(f"\n{'='*60}")
    print(f"Stage A: r_cut determination")
    print(f"  T*={T_star}, rho*={rho_star}, L={L:.3f}")
    print(f"  60K moves (1875 full sweeps); g(r) from last 30K")
    print(f"{'='*60}")

    energy_fn = _make_lj(N, D, L)
    os.makedirs(out_dir, exist_ok=True)

    # Single chain (B=1)
    key, k_init, k_run = jax.random.split(key, 3)
    x0 = make_liquid_init(N, D, L, rng_key=k_init)
    configs = x0[None]   # (1, N*D)

    # Hot equilibration: 15K moves
    key, k1 = jax.random.split(key)
    configs, _ = run_mcmc(configs, energy_fn, 0.2 * beta, 15_000,
                          k1, step_size, L, N, D)

    # Cold equilibration: 15K moves
    key, k2 = jax.random.split(key)
    configs, _ = run_mcmc(configs, energy_fn, beta, 15_000,
                          k2, step_size, L, N, D)

    # Production: 30K moves, sample every 300 → 100 samples
    samples = []
    for _ in range(100):
        key, k3 = jax.random.split(key)
        configs, _ = run_mcmc(configs, energy_fn, beta, 300,
                              k3, step_size, L, N, D)
        samples.append(np.array(configs[0]))  # (N*D,)
    samples = np.stack(samples)   # (100, N*D)

    r_vals, gr_vals = radial_distribution_function(samples, N, D, L)
    r_cut = find_rcut_from_gr(r_vals, gr_vals, r_min=1.0, r_max=2.2)
    print(f"  → r_cut = {r_cut:.4f} σ")

    # Save g(r) plot
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(r_vals, gr_vals, 'k-', lw=1.5)
    ax.axhline(1.0, color='gray', ls='--', lw=1, alpha=0.6)
    ax.axvline(r_cut, color='red', ls=':', lw=1.5, label=f'r_cut={r_cut:.3f}')
    ax.set_xlabel('r / σ')
    ax.set_ylabel('g(r)')
    ax.set_title(f'Stage A: g(r) at T*={T_star}, ρ*={rho_star}')
    ax.legend()
    ax.set_xlim(0, L / 2)
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, 'stage_a_gr.png'), dpi=120)
    plt.close(fig)

    # Save r_cut
    result = {'r_cut': r_cut, 'T_star': T_star, 'rho_star': rho_star, 'L': L}
    with open(os.path.join(out_dir, 'stage_a_rcut.json'), 'w') as f:
        json.dump(result, f, indent=2)
    print(f"  Saved: stage_a_gr.png, stage_a_rcut.json")
    return r_cut, L


# ---------------------------------------------------------------------------
# Stage B — Coarse sweep
# ---------------------------------------------------------------------------

T_LIST  = [0.30, 0.33, 0.36, 0.38, 0.40, 0.42, 0.44, 0.46, 0.48, 0.50, 0.55, 0.65]
RHO_LIST = [0.25, 0.30, 0.35, 0.40, 0.50]

# Single-particle move counts [A3]
# 20K hot = 625 sweeps; 50K cold = 1562 sweeps; 50K prod = 1562 sweeps
_B_N_HOT   = 20_000
_B_N_COLD  = 50_000
_B_N_PROD  = 50_000
_B_SAVE_EV = 200      # save every 200 moves → 250 samples/chain
_B_N_CHAINS = 8       # per init type


def stage_b(N, D, r_cut, out_dir, key, seed):
    """Coarse (T*, rho*) sweep to locate the coexistence region.

    For each of 60 grid points: auto-calibrate step_size, run 8 liquid-init +
    8 gas-init chains, compute OP gap, classify COEXISTENCE / UNCLEAR /
    SINGLE PHASE.

    [A3] All move counts are single-particle moves.
    """
    print(f"\n{'='*60}")
    print(f"Stage B: Coarse sweep  ({len(T_LIST)}×{len(RHO_LIST)} = "
          f"{len(T_LIST)*len(RHO_LIST)} grid points)")
    print(f"  Per point: {_B_N_CHAINS} liq-init + {_B_N_CHAINS} gas-init chains")
    print(f"  Moves: {_B_N_HOT} hot + {_B_N_COLD} cold + {_B_N_PROD} prod "
          f"({_B_N_PROD//_B_SAVE_EV} samples/chain)")
    print(f"{'='*60}")

    coarse_dir = os.path.join(out_dir, 'coarse_sweep')
    os.makedirs(coarse_dir, exist_ok=True)

    rows = []
    best_op_gap = -1.0
    best_T = None
    best_rho = None

    for T_star in T_LIST:
        for rho_star in RHO_LIST:
            beta = 1.0 / T_star
            L = float(np.sqrt(N / rho_star))
            energy_fn = _make_lj(N, D, L)

            print(f"  T*={T_star:.2f}  rho*={rho_star:.2f}  L={L:.3f}  ", end='', flush=True)

            # --- Step size calibration ---
            key, k_cal = jax.random.split(key)
            x0 = make_liquid_init(N, D, L)
            step_size, rates = calibrate_step_size(
                x0[None], energy_fn, beta, N, D, L, k_cal,
                candidates=(0.05, 0.1, 0.2, 0.4), n_test=1000)

            # --- Build initial configs ---
            key, k_liq, k_gas = jax.random.split(key, 3)
            liq_inits = _build_init_batch(make_liquid_init, _B_N_CHAINS, N, D, L, k_liq)
            gas_inits = _build_init_batch(make_gas_init,    _B_N_CHAINS, N, D, L, k_gas)

            # --- Run chains ---
            key, k1, k2 = jax.random.split(key, 3)
            samp_liq = _run_chains(liq_inits, energy_fn, beta, step_size,
                                   _B_N_HOT, _B_N_COLD, _B_N_PROD, _B_SAVE_EV,
                                   N, D, L, k1)  # (n_saves, n_chains, N*D)
            samp_gas = _run_chains(gas_inits, energy_fn, beta, step_size,
                                   _B_N_HOT, _B_N_COLD, _B_N_PROD, _B_SAVE_EV,
                                   N, D, L, k2)

            # Flatten: (n_saves*n_chains, N*D)
            flat_liq = samp_liq.reshape(-1, N * D)
            flat_gas = samp_gas.reshape(-1, N * D)

            op_liq = compute_op_batch(flat_liq, N, D, L, r_cut)
            op_gas = compute_op_batch(flat_gas, N, D, L, r_cut)

            mean_op_liq = float(op_liq.mean())
            mean_op_gas = float(op_gas.mean())
            op_gap = abs(mean_op_liq - mean_op_gas)

            if op_gap > 0.4:
                classification = 'COEXISTENCE'
            elif op_gap < 0.1:
                classification = 'SINGLE_PHASE'
            else:
                classification = 'UNCLEAR'

            print(f"OP_liq={mean_op_liq:.3f}  OP_gas={mean_op_gas:.3f}  "
                  f"gap={op_gap:.3f}  → {classification}")

            rows.append({
                'T_star': T_star, 'rho_star': rho_star, 'L': round(L, 4),
                'step_size': step_size,
                'mean_op_liq': round(mean_op_liq, 4),
                'mean_op_gas': round(mean_op_gas, 4),
                'op_gap': round(op_gap, 4),
                'classification': classification,
            })

            if op_gap > best_op_gap:
                best_op_gap = op_gap
                best_T = T_star
                best_rho = rho_star

            # Save diagnostic plots for classified coexistence points
            if classification == 'COEXISTENCE':
                tag = f'T{T_star:.2f}_rho{rho_star:.2f}'
                multi_panel_diagnostic(
                    flat_liq[:200], N, D, L, r_cut, energy_fn,
                    title=f'Liq-init  T*={T_star} ρ*={rho_star} {classification}',
                    save_path=os.path.join(coarse_dir, f'{tag}_liq.png'))
                multi_panel_diagnostic(
                    flat_gas[:200], N, D, L, r_cut, energy_fn,
                    title=f'Gas-init  T*={T_star} ρ*={rho_star} {classification}',
                    save_path=os.path.join(coarse_dir, f'{tag}_gas.png'))

    # --- Save summary CSV ---
    csv_path = os.path.join(coarse_dir, 'summary.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n  Summary saved: {csv_path}")
    print(f"  Best OP gap {best_op_gap:.3f} at T*={best_T}, rho*={best_rho}")

    # --- Phase diagram scatter ---
    fig, ax = plt.subplots(figsize=(8, 5))
    colours = {'COEXISTENCE': 'red', 'UNCLEAR': 'orange', 'SINGLE_PHASE': 'steelblue'}
    for row in rows:
        ax.scatter(row['rho_star'], row['T_star'],
                   c=colours[row['classification']],
                   s=200 * row['op_gap'] + 30,
                   edgecolors='k', linewidths=0.4, zorder=2)
    for label, c in colours.items():
        ax.scatter([], [], c=c, label=label, s=60)
    ax.set_xlabel('ρ*')
    ax.set_ylabel('T*')
    ax.set_title('Stage B: Phase diagram (dot size ∝ OP gap)')
    ax.legend()
    plt.tight_layout()
    fig.savefig(os.path.join(coarse_dir, 'phase_diagram.png'), dpi=120)
    plt.close(fig)

    return best_rho, rows


# ---------------------------------------------------------------------------
# Stage C — Fine temperature sweep
# ---------------------------------------------------------------------------

# Single-particle move counts [A3]
# 30K hot = 937 sweeps; 100K cold = 3125 sweeps; 200K prod = 6250 sweeps
_C_N_HOT   = 30_000
_C_N_COLD  = 100_000
_C_N_PROD  = 200_000
_C_SAVE_EV = 100      # save every 100 moves → 2000 samples/chain
_C_N_CHAINS = 16      # per init type


def stage_c(N, D, rho_best, r_cut, out_dir, key, coarse_rows):
    """Fine temperature sweep at rho_best.

    12 temperatures ΔT*=0.01 centred on the coarse-sweep coexistence range.
    Selects T_coex = T maximising Ashman's D with both phases metastable.
    """
    # Find the coexistence temperature range from Stage B
    coex_rows = [r for r in coarse_rows
                 if r['rho_star'] == rho_best and r['classification'] == 'COEXISTENCE']
    if not coex_rows:
        coex_rows = [r for r in coarse_rows if r['rho_star'] == rho_best]
        coex_rows.sort(key=lambda r: -r['op_gap'])

    T_centre = coex_rows[0]['T_star'] if coex_rows else 0.40
    T_list_fine = [round(T_centre - 0.05 + i * 0.01, 3) for i in range(12)]

    print(f"\n{'='*60}")
    print(f"Stage C: Fine T* sweep at rho*={rho_best:.2f}")
    print(f"  T* range: {T_list_fine[0]:.3f} – {T_list_fine[-1]:.3f}  "
          f"(ΔT*=0.01, {len(T_list_fine)} points)")
    print(f"  Per point: {_C_N_CHAINS} liq-init + {_C_N_CHAINS} gas-init chains")
    print(f"  Moves: {_C_N_HOT} hot + {_C_N_COLD} cold + {_C_N_PROD} prod "
          f"({_C_N_PROD//_C_SAVE_EV} samples/chain)")
    print(f"{'='*60}")

    fine_dir = os.path.join(out_dir, 'fine_sweep')
    os.makedirs(fine_dir, exist_ok=True)

    L = float(np.sqrt(N / rho_best))
    energy_fn = _make_lj(N, D, L)

    results = []
    best_D_val = -1.0
    T_coex = T_list_fine[0]

    for T_star in T_list_fine:
        beta = 1.0 / T_star
        print(f"  T*={T_star:.3f}  ", end='', flush=True)

        # Calibrate step size
        key, k_cal = jax.random.split(key)
        x0 = make_liquid_init(N, D, L)
        step_size, _ = calibrate_step_size(
            x0[None], energy_fn, beta, N, D, L, k_cal, n_test=2000)

        # Build and run chains
        key, k_liq, k_gas = jax.random.split(key, 3)
        liq_inits = _build_init_batch(make_liquid_init, _C_N_CHAINS, N, D, L, k_liq)
        gas_inits = _build_init_batch(make_gas_init,    _C_N_CHAINS, N, D, L, k_gas)

        key, k1, k2 = jax.random.split(key, 3)
        samp_liq = _run_chains(liq_inits, energy_fn, beta, step_size,
                               _C_N_HOT, _C_N_COLD, _C_N_PROD, _C_SAVE_EV,
                               N, D, L, k1)
        samp_gas = _run_chains(gas_inits, energy_fn, beta, step_size,
                               _C_N_HOT, _C_N_COLD, _C_N_PROD, _C_SAVE_EV,
                               N, D, L, k2)

        # (n_saves, n_chains, N*D) → flatten chains, keep time axis
        # op_liq_series: (n_saves, n_chains)
        op_liq_series = np.array([
            compute_op_batch(samp_liq[t], N, D, L, r_cut)
            for t in range(samp_liq.shape[0])])
        op_gas_series = np.array([
            compute_op_batch(samp_gas[t], N, D, L, r_cut)
            for t in range(samp_gas.shape[0])])

        # Combined OP for Ashman's D
        all_op = np.concatenate([op_liq_series.ravel(), op_gas_series.ravel()])
        D_val, gmm = ashmans_d(all_op)

        op_gap = abs(op_liq_series.mean() - op_gas_series.mean())
        n_transitions_liq = _count_transitions(op_liq_series.mean(axis=1))
        n_transitions_gas = _count_transitions(op_gas_series.mean(axis=1))

        print(f"Ashman D={D_val:.2f}  OP gap={op_gap:.3f}  "
              f"transitions liq={n_transitions_liq} gas={n_transitions_gas}")

        # Save OP time series plot
        tag = f'T{T_star:.3f}'
        plot_op_timeseries(
            op_liq_series.T, op_gas_series.T,
            title=f'Stage C: T*={T_star:.3f}  ρ*={rho_best:.2f}  D={D_val:.2f}',
            save_path=os.path.join(fine_dir, f'{tag}_timeseries.png'))

        # OP histogram
        _plot_op_histogram(all_op, gmm, T_star, rho_best, D_val,
                           os.path.join(fine_dir, f'{tag}_op_hist.png'))

        # g(r) from liq-init and gas-init separately
        flat_liq = samp_liq.reshape(-1, N * D)
        flat_gas = samp_gas.reshape(-1, N * D)
        r_liq, g_liq = radial_distribution_function(flat_liq[:500], N, D, L)
        r_gas, g_gas = radial_distribution_function(flat_gas[:500], N, D, L)
        _plot_gr_pair(r_liq, g_liq, r_gas, g_gas, r_cut, T_star,
                      os.path.join(fine_dir, f'{tag}_gr.png'))

        results.append({
            'T_star': T_star, 'D_ashman': D_val, 'op_gap': op_gap,
            'n_trans_liq': n_transitions_liq, 'n_trans_gas': n_transitions_gas,
            'step_size': step_size,
        })

        # Select T_coex: maximise Ashman's D (bimodality)
        if np.isfinite(D_val) and D_val > best_D_val and op_gap > 0.2:
            best_D_val = D_val
            T_coex = T_star

    if best_D_val < 2.0:
        warnings.warn(
            f"Stage C: No temperature achieved Ashman D > 2 "
            f"(best D={best_D_val:.2f} at T*={T_coex:.3f}). "
            "Consider increasing N to 64 and rerunning.")

    print(f"\n  → T_coex = {T_coex:.3f}  (Ashman D = {best_D_val:.2f})")

    # D vs T* summary plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    T_vals = [r['T_star'] for r in results]
    D_vals = [r['D_ashman'] for r in results]
    op_vals = [r['op_gap'] for r in results]
    axes[0].plot(T_vals, D_vals, 'o-', color='navy')
    axes[0].axhline(2.0, color='red', ls='--', lw=1, label="D=2 threshold")
    axes[0].axvline(T_coex, color='green', ls=':', lw=1.5, label=f'T_coex={T_coex:.3f}')
    axes[0].set_xlabel('T*')
    axes[0].set_ylabel("Ashman's D")
    axes[0].set_title("Stage C: Bimodality vs T*")
    axes[0].legend(fontsize=8)
    axes[1].plot(T_vals, op_vals, 's-', color='darkorange')
    axes[1].axvline(T_coex, color='green', ls=':', lw=1.5)
    axes[1].set_xlabel('T*')
    axes[1].set_ylabel('OP gap (|liq − gas|)')
    axes[1].set_title("Stage C: OP gap vs T*")
    plt.tight_layout()
    fig.savefig(os.path.join(fine_dir, 'summary_D_and_gap.png'), dpi=120)
    plt.close(fig)

    with open(os.path.join(fine_dir, 'results.json'), 'w') as f:
        json.dump({'T_coex': T_coex, 'rho_best': rho_best,
                   'D_ashman_best': best_D_val, 'sweep': results}, f, indent=2)

    return T_coex


# ---------------------------------------------------------------------------
# Stage D — Verify T_base
# ---------------------------------------------------------------------------

# [A3]: 50K moves = 1562 full sweeps; 100K prod = 3125 full sweeps
_D_N_EQUIL  = 50_000
_D_N_PROD   = 100_000
_D_SAVE_EV  = 200     # → 500 samples/chain
_D_N_CHAINS = 16      # 8 liq + 8 gas


def stage_d(N, D, T_coex, rho_best, r_cut, out_dir, key):
    """Test T_base candidates — find the lowest T above T_coex that is
    ergodic (both inits → same distribution, unimodal OP).

    Tests: [T_coex+0.10, T_coex+0.20, T_coex+0.30, 0.80]
    Acceptance criterion:
        KS test p > 0.05  (both inits give same OP distribution)
        Ashman's D < 1.5  (unimodal)
    """
    from scipy.stats import ks_2samp

    candidates = sorted(set([
        round(T_coex + 0.10, 3),
        round(T_coex + 0.20, 3),
        round(T_coex + 0.30, 3),
        0.80,
    ]))
    L = float(np.sqrt(N / rho_best))

    print(f"\n{'='*60}")
    print(f"Stage D: T_base verification at rho*={rho_best:.2f}, L={L:.3f}")
    print(f"  Candidates: {candidates}")
    print(f"  Moves: {_D_N_EQUIL} equil + {_D_N_PROD} prod "
          f"({_D_N_PROD//_D_SAVE_EV} samples/chain)")
    print(f"{'='*60}")

    d_dir = os.path.join(out_dir, 'stage_d')
    os.makedirs(d_dir, exist_ok=True)
    energy_fn = _make_lj(N, D, L)

    T_base = None
    results = []

    for T_star in candidates:
        beta = 1.0 / T_star
        print(f"  T*={T_star:.3f}  ", end='', flush=True)

        n_liq = _D_N_CHAINS // 2
        n_gas = _D_N_CHAINS - n_liq

        key, k_cal = jax.random.split(key)
        x0 = make_liquid_init(N, D, L)
        step_size, _ = calibrate_step_size(
            x0[None], energy_fn, beta, N, D, L, k_cal, n_test=2000)

        key, k_liq, k_gas = jax.random.split(key, 3)
        liq_inits = _build_init_batch(make_liquid_init, n_liq, N, D, L, k_liq)
        gas_inits = _build_init_batch(make_gas_init,    n_gas, N, D, L, k_gas)

        # For Stage D, combine equil into a single MCMC call
        def _run_d(inits, k):
            cfgs = jnp.array(inits)
            k, k1 = jax.random.split(k)
            cfgs, _ = run_mcmc(cfgs, energy_fn, 0.2 * beta, _D_N_EQUIL // 2,
                               k1, step_size, L, N, D)
            k, k2 = jax.random.split(k)
            cfgs, _ = run_mcmc(cfgs, energy_fn, beta, _D_N_EQUIL // 2,
                               k2, step_size, L, N, D)
            samples = []
            for _ in range(_D_N_PROD // _D_SAVE_EV):
                k, k3 = jax.random.split(k)
                cfgs, _ = run_mcmc(cfgs, energy_fn, beta, _D_SAVE_EV,
                                   k3, step_size, L, N, D)
                samples.append(np.array(cfgs))
            return np.stack(samples)   # (n_saves, n_chains, N*D)

        key, k1, k2 = jax.random.split(key, 3)
        samp_liq = _run_d(liq_inits, k1)
        samp_gas = _run_d(gas_inits, k2)

        flat_liq = samp_liq.reshape(-1, N * D)
        flat_gas = samp_gas.reshape(-1, N * D)
        op_liq = compute_op_batch(flat_liq, N, D, L, r_cut)
        op_gas = compute_op_batch(flat_gas, N, D, L, r_cut)
        all_op  = np.concatenate([op_liq, op_gas])

        ks_stat, ks_p = ks_2samp(op_liq, op_gas)
        D_val, _ = ashmans_d(all_op)

        ergodic = (ks_p > 0.05) and (np.isfinite(D_val) and D_val < 1.5)
        print(f"KS p={ks_p:.3f}  Ashman D={D_val:.2f}  "
              f"→ {'PASS ✓' if ergodic else 'FAIL ✗'}")

        results.append({
            'T_star': T_star, 'ks_p': ks_p, 'D_ashman': D_val,
            'ergodic': ergodic, 'step_size': step_size,
        })

        if ergodic and T_base is None:
            T_base = T_star   # lowest passing candidate

        multi_panel_diagnostic(
            flat_liq[:200], N, D, L, r_cut, energy_fn,
            title=f'Stage D: T*={T_star} liq-init  KS_p={ks_p:.3f} D={D_val:.2f}',
            save_path=os.path.join(d_dir, f'T{T_star:.3f}_liq.png'))

    if T_base is None:
        T_base = candidates[-1]
        warnings.warn(
            "Stage D: No candidate passed both KS and Ashman D tests. "
            f"Using T_base={T_base:.3f} (highest candidate) as fallback.")

    print(f"\n  → T_base = {T_base:.3f}")

    with open(os.path.join(d_dir, 'results.json'), 'w') as f:
        json.dump({'T_base': T_base, 'T_coex': T_coex,
                   'rho_best': rho_best, 'sweep': results}, f, indent=2)

    return T_base


# ---------------------------------------------------------------------------
# Stage E — Visual gallery PDF
# ---------------------------------------------------------------------------

def stage_e(N, D, T_base, T_coex, rho_best, r_cut, out_dir, key):
    """Generate a 7-page diagnostic PDF gallery.

    Page 1: Phase diagram + operating points.
    Page 2: Base (T_base) — snapshots, OP, energy, g(r).
    Page 3: Target liquid — same.
    Page 4: Target gas — same.
    Page 5: Coexistence — bimodal OP, OP time series, Ashman D.
    Page 6: T dependence — OP histograms across temperatures.
    Page 7: Summary table.
    """
    print(f"\n{'='*60}")
    print(f"Stage E: Visual gallery PDF")
    print(f"  T_base={T_base:.3f}  T_coex={T_coex:.3f}  rho*={rho_best:.2f}")
    print(f"{'='*60}")

    L = float(np.sqrt(N / rho_best))
    energy_fn = _make_lj(N, D, L)
    pdf_path = os.path.join(out_dir, 'calibration_gallery.pdf')

    def _run_phase(T, n_chains, init_fn):
        """Quick run: 20K hot + 50K cold + 20K prod, save every 200."""
        beta = 1.0 / T
        nonlocal key
        key, k_cal = jax.random.split(key)
        x0 = make_liquid_init(N, D, L)
        step_size, _ = calibrate_step_size(
            x0[None], energy_fn, beta, N, D, L, k_cal, n_test=1000)
        key, ki = jax.random.split(key)
        inits = _build_init_batch(init_fn, n_chains, N, D, L, ki)
        key, kr = jax.random.split(key)
        samp = _run_chains(inits, energy_fn, beta, step_size,
                           20_000, 50_000, 20_000, 200, N, D, L, kr)
        return samp.reshape(-1, N * D), step_size

    configs_base, _  = _run_phase(T_base, 8, make_liquid_init)
    configs_liq, _   = _run_phase(T_coex, 8, make_liquid_init)
    configs_gas, _   = _run_phase(T_coex, 8, make_gas_init)

    # Coexistence: run more chains to show bimodality
    def _run_coex_series(T, n_each):
        nonlocal key
        beta = 1.0 / T
        key, k_cal = jax.random.split(key)
        x0 = make_liquid_init(N, D, L)
        step_size, _ = calibrate_step_size(
            x0[None], energy_fn, beta, N, D, L, k_cal, n_test=1000)
        key, kl, kg = jax.random.split(key, 3)
        l_inits = _build_init_batch(make_liquid_init, n_each, N, D, L, kl)
        g_inits = _build_init_batch(make_gas_init,    n_each, N, D, L, kg)
        key, k1, k2 = jax.random.split(key, 3)
        sl = _run_chains(l_inits, energy_fn, beta, step_size,
                         _C_N_HOT, _C_N_COLD, 40_000, 200, N, D, L, k1)
        sg = _run_chains(g_inits, energy_fn, beta, step_size,
                         _C_N_HOT, _C_N_COLD, 40_000, 200, N, D, L, k2)
        op_l = np.array([compute_op_batch(sl[t], N, D, L, r_cut)
                         for t in range(sl.shape[0])])
        op_g = np.array([compute_op_batch(sg[t], N, D, L, r_cut)
                         for t in range(sg.shape[0])])
        return sl.reshape(-1, N*D), sg.reshape(-1, N*D), op_l, op_g

    coex_liq, coex_gas, op_l_series, op_g_series = _run_coex_series(T_coex, 8)
    all_coex = np.concatenate([coex_liq, coex_gas])
    all_op = compute_op_batch(all_coex, N, D, L, r_cut)
    D_coex, gmm_coex = ashmans_d(all_op)

    with PdfPages(pdf_path) as pdf:

        # Page 1: Phase diagram
        coarse_csv = os.path.join(out_dir, 'coarse_sweep', 'summary.csv')
        if os.path.exists(coarse_csv):
            fig = _page1_phase_diagram(coarse_csv, T_base, T_coex, rho_best)
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)

        # Pages 2-4: diagnostic panels
        for configs, label in [
            (configs_base, f'Base T*={T_base:.3f}'),
            (configs_liq,  f'Target liquid T*={T_coex:.3f}'),
            (configs_gas,  f'Target gas T*={T_coex:.3f}'),
        ]:
            fig = multi_panel_diagnostic(configs[:200], N, D, L, r_cut,
                                         energy_fn, title=label)
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)

        # Page 5: coexistence bimodality + time series
        fig = _page5_coexistence(all_op, op_l_series, op_g_series,
                                  D_coex, T_coex, rho_best)
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)

        # Page 6: T-dependence OP histograms (using Stage C fine sweep data)
        fine_json = os.path.join(out_dir, 'fine_sweep', 'results.json')
        if os.path.exists(fine_json):
            fig = _page6_T_dependence(fine_json)
            if fig is not None:
                pdf.savefig(fig, bbox_inches='tight')
                plt.close(fig)

        # Page 7: Summary table
        fig = _page7_summary(T_base, T_coex, rho_best, r_cut, L, D_coex)
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)

    print(f"  Gallery saved: {pdf_path}")
    return pdf_path


# ---------------------------------------------------------------------------
# Gallery helper plots
# ---------------------------------------------------------------------------

def _page1_phase_diagram(coarse_csv, T_base, T_coex, rho_best):
    rows = []
    with open(coarse_csv) as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    fig, ax = plt.subplots(figsize=(8, 6))
    colours = {'COEXISTENCE': 'red', 'UNCLEAR': 'orange', 'SINGLE_PHASE': 'steelblue'}
    for row in rows:
        ax.scatter(float(row['rho_star']), float(row['T_star']),
                   c=colours.get(row['classification'], 'gray'),
                   s=200 * float(row['op_gap']) + 30,
                   edgecolors='k', linewidths=0.4, zorder=2)
    for label, c in colours.items():
        ax.scatter([], [], c=c, label=label, s=60)
    ax.scatter(rho_best, T_coex, marker='*', s=400, c='gold',
               edgecolors='k', zorder=5, label=f'T_coex={T_coex:.3f}')
    ax.scatter(rho_best, T_base, marker='^', s=300, c='lime',
               edgecolors='k', zorder=5, label=f'T_base={T_base:.3f}')
    ax.set_xlabel('ρ*')
    ax.set_ylabel('T*')
    ax.set_title('Phase diagram (dot size ∝ OP gap)')
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


def _page5_coexistence(all_op, op_l_series, op_g_series, D_coex, T_coex, rho_best):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].hist(all_op, bins=40, color='purple', edgecolor='white', lw=0.4)
    axes[0].set_xlabel('OP')
    axes[0].set_ylabel('Count')
    axes[0].set_title(f"Coexistence OP (Ashman D={D_coex:.2f})")
    for i, ts in enumerate(op_l_series.T):
        axes[1].plot(ts, color='red', alpha=0.4, lw=0.5,
                     label='liq-init' if i == 0 else None)
    for i, ts in enumerate(op_g_series.T):
        axes[1].plot(ts, color='steelblue', alpha=0.4, lw=0.5,
                     label='gas-init' if i == 0 else None)
    axes[1].set_ylim(0, 1)
    axes[1].set_xlabel('Sample index')
    axes[1].set_ylabel('OP')
    axes[1].set_title(f'T*={T_coex:.3f}  ρ*={rho_best:.2f}  OP time series')
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    return fig


def _page6_T_dependence(fine_json):
    try:
        with open(fine_json) as f:
            data = json.load(f)
        T_vals = [r['T_star'] for r in data['sweep']]
        D_vals = [r['D_ashman'] for r in data['sweep']]
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(T_vals, D_vals, 'o-', color='navy')
        ax.axhline(2.0, color='red', ls='--', lw=1, label="D=2 threshold")
        ax.axvline(data.get('T_coex', T_vals[0]), color='green', ls=':',
                   lw=1.5, label=f"T_coex={data.get('T_coex','?'):.3f}")
        ax.set_xlabel('T*')
        ax.set_ylabel("Ashman's D")
        ax.set_title("T-dependence of bimodality (Stage C)")
        ax.legend()
        fig.tight_layout()
        return fig
    except Exception:
        return None


def _page7_summary(T_base, T_coex, rho_best, r_cut, L, D_coex):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.axis('off')
    text = (
        "CALIBRATION SUMMARY\n"
        "=" * 40 + "\n\n"
        f"  T_coex       = {T_coex:.4f}  (beta_target = {1/T_coex:.4f})\n"
        f"  T_base       = {T_base:.4f}  (beta_base   = {1/T_base:.4f})\n"
        f"  rho_best     = {rho_best:.4f}\n"
        f"  L_best       = {L:.4f} σ\n"
        f"  r_cut        = {r_cut:.4f} σ\n"
        f"  Ashman D     = {D_coex:.3f}  (>2 = clean bimodality)\n\n"
        "Training command:\n\n"
        f"  python coex_train.py \\\n"
        f"    --beta-base {1/T_base:.4f} \\\n"
        f"    --beta-target {1/T_coex:.4f} \\\n"
        f"    --box-length {L:.4f} \\\n"
        f"    --w-xz 2.0 --w-zx 1.0 \\\n"
        f"    --data-dir ./data/coex/ \\\n"
        f"    --n-samples 1000000"
    )
    ax.text(0.05, 0.95, text, transform=ax.transAxes,
            fontsize=10, va='top', family='monospace',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.6))
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def _count_transitions(op_series, midpoint=0.5):
    """Count how many times the mean OP crosses the midpoint."""
    above = op_series > midpoint
    return int(np.sum(np.diff(above.astype(int)) != 0))


def _plot_op_histogram(all_op, gmm, T_star, rho_best, D_val, save_path):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(all_op, bins=40, color='purple', edgecolor='white', lw=0.4,
            density=True, alpha=0.7)
    if gmm is not None:
        x = np.linspace(0, 1, 200).reshape(-1, 1)
        from scipy.stats import norm
        weights = gmm.weights_.ravel()
        means   = gmm.means_.ravel()
        stds    = np.sqrt(gmm.covariances_.ravel())
        for w, mu, s in zip(weights, means, stds):
            ax.plot(x.ravel(), w * norm.pdf(x.ravel(), mu, s),
                    '--', lw=1.5, alpha=0.8)
    ax.set_xlabel('OP (largest cluster fraction)')
    ax.set_ylabel('Density')
    ax.set_title(f'T*={T_star:.3f}  ρ*={rho_best:.2f}  Ashman D={D_val:.2f}')
    fig.tight_layout()
    fig.savefig(save_path, dpi=100)
    plt.close(fig)


def _plot_gr_pair(r_l, g_l, r_g, g_g, r_cut, T_star, save_path):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(r_l, g_l, 'r-', lw=1.5, label='liq-init')
    ax.plot(r_g, g_g, 'b-', lw=1.5, label='gas-init', alpha=0.8)
    ax.axhline(1.0, color='gray', ls='--', lw=1, alpha=0.5)
    ax.axvline(r_cut, color='black', ls=':', lw=1.5,
               label=f'r_cut={r_cut:.2f}')
    ax.set_xlabel('r / σ')
    ax.set_ylabel('g(r)')
    ax.set_title(f'g(r) at T*={T_star:.3f}')
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(save_path, dpi=100)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Thermodynamic calibration for 2D LJ coexistence")
    parser.add_argument('--out-dir', default='experiments/calibration',
                        help='Output directory')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--n-particles', type=int, default=32)
    parser.add_argument('--stage', choices=['A', 'B', 'C', 'D', 'E', 'all'],
                        default='all',
                        help='Run a specific stage or all (default)')
    args = parser.parse_args()

    N = args.n_particles
    D = 2
    key = jax.random.PRNGKey(args.seed)
    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    print(f"Calibration pipeline  N={N}  seed={args.seed}  out={out_dir}")
    print(f"[A1] use_lrc=False throughout")
    print(f"[A3] Move counts are single-particle moves (÷N = full sweeps)")

    run_all = (args.stage == 'all')

    # ---------- Stage A ----------
    rcut_json = os.path.join(out_dir, 'stage_a_rcut.json')
    if run_all or args.stage == 'A':
        key, subkey = jax.random.split(key)
        r_cut, _ = stage_a(N, D, out_dir, subkey, args.seed)
    elif os.path.exists(rcut_json):
        with open(rcut_json) as f:
            r_cut = json.load(f)['r_cut']
        print(f"\nLoaded r_cut={r_cut:.4f} from {rcut_json}")
    else:
        raise RuntimeError(f"Stage A not run and {rcut_json} not found.")

    # ---------- Stage B ----------
    b_csv = os.path.join(out_dir, 'coarse_sweep', 'summary.csv')
    if run_all or args.stage == 'B':
        key, subkey = jax.random.split(key)
        rho_best, coarse_rows = stage_b(N, D, r_cut, out_dir, subkey, args.seed)
    elif os.path.exists(b_csv):
        coarse_rows = []
        with open(b_csv) as f:
            for row in csv.DictReader(f):
                coarse_rows.append({
                    'T_star': float(row['T_star']),
                    'rho_star': float(row['rho_star']),
                    'op_gap': float(row['op_gap']),
                    'classification': row['classification'],
                })
        rho_best = max(set(r['rho_star'] for r in coarse_rows),
                       key=lambda rho: max(r['op_gap'] for r in coarse_rows
                                           if r['rho_star'] == rho))
        print(f"\nLoaded Stage B results. rho_best={rho_best:.2f}")
    else:
        raise RuntimeError(f"Stage B not run and {b_csv} not found.")

    # ---------- Stage C ----------
    c_json = os.path.join(out_dir, 'fine_sweep', 'results.json')
    if run_all or args.stage == 'C':
        key, subkey = jax.random.split(key)
        T_coex = stage_c(N, D, rho_best, r_cut, out_dir, subkey, coarse_rows)
    elif os.path.exists(c_json):
        with open(c_json) as f:
            T_coex = json.load(f)['T_coex']
        print(f"\nLoaded T_coex={T_coex:.4f} from {c_json}")
    else:
        raise RuntimeError(f"Stage C not run and {c_json} not found.")

    # ---------- Stage D ----------
    d_json = os.path.join(out_dir, 'stage_d', 'results.json')
    if run_all or args.stage == 'D':
        key, subkey = jax.random.split(key)
        T_base = stage_d(N, D, T_coex, rho_best, r_cut, out_dir, subkey)
    elif os.path.exists(d_json):
        with open(d_json) as f:
            T_base = json.load(f)['T_base']
        print(f"\nLoaded T_base={T_base:.4f} from {d_json}")
    else:
        raise RuntimeError(f"Stage D not run and {d_json} not found.")

    # ---------- Stage E ----------
    if run_all or args.stage == 'E':
        key, subkey = jax.random.split(key)
        stage_e(N, D, T_base, T_coex, rho_best, r_cut, out_dir, subkey)

    # ---------- Final summary ----------
    summary = {
        'N': N, 'D': D,
        'T_coex': T_coex, 'beta_target': round(1.0 / T_coex, 6),
        'T_base': T_base,  'beta_base':   round(1.0 / T_base, 6),
        'rho_best': rho_best,
        'L_best': round(float(np.sqrt(N / rho_best)), 6),
        'r_cut': r_cut,
    }
    summary_path = os.path.join(out_dir, 'final_params.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*60}")
    print("CALIBRATION COMPLETE")
    print(f"{'='*60}")
    print(json.dumps(summary, indent=2))
    print(f"\nFinal parameters saved: {summary_path}")
    print("\n*** STOP — review calibration_gallery.pdf before proceeding ***")


if __name__ == '__main__':
    main()
