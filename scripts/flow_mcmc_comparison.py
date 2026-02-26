"""Flow-augmented MCMC comparison.

Loads trained flow from checkpoints/coex/params_final.pkl and runs a
head-to-head comparison at T*=0.36, rho*=0.30:

  WITHOUT flow: 8 gas-init + 8 liq-init chains, 500K local MCMC moves each.
  WITH flow:    same init, 495K local moves + 1 flow proposal per 990 local moves
                (≈500 proposals per chain, same local-move budget).

Records U/N every 5K local moves (100 checkpoints per chain).
Computes mixing time: first checkpoint where gas-init U/N <= liq_mean + 0.10.

Outputs:
    experiments/coex_comparison/timeseries_no_flow.png
    experiments/coex_comparison/timeseries_with_flow.png
    experiments/coex_comparison/mixing_time_comparison.png
    experiments/coex_comparison/results.npz
    experiments/coex_comparison/results_summary.txt

Usage:
    python scripts/flow_mcmc_comparison.py
    python scripts/flow_mcmc_comparison.py --params checkpoints/coex/params_epoch_100.pkl
"""

import argparse
import functools
import os
import pickle
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from physics import make_lj_fn, run_mcmc, center_particle
from diagnostics import make_liquid_init, make_gas_init, calibrate_step_size
from coex_config import make_coex_config, T_BASE, T_TARGET, N, D, L, R_CUT_OP
from coex_pipeline import build_flow

# ── constants ─────────────────────────────────────────────────────────────────
N_CHAINS     = 8        # per init type (liq + gas)
N_LOCAL      = 500_000  # total local MCMC moves per chain
CHUNK        = 5_000    # local moves between U/N recordings (100 checkpoints)
FLOW_EVERY   = 990      # local moves between each flow proposal attempt
                         # → ≈500 proposals per chain in 495K local moves
MIX_THRESH   = 0.10     # gas chain "mixed" when U/N <= liq_mean + MIX_THRESH
OUT_DIR      = 'experiments/coex_comparison'
PARAMS_PATH  = 'checkpoints/coex/params_final.pkl'
BASE_CACHE   = 'data/coex/base_configs.npy'


# ── flow proposal step (JIT'd) ────────────────────────────────────────────────

def _make_flow_proposal_fn(apply_fn, energy_fn, beta_target, beta_base):
    """Return a JIT'd function that makes one round of flow proposals.

    One proposal per chain simultaneously.  Acceptance uses the full
    NF-MCMC ratio (Stimper et al. Eq. 6):
        log α = -βt*(U(y)-U(x)) - βb*(U(z_x)-U(z_new)) + logdet_inv_x + logdet_fwd_new
    """
    @functools.partial(jax.jit)
    def propose(configs, energies, params, z_new, rng):
        r1, r2 = jax.random.split(rng)
        # Decode base samples → proposed target configs
        y, log_det_fwd = apply_fn(params, r1, z_new, inverse=False)
        # Encode current configs → latent
        z_x, log_det_inv_x = apply_fn(params, r2, configs, inverse=True)

        E_y    = energy_fn(y)
        E_z_x  = energy_fn(z_x)
        E_z_new = energy_fn(z_new)

        # Hard reject unphysical proposals (overlap)
        finite_mask = jnp.isfinite(E_y) & (E_y < 1e6)

        log_acc_raw = (
            -beta_target * (E_y - energies)
            - beta_base  * (E_z_x - E_z_new)
            + log_det_inv_x + log_det_fwd
        )
        log_acc = jnp.where(finite_mask, log_acc_raw, -jnp.inf)
        log_u   = jnp.log(jax.random.uniform(r1, shape=log_acc.shape))
        accepted = log_u < log_acc

        new_configs  = jnp.where(accepted[:, None], y,     configs)
        new_energies = jnp.where(accepted,           E_y,  energies)
        return new_configs, new_energies, accepted

    return propose


# ── run one set of chains ─────────────────────────────────────────────────────

def run_chains(init_configs, energy_fn, beta, step_size, N, D, L,
               key, n_local, chunk,
               apply_fn=None, params=None, base_cache=None,
               flow_every=None, beta_base=None, beta_target=None,
               flow_propose_fn=None):
    """Run chains with or without flow proposals.

    Returns:
        energies_hist: (n_chains, n_checkpoints) U/N timeseries
        flow_accept:   scalar mean acceptance rate (NaN if no flow)
    """
    configs  = jnp.array(init_configs)
    energies = energy_fn(configs)

    n_chunks = n_local // chunk
    use_flow = (apply_fn is not None and params is not None
                and base_cache is not None)

    energies_hist = []
    total_flow_proposals = 0
    total_flow_accepted  = 0

    for ci in range(n_chunks):
        # ── local MCMC ────────────────────────────────────────────────────────
        # If using flow: interleave flow proposals every `flow_every` local moves
        if use_flow:
            local_done = 0
            local_remaining = chunk
            while local_remaining > 0:
                step = min(flow_every, local_remaining)
                key, k_local = jax.random.split(key)
                configs, energies = run_mcmc(
                    configs, energy_fn, beta, step,
                    k_local, step_size, L, N, D)
                local_done      += step
                local_remaining -= step

                # Flow proposal after every flow_every local moves
                if local_done % flow_every == 0:
                    key, k_samp, k_prop = jax.random.split(key, 3)
                    # Sample one z_new per chain from base cache
                    idx   = jax.random.randint(
                        k_samp, (configs.shape[0],), 0, base_cache.shape[0])
                    z_new = base_cache[idx]
                    configs, energies, accepted = flow_propose_fn(
                        configs, energies, params, z_new, k_prop)
                    total_flow_proposals += configs.shape[0]
                    total_flow_accepted  += int(accepted.sum())
        else:
            key, k_local = jax.random.split(key)
            configs, energies = run_mcmc(
                configs, energy_fn, beta, chunk,
                k_local, step_size, L, N, D)

        energies_hist.append(np.array(energies) / N)

    energies_hist = np.array(energies_hist).T  # (n_chains, n_checkpoints)

    flow_acc = (total_flow_accepted / total_flow_proposals
                if total_flow_proposals > 0 else float('nan'))
    return energies_hist, flow_acc


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--params',      default=PARAMS_PATH)
    parser.add_argument('--base-cache',  default=BASE_CACHE)
    parser.add_argument('--out-dir',     default=OUT_DIR)
    parser.add_argument('--n-local',     type=int, default=N_LOCAL)
    parser.add_argument('--seed',        type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # ── Config + energy function ───────────────────────────────────────────────
    cfg       = make_coex_config()
    lj_fn     = make_lj_fn(cfg)
    step_size = cfg.mcmc.step_size
    beta_t    = cfg.beta_target
    beta_b    = cfg.beta_source

    key = jax.random.PRNGKey(args.seed)
    print(f'Flow MCMC comparison  N={N}  ρ*={cfg.system.rho}  '
          f'T*={T_TARGET:.2f}  L={L:.3f}')
    print(f'  {N_CHAINS} liq-init + {N_CHAINS} gas-init chains')
    print(f'  {args.n_local//1000}K local moves, chunk={CHUNK}, '
          f'flow_every={FLOW_EVERY}')

    # ── Load trained flow ──────────────────────────────────────────────────────
    if not os.path.exists(args.params):
        raise FileNotFoundError(
            f'Flow params not found: {args.params}\n'
            f'Run training first: python coex_train.py')
    print(f'\nLoading flow params: {args.params}')
    with open(args.params, 'rb') as f:
        params = pickle.load(f)

    _, apply_fn = build_flow(cfg)

    # Quick invertibility check [A5]
    print('  Checking flow invertibility…', end=' ', flush=True)
    key, k_check = jax.random.split(key)
    dummy = jnp.zeros((4, N * D))
    z_check, _ = apply_fn(params, k_check, dummy, inverse=True)
    x_rec, _   = apply_fn(params, k_check, z_check, inverse=False)
    max_err = float(jnp.abs(x_rec - dummy).max())
    print(f'max recon err = {max_err:.2e}', end='')
    if max_err > 0.1:
        print(' WARNING: large reconstruction error — flow may be poorly trained')
    else:
        print(' OK')

    # ── Load base cache ────────────────────────────────────────────────────────
    if not os.path.exists(args.base_cache):
        raise FileNotFoundError(
            f'Base cache not found: {args.base_cache}\n'
            f'Run: python coex_train.py --generate-data')
    print(f'Loading base cache: {args.base_cache}')
    base_cache = jnp.array(np.load(args.base_cache))
    print(f'  Base cache shape: {base_cache.shape}')

    # ── Build JIT'd flow proposal ──────────────────────────────────────────────
    flow_propose_fn = _make_flow_proposal_fn(
        apply_fn, lj_fn, beta_t, beta_b)
    # Warm up JIT
    key, k_warm = jax.random.split(key)
    dummy_cfg = jnp.zeros((2, N * D))
    dummy_z   = base_cache[:2]
    _ = flow_propose_fn(dummy_cfg, jnp.zeros(2), params, dummy_z, k_warm)
    print('  Flow proposal JIT compiled.')

    # ── Initial configurations ─────────────────────────────────────────────────
    key, kl, kg = jax.random.split(key, 3)
    liq_inits = jnp.stack([make_liquid_init(N, D, L, rng_key=k)
                            for k in jax.random.split(kl, N_CHAINS)])
    gas_inits = jnp.stack([make_gas_init(N, D, L, rng_key=k)
                            for k in jax.random.split(kg, N_CHAINS)])

    # ── Calibrate step size ────────────────────────────────────────────────────
    key, k1, k2 = jax.random.split(key, 3)
    step_size, rates = calibrate_step_size(liq_inits, lj_fn, beta_t, N, D, L, k2)
    print(f'  step_size={step_size:.4f}  acc≈{rates[step_size]:.3f}')

    # ── Run all four conditions ────────────────────────────────────────────────
    results = {}
    for label, inits, use_flow in [
        ('liq_no_flow', liq_inits, False),
        ('gas_no_flow', gas_inits, False),
        ('liq_with_flow', liq_inits, True),
        ('gas_with_flow', gas_inits, True),
    ]:
        flow_kw = dict(
            apply_fn=apply_fn, params=params, base_cache=base_cache,
            flow_every=FLOW_EVERY, beta_base=beta_b, beta_target=beta_t,
            flow_propose_fn=flow_propose_fn,
        ) if use_flow else {}

        print(f'\nRunning {label}  ({args.n_local//1000}K moves × {N_CHAINS} chains)…',
              flush=True)
        key, k_run = jax.random.split(key)
        t0 = time.time()
        E_hist, acc = run_chains(
            inits, lj_fn, beta_t, step_size, N, D, L, k_run,
            n_local=args.n_local, chunk=CHUNK,
            **flow_kw)
        dt = time.time() - t0
        results[label] = {'E': E_hist, 'acc': acc}
        print(f'  done {dt:.0f}s  '
              f'final U/N={E_hist[:,-1].mean():.3f}  '
              + (f'flow_acc={acc:.3f}' if use_flow else 'no flow'),
              flush=True)

    # ── Compute mixing times ───────────────────────────────────────────────────
    sample_times = np.arange(1, args.n_local // CHUNK + 1) * CHUNK

    def mixing_time(E_gas, E_liq):
        """First sample where gas U/N <= liq_mean + MIX_THRESH."""
        liq_mean = E_liq[:, -20:].mean()
        target   = liq_mean + MIX_THRESH
        n_chains, n_samples = E_gas.shape
        times = []
        for c in range(n_chains):
            reached = np.where(E_gas[c] <= target)[0]
            times.append(float(sample_times[reached[0]]) if len(reached) else np.inf)
        return np.array(times), liq_mean

    tau_no_flow,   lm_nf = mixing_time(results['gas_no_flow']['E'],
                                        results['liq_no_flow']['E'])
    tau_with_flow, lm_wf = mixing_time(results['gas_with_flow']['E'],
                                        results['liq_with_flow']['E'])

    print('\n─── Mixing time results ───')
    for label, tau in [('Without flow', tau_no_flow),
                        ('With flow',   tau_with_flow)]:
        finite = tau[np.isfinite(tau)]
        print(f'  {label}: {len(finite)}/{len(tau)} chains mixed'
              + (f'  median τ={np.median(finite)/1000:.0f}K'
                 if len(finite) else '  NONE'))

    # ── Save NPZ ───────────────────────────────────────────────────────────────
    np.savez(os.path.join(args.out_dir, 'results.npz'),
             sample_times=sample_times,
             liq_no_flow=results['liq_no_flow']['E'],
             gas_no_flow=results['gas_no_flow']['E'],
             liq_with_flow=results['liq_with_flow']['E'],
             gas_with_flow=results['gas_with_flow']['E'],
             tau_no_flow=tau_no_flow,
             tau_with_flow=tau_with_flow,
             flow_acc_liq=results['liq_with_flow']['acc'],
             flow_acc_gas=results['gas_with_flow']['acc'])

    # ── Summary text ──────────────────────────────────────────────────────────
    summary_lines = [
        f'Flow MCMC Comparison  T*={T_TARGET}  rho*={cfg.system.rho}  N={N}',
        f'Local moves: {args.n_local}  Flow proposals per chain: '
        f'{args.n_local // FLOW_EVERY if FLOW_EVERY else 0}',
        f'Flow acceptance rate (gas-init): {results["gas_with_flow"]["acc"]:.3f}',
        f'Flow acceptance rate (liq-init): {results["liq_with_flow"]["acc"]:.3f}',
        '',
        'Gas-init mixing time (U/N within 0.10 of liq_mean):',
        f'  Without flow: {sum(np.isfinite(tau_no_flow))}/{len(tau_no_flow)} mixed'
        + (f'  median {np.median(tau_no_flow[np.isfinite(tau_no_flow)])/1000:.0f}K'
           if np.any(np.isfinite(tau_no_flow)) else '  NONE'),
        f'  With flow:    {sum(np.isfinite(tau_with_flow))}/{len(tau_with_flow)} mixed'
        + (f'  median {np.median(tau_with_flow[np.isfinite(tau_with_flow)])/1000:.0f}K'
           if np.any(np.isfinite(tau_with_flow)) else '  NONE'),
    ]
    summary = '\n'.join(summary_lines)
    print('\n' + summary)
    with open(os.path.join(args.out_dir, 'results_summary.txt'), 'w') as f:
        f.write(summary + '\n')

    # ── Plots ─────────────────────────────────────────────────────────────────
    x = sample_times / 1000  # k-moves

    for tag, E_liq, E_gas, acc in [
        ('no_flow',   results['liq_no_flow']['E'],   results['gas_no_flow']['E'],   None),
        ('with_flow', results['liq_with_flow']['E'], results['gas_with_flow']['E'],
         results['gas_with_flow']['acc']),
    ]:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

        liq_mean = E_liq[:, -20:].mean()

        for ci in range(E_liq.shape[0]):
            axes[0].plot(x, E_liq[ci], color='crimson', alpha=0.5, lw=1.2,
                         label='liq-init' if ci == 0 else None)
        for ci in range(E_gas.shape[0]):
            axes[0].plot(x, E_gas[ci], color='steelblue', alpha=0.5, lw=1.2,
                         label='gas-init' if ci == 0 else None)
        axes[0].axhline(liq_mean, color='crimson', ls='--', lw=1.5,
                        label=f'liq mean={liq_mean:.3f}')
        axes[0].axhline(liq_mean + MIX_THRESH, color='grey', ls=':', lw=1,
                        label=f'mix threshold (±{MIX_THRESH})')
        axes[0].set_xlabel('Moves (×10³)'); axes[0].set_ylabel('U/N')
        axes[0].set_title(f'Energy timeseries {"with" if acc else "without"} flow')
        axes[0].legend(fontsize=7)

        # Mixing time scatter
        tau_arr = tau_with_flow if acc else tau_no_flow
        finite  = tau_arr[np.isfinite(tau_arr)]
        inf_y   = args.n_local * 1.05 / 1000
        axes[1].scatter(np.arange(len(finite)), finite / 1000,
                        s=60, c='steelblue', zorder=3)
        inf_ct = np.sum(np.isinf(tau_arr))
        if inf_ct:
            axes[1].scatter(np.arange(len(finite), len(tau_arr)),
                            np.full(inf_ct, inf_y * 0.98),
                            s=80, marker='^', c='firebrick', zorder=3,
                            label='Never mixed')
            axes[1].axhline(inf_y, color='firebrick', ls=':', alpha=0.5)
        axes[1].set_xlabel('Chain index')
        axes[1].set_ylabel('Mixing time (×10³ moves)')
        title = f'Mixing time {"with" if acc else "without"} flow'
        if acc:
            title += f'  (flow acc={acc:.3f})'
        axes[1].set_title(title)
        if inf_ct:
            axes[1].legend(fontsize=8)

        fig.suptitle(
            f'T*={T_TARGET}  ρ*={cfg.system.rho}  N={N}  '
            f'{"with" if acc else "without"} flow proposals',
            fontsize=10)
        fig.tight_layout()
        out_path = os.path.join(args.out_dir, f'timeseries_{tag}.png')
        fig.savefig(out_path, dpi=130, bbox_inches='tight')
        plt.close(fig)
        print(f'Saved: {out_path}')

    # ── Side-by-side mixing time comparison ──────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 4.5))
    inf_y = args.n_local * 1.05 / 1000
    for i, (tau, color, label) in enumerate([
        (tau_no_flow,   'steelblue', 'Without flow'),
        (tau_with_flow, 'darkorange', 'With flow'),
    ]):
        finite = tau[np.isfinite(tau)]
        jitter = (np.random.default_rng(i).uniform(-0.05, 0.05, len(finite)))
        ax.scatter(np.full(len(finite), i) + jitter, finite / 1000,
                   s=60, c=color, alpha=0.8, zorder=3, label=label)
        inf_ct = np.sum(np.isinf(tau))
        if inf_ct:
            ax.scatter(np.full(inf_ct, i),
                       np.full(inf_ct, inf_y * 0.97),
                       s=90, marker='^', c='firebrick', alpha=0.8, zorder=3)
        if len(finite):
            ax.hlines(np.median(finite / 1000), i - 0.2, i + 0.2,
                      colors=color, linewidths=2.5, zorder=4)

    ax.axhline(inf_y, color='firebrick', ls=':', lw=1, alpha=0.5)
    ax.text(1.55, inf_y, '∞', va='center', fontsize=12, color='firebrick')
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['Without flow', 'With flow'], fontsize=11)
    ax.set_ylabel('Mixing time (×10³ moves)')
    ax.set_title(f'Mixing time comparison  T*={T_TARGET}  ρ*={cfg.system.rho}  N={N}\n'
                 f'(horizontal bar = median; ▲ = never mixed in {args.n_local//1000}K moves)')
    ax.legend(fontsize=9)
    fig.tight_layout()
    comp_path = os.path.join(args.out_dir, 'mixing_time_comparison.png')
    fig.savefig(comp_path, dpi=130, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {comp_path}')
    print('\nComparison complete.')


if __name__ == '__main__':
    main()
