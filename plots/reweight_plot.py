#!/usr/bin/env python3
"""Reweighting energy histogram plot for the WCA→LJ normalizing flow.

Produces a publication-quality plot with four distributions:
  - Reference: true LJ Boltzmann-distributed energies
  - Identity: WCA samples evaluated under LJ potential (no flow)
  - Transformed: flow-mapped WCA→LJ sample energies
  - Reweighted: bootstrap-reweighted histogram with error bars

Requires a trained flow checkpoint (params_final.pkl) and data files.

Usage:
  python reweight_plot.py
  python reweight_plot.py --checkpoint ./checkpoints/params_final.pkl --n-replicas 2 --samples-per-replica 5000
"""

import argparse
import functools
import os
import pickle
import time

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt

from config import PipelineConfig, FlowConfig
from physics import (
    make_lj_fn, make_wca_fn, ress, center_particle,
    generate_dataset,
)
from jax_pipeline import build_flow


# ---------------------------------------------------------------------------
# Bootstrap reweighting
# ---------------------------------------------------------------------------

def bootstrap_observable(energies, log_weights, ref_bins, n_bootstrap=100,
                         rng=None):
    """Compute a bootstrap-averaged weighted energy histogram.

    For each of n_bootstrap iterations:
      1. Resample indices uniformly WITH replacement.
      2. Build a weighted histogram (density=True) using the importance
         weights at those indices.
    Return the mean histogram (and std) across bootstrap iterations.

    Args:
        energies: (N,) energy values of transformed configurations.
        log_weights: (N,) log importance weights.
        ref_bins: (n_bins+1,) histogram bin edges.
        n_bootstrap: number of bootstrap iterations.
        rng: numpy random Generator (optional).

    Returns:
        hist_mean: (n_bins,) mean histogram across bootstrap iterations.
        hist_std: (n_bins,) std of histograms across iterations.
    """
    if rng is None:
        rng = np.random.default_rng()

    N = len(energies)
    n_bins = len(ref_bins) - 1
    hists = np.zeros((n_bootstrap, n_bins))

    for i in range(n_bootstrap):
        idx = rng.choice(N, size=N, replace=True)
        u_boot = energies[idx]
        lw_boot = log_weights[idx]
        # Numerically stable weights
        w = np.exp(lw_boot - lw_boot.max())
        h, _ = np.histogram(u_boot, bins=ref_bins, density=True, weights=w)
        hists[i] = h

    return hists.mean(axis=0), hists.std(axis=0, ddof=1)


# ---------------------------------------------------------------------------
# Batched flow evaluation
# ---------------------------------------------------------------------------

def batched_forward(apply_fn, params, z, batch_size, key):
    """Run forward flow in batches to avoid OOM.

    Args:
        apply_fn: haiku transformed apply function.
        params: flow parameters.
        z: (N, n_dof) source configurations.
        batch_size: max configs per batch.
        key: PRNG key.

    Returns:
        Tz: (N, n_dof) transformed configurations.
        logJ: (N,) log-det Jacobians.
    """
    N = z.shape[0]
    Tz_parts = []
    logJ_parts = []

    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        key, k = jax.random.split(key)
        tz_batch, lj_batch = apply_fn(params, k, z[start:end], inverse=False)
        Tz_parts.append(tz_batch)
        logJ_parts.append(lj_batch)

    return jnp.concatenate(Tz_parts, axis=0), jnp.concatenate(logJ_parts, axis=0)


def batched_energy(energy_fn, x, batch_size):
    """Evaluate energy function in batches.

    Args:
        energy_fn: JIT'd energy function (B, N*D) -> (B,).
        x: (N, N*D) configurations.
        batch_size: max configs per batch.

    Returns:
        (N,) energies.
    """
    parts = []
    for start in range(0, x.shape[0], batch_size):
        end = min(start + batch_size, x.shape[0])
        parts.append(energy_fn(x[start:end]))
    return jnp.concatenate(parts, axis=0)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Produce reweighting energy histogram plot for WCA→LJ flow")
    parser.add_argument("--model", choices=["lorenzo", "correti"], default="lorenzo",
                        help="Flow architecture used for training")
    parser.add_argument("--n-blocks", type=int, default=8,
                        help="Number of super-blocks (must match the trained model)")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to trained flow parameters (pickle)")
    parser.add_argument("--data-dir", type=str, default="./data")
    parser.add_argument("--output", type=str, default="./plots/reweighting_plot.png",
                        help="Output plot path")
    parser.add_argument("--n-replicas", type=int, default=10)
    parser.add_argument("--samples-per-replica", type=int, default=50000)
    parser.add_argument("--n-bootstrap", type=int, default=100)
    parser.add_argument("--n-bins", type=int, default=80)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--batch-size", type=int, default=5000,
                        help="Batch size for flow/energy evaluation")
    args = parser.parse_args()

    print("=" * 70)
    print("Reweighting Plot Generator")
    print("=" * 70)

    # --- Config ---
    if args.checkpoint is None:
        ckpt_dir = f"./checkpoints/{args.model}"
        args.checkpoint = os.path.join(ckpt_dir, "params_final.pkl")
    cfg = PipelineConfig(data_dir=args.data_dir,
                         flow=FlowConfig(model_type=args.model,
                                         n_blocks=args.n_blocks))
    lj_fn = make_lj_fn(cfg)
    wca_fn = make_wca_fn(cfg)

    N = cfg.system.n_particles
    D = cfg.system.dimensions
    L = cfg.system.box_length
    beta_source = cfg.beta_source
    beta_target = cfg.beta_target

    key = jax.random.PRNGKey(args.seed)

    # --- Load checkpoint ---
    print(f"Loading checkpoint: {args.checkpoint}")
    with open(args.checkpoint, "rb") as f:
        params = pickle.load(f)

    # --- Build flow ---
    print("Building flow...")
    init_fn, apply_fn = build_flow(cfg)
    # JIT the apply function
    apply_fn_jit = jax.jit(apply_fn, static_argnames=("inverse",))

    # --- Load reference LJ data ---
    lj_path = os.path.join(args.data_dir, "lj_configs.npy")
    print(f"Loading reference LJ data: {lj_path}")
    lj_data = jnp.array(np.load(lj_path))

    # Center reference data on particle 0
    lj_centered = center_particle(lj_data, 0, N, D, L)

    print(f"  Reference configs: {lj_centered.shape[0]}")
    print(f"Computing reference LJ energies...")
    U_ref = np.array(batched_energy(lj_fn, lj_centered, args.batch_size))
    print(f"  U_ref range: [{U_ref.min():.2f}, {U_ref.max():.2f}], "
          f"mean: {U_ref.mean():.2f}")

    # --- Replica generation & transformation ---
    print(f"\nGenerating {args.n_replicas} replicas x "
          f"{args.samples_per_replica} samples each")
    print(f"  (Full independent MCMC per replica)\n")

    all_U_identity = []
    all_U_transformed = []
    all_log_w = []
    all_ress = []

    for r in range(args.n_replicas):
        t0 = time.time()
        print(f"--- Replica {r+1}/{args.n_replicas} ---")

        # 1. Generate independent WCA samples via full MCMC
        key, k_mcmc = jax.random.split(key)
        print(f"  MCMC sampling {args.samples_per_replica} WCA configs...")
        z = generate_dataset(
            wca_fn, beta_source, args.samples_per_replica, cfg, k_mcmc)

        # Center on particle 0
        z = center_particle(z, 0, N, D, L)

        # 2. Identity baseline: WCA configs evaluated under LJ potential
        print("  Computing identity (WCA→LJ) energies...")
        U_identity = np.array(batched_energy(lj_fn, z, args.batch_size))
        all_U_identity.append(U_identity)

        # 3. Forward transform through flow
        print("  Running flow forward (z→x)...")
        key, k_flow = jax.random.split(key)
        Tz, logJ_zx = batched_forward(
            apply_fn_jit, params, z, args.batch_size, k_flow)

        # 4. Energies of transformed configs
        print("  Computing transformed energies...")
        U_Tz = np.array(batched_energy(lj_fn, Tz, args.batch_size))
        U_z_wca = np.array(batched_energy(wca_fn, z, args.batch_size))
        logJ_np = np.array(logJ_zx)

        all_U_transformed.append(U_Tz)

        # 5. Log importance weights
        #    log_w = -beta_target * U_LJ(Tz) + beta_source * U_WCA(z) + logJ_zx
        log_w = (-beta_target * U_Tz
                 + beta_source * U_z_wca
                 + logJ_np)
        all_log_w.append(log_w)

        # 6. RESS
        r_val = float(ress(jnp.array(log_w)))
        all_ress.append(r_val)

        dt = time.time() - t0
        print(f"  U_transformed: [{U_Tz.min():.2f}, {U_Tz.max():.2f}], "
              f"mean: {U_Tz.mean():.2f}")
        print(f"  RESS: {r_val:.4f}  ({dt:.1f}s)\n")

    print(f"Mean RESS across replicas: {np.mean(all_ress):.4f} "
          f"(+/- {np.std(all_ress, ddof=1):.4f})")

    # --- Global bin edges ---
    U_id_all = np.concatenate(all_U_identity)
    U_tr_all = np.concatenate(all_U_transformed)

    global_min = min(U_ref.min(), U_id_all.min(), U_tr_all.min())
    global_max = max(U_ref.max(), U_id_all.max(), U_tr_all.max())
    # Trim extreme identity tails (they can stretch very far)
    # Use 0.1th and 99.9th percentiles to avoid extreme outliers dominating
    pct_low = min(np.percentile(U_ref, 0.1),
                  np.percentile(U_tr_all, 0.1))
    pct_high = max(np.percentile(U_ref, 99.9),
                   np.percentile(U_id_all, 99.9),
                   np.percentile(U_tr_all, 99.9))
    # Use the wider of (percentile range, ref range) for bins
    bin_lo = min(global_min, pct_low)
    bin_hi = max(global_max, pct_high)
    ref_bins = np.linspace(bin_lo, bin_hi, args.n_bins + 1)
    mids = 0.5 * (ref_bins[:-1] + ref_bins[1:])

    # --- Reference histogram ---
    print("\nComputing reference histogram...")
    hist_ref, _ = np.histogram(U_ref, bins=ref_bins, density=True)

    # --- Identity histogram (mean across replicas) ---
    print("Computing identity histograms...")
    hists_identity = []
    for U_id in all_U_identity:
        h, _ = np.histogram(U_id, bins=ref_bins, density=True)
        hists_identity.append(h)
    hists_identity = np.array(hists_identity)
    hist_identity_mean = hists_identity.mean(axis=0)

    # --- Transformed histogram (mean across replicas) ---
    print("Computing transformed histograms...")
    hists_transformed = []
    for U_tr in all_U_transformed:
        h, _ = np.histogram(U_tr, bins=ref_bins, density=True)
        hists_transformed.append(h)
    hists_transformed = np.array(hists_transformed)
    hist_transformed_mean = hists_transformed.mean(axis=0)

    # --- Bootstrap reweighted histogram ---
    print(f"Bootstrap reweighting ({args.n_bootstrap} iterations "
          f"x {args.n_replicas} replicas)...")
    rng = np.random.default_rng(args.seed + 999)
    hists_bootstrap = []
    for r in range(args.n_replicas):
        h_mean, h_std = bootstrap_observable(
            all_U_transformed[r],
            all_log_w[r],
            ref_bins,
            n_bootstrap=args.n_bootstrap,
            rng=rng,
        )
        hists_bootstrap.append(h_mean)

    hists_bootstrap = np.array(hists_bootstrap)  # (n_replicas, n_bins)
    hist_reweighted_mean = hists_bootstrap.mean(axis=0)
    hist_reweighted_err = (hists_bootstrap.std(axis=0, ddof=1)
                           / np.sqrt(args.n_replicas))

    # --- Mean energies ---
    # Weighted average of bin centers by histogram values
    def weighted_mean_energy(hist, bin_mids):
        total = hist.sum()
        if total > 0:
            return np.sum(bin_mids * hist) / total
        return np.nan

    U_ref_mean = weighted_mean_energy(hist_ref, mids)
    U_reweighted_mean = weighted_mean_energy(hist_reweighted_mean, mids)

    print(f"\n  <U_B> Reference:    {U_ref_mean:.2f}")
    print(f"  <U_B> Reweighted:   {U_reweighted_mean:.2f}")

    # --- Plot ---
    print(f"\nGenerating plot -> {args.output}")

    fig, ax = plt.subplots(1, 1, figsize=(8, 5), dpi=300)
    bar_width = mids[1] - mids[0]

    # Reference
    ax.bar(mids, hist_ref, width=bar_width, alpha=0.5,
           color="C0", label="Reference (LJ)", edgecolor="none")

    # Identity
    ax.step(np.append(ref_bins[:-1], ref_bins[-1]),
            np.append(hist_identity_mean, hist_identity_mean[-1]),
            where="post", color="gray", linewidth=1.2,
            linestyle="--", label="Identity (WCA as LJ)")

    # Transformed (no reweighting)
    ax.step(np.append(ref_bins[:-1], ref_bins[-1]),
            np.append(hist_transformed_mean, hist_transformed_mean[-1]),
            where="post", color="C1", linewidth=1.2,
            label="Transformed (flow)")

    # Reweighted (bootstrap)
    ax.bar(mids, hist_reweighted_mean, width=bar_width, alpha=0.6,
           color="C3", label="Reweighted (bootstrap)",
           edgecolor="none",
           yerr=hist_reweighted_err, capsize=1.5, error_kw={"linewidth": 0.8})

    # Mean energy vertical lines
    ax.axvline(U_ref_mean, color="C0", linestyle=":", linewidth=1,
               label=f"$\\langle U_B \\rangle$ ref = {U_ref_mean:.1f}")
    ax.axvline(U_reweighted_mean, color="C3", linestyle=":", linewidth=1,
               label=f"$\\langle U_B \\rangle$ rew = {U_reweighted_mean:.1f}")

    ax.set_xlabel("Energy $U_{LJ}$", fontsize=12)
    ax.set_ylabel("Probability density", fontsize=12)
    ax.set_title(f"WCA $\\to$ LJ Reweighting  "
                 f"(N={N}, {args.n_replicas}x{args.samples_per_replica} samples, "
                 f"RESS={np.mean(all_ress):.3f})", fontsize=11)
    ax.legend(fontsize=8, frameon=False, loc="upper left")

    # Trim x-axis to relevant range
    ax.set_xlim(np.percentile(U_ref, 0.05) - 5,
                np.percentile(U_ref, 99.95) + 15)
    ax.set_ylim(bottom=0)

    plt.tight_layout()
    plt.savefig(args.output, dpi=300, bbox_inches="tight")
    print(f"Saved: {args.output}")
    plt.close()

    # --- Summary ---
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    print(f"  Replicas:            {args.n_replicas}")
    print(f"  Samples/replica:     {args.samples_per_replica}")
    print(f"  Total raw samples:   {args.n_replicas * args.samples_per_replica}")
    print(f"  Bootstrap iters:     {args.n_bootstrap}")
    print(f"  Histogram bins:      {args.n_bins}")
    print(f"  Mean RESS (z→x):     {np.mean(all_ress):.4f} "
          f"+/- {np.std(all_ress, ddof=1):.4f}")
    for r, rv in enumerate(all_ress):
        print(f"    Replica {r}: RESS = {rv:.4f}")
    print(f"  <U_B> Reference:     {U_ref_mean:.2f}")
    print(f"  <U_B> Reweighted:    {U_reweighted_mean:.2f}")
    print(f"  Output:              {args.output}")
    print("=" * 70)


if __name__ == "__main__":
    main()
