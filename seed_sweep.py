#!/usr/bin/env python3
"""Seed sweep: run short training probes for multiple seeds and both backends.

Reports ESS at each epoch so we can identify seeds that give comparable
ESS for lorenzo and correti backends.
"""

import argparse
import json
import os
import sys
import time

# Ensure we can import from the project
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import PipelineConfig, FlowConfig, TrainConfig, MCMCConfig


def run_probe(model_type, seed, n_epochs, save_dir, data_dir):
    """Run a short training probe and return metrics."""
    import jax
    import jax.numpy as jnp
    import numpy as np

    from jax_pipeline import train

    cfg = PipelineConfig(
        backend="jax",
        seed=seed,
        save_dir=save_dir,
        data_dir=data_dir,
        flow=FlowConfig(model_type=model_type),
        train=TrainConfig(
            n_epochs=n_epochs,
            n_dump=1,      # validate every epoch
            n_save=0,       # no intermediate checkpoints
        ),
    )

    # Monkey-patch to skip final checkpoint save (we don't need it for probes)
    params, metrics_log = train(cfg)

    return metrics_log


def main():
    parser = argparse.ArgumentParser(description="Seed sweep for ESS comparison")
    parser.add_argument("--seeds", type=int, nargs="+",
                        default=[42, 123, 7, 2024, 0],
                        help="Seeds to test")
    parser.add_argument("--backends", nargs="+",
                        default=["lorenzo", "correti"],
                        choices=["lorenzo", "correti"],
                        help="Backends to test")
    parser.add_argument("--n-epochs", type=int, default=10,
                        help="Epochs per probe")
    parser.add_argument("--data-dir", type=str, default="./data")
    parser.add_argument("--output", type=str, default="./seed_sweep_results.json",
                        help="Output JSON file")
    args = parser.parse_args()

    results = {}
    total_runs = len(args.backends) * len(args.seeds)
    run_idx = 0

    for backend in args.backends:
        results[backend] = {}
        for seed in args.seeds:
            run_idx += 1
            tag = f"{backend}_seed{seed}"
            save_dir = f"./checkpoints/sweep_{tag}"
            os.makedirs(save_dir, exist_ok=True)

            print(f"\n{'='*70}")
            print(f"Run {run_idx}/{total_runs}: {backend} seed={seed}")
            print(f"{'='*70}")

            t0 = time.time()
            try:
                metrics = run_probe(
                    model_type=backend,
                    seed=seed,
                    n_epochs=args.n_epochs,
                    save_dir=save_dir,
                    data_dir=args.data_dir,
                )
                elapsed = time.time() - t0

                results[backend][str(seed)] = {
                    "metrics": metrics,
                    "elapsed_s": elapsed,
                }

                # Print summary for this run
                if metrics:
                    last = metrics[-1]
                    print(f"\n  Final epoch {last['epoch']}: "
                          f"RESS_xz={last['ress_xz']:.4f} "
                          f"RESS_zx={last['ress_zx']:.4f} "
                          f"val_xz={last['val_loss_xz']:.3f} "
                          f"val_zx={last['val_loss_zx']:.3f}")

            except Exception as e:
                print(f"  FAILED: {e}")
                results[backend][str(seed)] = {"error": str(e)}

    # Save all results
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {args.output}")

    # Print comparison table
    print(f"\n{'='*70}")
    print("SEED SWEEP SUMMARY")
    print(f"{'='*70}")

    # Reference: original correti at convergence
    print(f"\nReference (original correti PyTorch, epoch 250):")
    print(f"  RESS_xz=0.0560  RESS_zx=0.0734  val_xz=4.222  val_zx=-48.826")

    for backend in args.backends:
        print(f"\n--- {backend} backend ---")
        print(f"{'Seed':>6s}  {'RESS_xz':>8s}  {'RESS_zx':>8s}  "
              f"{'val_xz':>8s}  {'val_zx':>8s}  {'Time':>6s}")
        for seed in args.seeds:
            key = str(seed)
            if key in results[backend] and "metrics" in results[backend][key]:
                m = results[backend][key]["metrics"][-1]
                t = results[backend][key]["elapsed_s"]
                print(f"{seed:6d}  {m['ress_xz']:8.4f}  {m['ress_zx']:8.4f}  "
                      f"{m['val_loss_xz']:8.3f}  {m['val_loss_zx']:8.3f}  "
                      f"{t:5.0f}s")
            else:
                print(f"{seed:6d}  {'FAILED':>8s}")


if __name__ == "__main__":
    main()
