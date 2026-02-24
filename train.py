#!/usr/bin/env python3
"""Entry point for the NF training pipeline."""

import argparse
from config import PipelineConfig, FlowConfig, TrainConfig, MCMCConfig


def main():
    parser = argparse.ArgumentParser(
        description="Train a WCA->LJ normalizing flow (32 particles, 2D)")
    parser.add_argument("--backend", choices=["jax", "torch"], default="jax")
    parser.add_argument("--generate-data", action="store_true",
                        help="Only generate MCMC data, then exit")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--n-samples", type=int, default=None,
                        help="Number of MCMC samples to generate")
    parser.add_argument("--model", choices=["lorenzo", "correti"], default="lorenzo",
                        help="Flow architecture: lorenzo or correti (both 32 couplings with default n_blocks=8)")
    parser.add_argument("--n-blocks", type=int, default=None,
                        help="Number of super-blocks (default 8). Total RQS layers = n_blocks*4 (lorenzo) or n_blocks*4 (correti)")
    parser.add_argument("--milestones-epochs", type=int, nargs=2, default=None,
                        metavar=("EP1", "EP2"),
                        help="Epoch milestones for LR decay, e.g. --milestones-epochs 15 22 "
                             "(default: 150 225)")
    parser.add_argument("--save-dir", type=str, default=None)
    parser.add_argument("--data-dir", type=str, default="./data")
    args = parser.parse_args()

    if args.save_dir is None:
        args.save_dir = f"./experiments/{args.model}"

    # Build config with overrides
    train_kwargs = {}
    if args.n_epochs is not None:
        train_kwargs["n_epochs"] = args.n_epochs
    if args.batch_size is not None:
        train_kwargs["batch_size"] = args.batch_size
    if args.lr is not None:
        train_kwargs["lr"] = args.lr
    if args.milestones_epochs is not None:
        train_kwargs["milestones_epochs"] = tuple(args.milestones_epochs)

    mcmc_kwargs = {}
    if args.n_samples is not None:
        mcmc_kwargs["n_samples"] = args.n_samples
        mcmc_kwargs["n_cached"] = int(args.n_samples * 0.9)

    cfg = PipelineConfig(
        backend=args.backend,
        seed=args.seed,
        save_dir=args.save_dir,
        data_dir=args.data_dir,
        flow=FlowConfig(
            model_type=args.model,
            **({"n_blocks": args.n_blocks} if args.n_blocks is not None else {}),
        ),
        train=TrainConfig(**train_kwargs),
        mcmc=MCMCConfig(**mcmc_kwargs),
    )

    if args.generate_data:
        import jax  # noqa: E402
        from jax_pipeline import load_or_generate_data
        key = jax.random.PRNGKey(cfg.seed)
        load_or_generate_data(cfg, key)
        print("Data generation complete.")
        return

    if args.backend == "jax":
        from jax_pipeline import train
    else:
        raise NotImplementedError("Torch backend not yet implemented")

    train(cfg)


if __name__ == "__main__":
    main()
