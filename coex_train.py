#!/usr/bin/env python3
"""Train a normalising flow for LJ gas-liquid coexistence.

Usage:
    python coex_train.py
    python coex_train.py --n-epochs 100 --batch-size 256
    python coex_train.py --generate-data   # only generate MCMC data, then exit
"""

import argparse

from coex_config import make_coex_config
from coex_pipeline import load_or_generate_coex_data, train_coex


def main():
    parser = argparse.ArgumentParser(
        description='Train LJ coexistence normalising flow (N=128, T*=0.36/0.50)')
    parser.add_argument('--generate-data', action='store_true',
                        help='Only generate and cache MCMC data, then exit')
    parser.add_argument('--seed',      type=int,   default=42)
    parser.add_argument('--n-epochs',  type=int,   default=None)
    parser.add_argument('--batch-size',type=int,   default=None)
    parser.add_argument('--n-samples', type=int,   default=None)
    parser.add_argument('--n-blocks',  type=int,   default=None)
    parser.add_argument('--model',     choices=['lorenzo', 'correti'],
                        default='correti')
    parser.add_argument('--w-xz',      type=float, default=None)
    parser.add_argument('--w-zx',      type=float, default=None)
    parser.add_argument('--save-dir',  default='./checkpoints/coex')
    parser.add_argument('--data-dir',  default='./data/coex')
    args = parser.parse_args()

    kwargs = dict(
        seed=args.seed,
        save_dir=args.save_dir,
        data_dir=args.data_dir,
        model_type=args.model,
    )
    if args.n_epochs  is not None: kwargs['n_epochs']  = args.n_epochs
    if args.batch_size is not None: kwargs['batch_size'] = args.batch_size
    if args.n_samples is not None: kwargs['n_samples'] = args.n_samples
    if args.n_blocks  is not None: kwargs['n_blocks']  = args.n_blocks
    if args.w_xz      is not None: kwargs['w_xz']      = args.w_xz
    if args.w_zx      is not None: kwargs['w_zx']      = args.w_zx

    cfg = make_coex_config(**kwargs)

    if args.generate_data:
        import jax
        key = jax.random.PRNGKey(cfg.seed)
        load_or_generate_coex_data(cfg, key)
        print('Data generation complete.')
        return

    train_coex(cfg)


if __name__ == '__main__':
    main()
