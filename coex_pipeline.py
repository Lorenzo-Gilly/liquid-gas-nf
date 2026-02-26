"""Coexistence NF pipeline: data generation + training loop.

Adapts jax_pipeline.py for a LJ→LJ mapping:
  Source (base):  LJ at T*=T_BASE  (single-phase fluid)
  Target:         LJ at T*=T_TARGET (coexistence — two phases)

Both distributions use the same LJ energy function with use_lrc=False.
The key difference from the WCA→LJ pipeline is that the target training
data is generated with BOTH liquid-cluster and dispersed-gas initialisations
so the flow sees both phases of the target distribution.
"""

import functools
import json
import os
import pickle
import time

import jax
import jax.numpy as jnp
import numpy as np

from physics import make_lj_fn, run_mcmc, center_particle
from diagnostics import make_liquid_init, make_gas_init

# Reuse unchanged components from jax_pipeline
from jax_pipeline import (
    build_flow,
    make_batches,
    augment_batch,
    DynamicPrior,
    loss_xz,
    loss_zx,
    total_loss,
    compute_val_metrics,
)

# ── data generation ───────────────────────────────────────────────────────────

_GEN_CHAINS  = 128   # parallel chains for data generation
_BASE_WARMUP = 10_000  # fast at T*=0.50
_TGT_WARMUP  = 50_000  # need stable droplet / stable gas at T*=0.36
_GEN_STRIDE  = 1_000   # moves between saved snapshots


def _collect_configs(energy_fn, beta, init_fn, n_configs, n_chains,
                     n_warmup, stride, step_size, N, D, L, key):
    """Run n_chains parallel chains; collect n_configs evenly spaced snapshots.

    Each snapshot is one saved configuration per chain.  Total saved =
    n_configs (may be slightly rounded down to a multiple of n_chains).
    """
    key, k_init = jax.random.split(key)
    configs = jnp.stack([init_fn(N, D, L, rng_key=k)
                         for k in jax.random.split(k_init, n_chains)])

    # Warmup
    key, k_w = jax.random.split(key)
    configs, _ = run_mcmc(configs, energy_fn, beta, n_warmup,
                          k_w, step_size, L, N, D)

    # Production: save one snapshot per chain every `stride` moves
    n_rounds   = n_configs // n_chains
    all_cfgs   = []
    for _ in range(n_rounds):
        key, k_p = jax.random.split(key)
        configs, _ = run_mcmc(configs, energy_fn, beta, stride,
                              k_p, step_size, L, N, D)
        # Center on particle 0 for consistency with augmentation
        centered = center_particle(configs, 0, N, D, L)
        all_cfgs.append(np.array(centered))

    return np.concatenate(all_cfgs, axis=0), key


def generate_coex_data(cfg, key):
    """Generate and save base + target configurations.

    Base:   T*=T_BASE  — from fluid initialisations, single phase.
    Target: T*=T_TARGET — 50 % from liquid-cluster init, 50 % from gas init.

    Saves to cfg.data_dir/base_configs.npy and target_configs.npy.
    """
    N  = cfg.system.n_particles
    D  = cfg.system.dimensions
    L  = cfg.system.box_length
    ss = cfg.mcmc.step_size
    n_total = cfg.mcmc.n_samples  # 50K default

    lj_fn = make_lj_fn(cfg)
    os.makedirs(cfg.data_dir, exist_ok=True)

    # ── Base configs ──────────────────────────────────────────────────────────
    print(f'  Generating {n_total} base configs  T*={1/cfg.beta_source:.2f}…',
          flush=True)
    t0 = time.time()
    base_configs, key = _collect_configs(
        lj_fn, cfg.beta_source, make_liquid_init,
        n_total, _GEN_CHAINS, _BASE_WARMUP, _GEN_STRIDE,
        ss, N, D, L, key)
    np.save(os.path.join(cfg.data_dir, 'base_configs.npy'), base_configs)
    print(f'    done ({time.time()-t0:.0f}s)  shape={base_configs.shape}',
          flush=True)

    # ── Target configs: liq-init ──────────────────────────────────────────────
    n_each = n_total // 2
    print(f'  Generating {n_each} liq-init target configs  T*={1/cfg.beta_target:.2f}…',
          flush=True)
    t0 = time.time()
    liq_cfgs, key = _collect_configs(
        lj_fn, cfg.beta_target, make_liquid_init,
        n_each, _GEN_CHAINS // 2, _TGT_WARMUP, _GEN_STRIDE * 2,
        ss, N, D, L, key)
    print(f'    done ({time.time()-t0:.0f}s)', flush=True)

    # ── Target configs: gas-init ──────────────────────────────────────────────
    print(f'  Generating {n_each} gas-init target configs  T*={1/cfg.beta_target:.2f}…',
          flush=True)
    t0 = time.time()
    gas_cfgs, key = _collect_configs(
        lj_fn, cfg.beta_target, make_gas_init,
        n_each, _GEN_CHAINS // 2, 0, _GEN_STRIDE * 2,
        ss, N, D, L, key)
    print(f'    done ({time.time()-t0:.0f}s)', flush=True)

    # Concatenate and shuffle target data so both phases are interleaved
    target_configs = np.concatenate([liq_cfgs, gas_cfgs], axis=0)
    rng = np.random.default_rng(42)
    rng.shuffle(target_configs)
    np.save(os.path.join(cfg.data_dir, 'target_configs.npy'), target_configs)
    print(f'  Saved target_configs {target_configs.shape}  '
          f'({n_each} liq + {n_each} gas, shuffled)', flush=True)

    return jnp.array(target_configs), jnp.array(base_configs)


def load_or_generate_coex_data(cfg, key):
    """Load cached coex data or generate if absent.

    Returns:
        (target_data, base_data) each (n_samples, N*D) as jax arrays.
        target_data = T*=T_TARGET configs (50 % liq-init, 50 % gas-init).
        base_data   = T*=T_BASE configs (single-phase fluid).
    """
    base_path   = os.path.join(cfg.data_dir, 'base_configs.npy')
    target_path = os.path.join(cfg.data_dir, 'target_configs.npy')

    if os.path.exists(base_path) and os.path.exists(target_path):
        print('Loading cached coex datasets…')
        target_data = jnp.array(np.load(target_path))
        base_data   = jnp.array(np.load(base_path))
        print(f'  target: {target_data.shape}  base: {base_data.shape}')
        return target_data, base_data

    print('Generating coex datasets via MCMC…')
    return generate_coex_data(cfg, key)


# ── training loop ─────────────────────────────────────────────────────────────

def train_coex(cfg):
    """Train normalising flow for LJ coexistence.

    Identical structure to jax_pipeline.train except:
      - Both source and target use the same LJ energy function.
      - Training data is loaded from data/coex/ (50% liq + 50% gas target).
      - DynamicPrior evolves base (T*=T_BASE) configurations during training.
    """
    print('=' * 70)
    print('Coexistence NF Training  (JAX backend)')
    print(f'  Source T*={1/cfg.beta_source:.2f}  Target T*={1/cfg.beta_target:.2f}')
    print(f'  N={cfg.system.n_particles}  ρ*={cfg.system.rho}  '
          f'L={cfg.system.box_length:.3f}')
    print(f'  w_xz={cfg.train.w_xz}  w_zx={cfg.train.w_zx}  '
          f'n_epochs={cfg.train.n_epochs}  batch={cfg.train.batch_size}')
    print('=' * 70)

    os.makedirs(cfg.save_dir, exist_ok=True)

    # Save config
    cfg_dict = {
        'N': cfg.system.n_particles, 'D': cfg.system.dimensions,
        'rho': cfg.system.rho, 'L': cfg.system.box_length,
        'T_base': round(1/cfg.beta_source, 4),
        'T_target': round(1/cfg.beta_target, 4),
        'beta_source': cfg.beta_source, 'beta_target': cfg.beta_target,
        'use_lrc': cfg.energy.use_lrc,
        'model_type': cfg.flow.model_type,
        'n_blocks': cfg.flow.n_blocks, 'n_bins': cfg.flow.n_bins,
        'embedding_size': cfg.flow.embedding_size,
        'n_epochs': cfg.train.n_epochs, 'batch_size': cfg.train.batch_size,
        'lr': cfg.train.lr, 'w_xz': cfg.train.w_xz, 'w_zx': cfg.train.w_zx,
        'milestones': list(cfg.train.milestones_epochs),
        'n_samples': cfg.mcmc.n_samples, 'seed': cfg.seed,
    }
    with open(os.path.join(cfg.save_dir, 'config.json'), 'w') as f:
        json.dump(cfg_dict, f, indent=2)

    key = jax.random.PRNGKey(cfg.seed)

    # ── Energy function (single LJ fn for both base and target) ────────────────
    lj_fn = make_lj_fn(cfg)

    # ── Data ──────────────────────────────────────────────────────────────────
    key, k_data = jax.random.split(key)
    target_data, base_data = load_or_generate_coex_data(cfg, k_data)

    N  = cfg.system.n_particles
    D  = cfg.system.dimensions
    L  = cfg.system.box_length
    tc = cfg.train

    n_test     = int(target_data.shape[0] * cfg.mcmc.test_fraction)
    tgt_train  = target_data[:-n_test]
    tgt_test   = target_data[-n_test:]
    tgt_test_e = lj_fn(tgt_test)

    # ── Flow ──────────────────────────────────────────────────────────────────
    print('Building flow…')
    init_fn, apply_fn = build_flow(cfg)
    key, k_init = jax.random.split(key)
    params = init_fn(k_init, jnp.zeros((1, N * D)), inverse=False)
    n_params = sum(p.size for p in jax.tree.leaves(params))
    print(f'  Parameters: {n_params:,}')

    # Correti-style weight init (matches original)
    def _reinit(params, key):
        new_params = {}
        for mname, mparams in params.items():
            new_mod = {}
            for pname, p in mparams.items():
                key, sk = jax.random.split(key)
                if pname in ('w', 'b'):
                    new_mod[pname] = cfg.flow.init_std * jax.random.normal(sk, p.shape)
                elif pname == 'scale':
                    new_mod[pname] = jnp.ones_like(p)
                elif pname == 'offset':
                    new_mod[pname] = jnp.zeros_like(p)
                elif pname == 'shift':
                    new_mod[pname] = jnp.zeros_like(p)
                else:
                    new_mod[pname] = p
            new_params[mname] = new_mod
        return new_params

    key, k_ri = jax.random.split(key)
    params = _reinit(params, k_ri)

    # ── Optimizer ──────────────────────────────────────────────────────────────
    import optax
    bpe   = cfg.batches_per_epoch
    steps = tc.n_epochs * bpe
    ms    = cfg.milestone_steps
    schedule = optax.join_schedules(
        schedules=[
            optax.constant_schedule(tc.lr),
            optax.constant_schedule(tc.lr * tc.gamma),
            optax.constant_schedule(tc.lr * tc.gamma ** 2),
        ],
        boundaries=list(ms),
    )
    optimizer = optax.adam(schedule)
    opt_state = optimizer.init(params)
    print(f'  Steps/epoch: {bpe}  Total: {steps}  LR milestones: {ms}')

    # ── Dynamic prior (base distribution — LJ at T_base) ───────────────────────
    print('Setting up dynamic base prior…')
    key, k_prior = jax.random.split(key)
    dynamic_prior = DynamicPrior(
        wca_fn=lj_fn,            # LJ energy, not WCA — name is legacy
        mcmc_cfg=cfg.mcmc,
        system_cfg=cfg.system,
        beta_source=cfg.beta_source,
        init_configs=base_data,
        key=k_prior,
    )

    # ── JIT steps ──────────────────────────────────────────────────────────────
    @jax.jit
    def train_step(params, opt_state, rng, x_batch, z_batch):
        (loss_val, aux), grads = jax.value_and_grad(
            total_loss, argnums=1, has_aux=True)(
            apply_fn, params, rng, x_batch, z_batch,
            lj_fn, lj_fn,          # both source and target use lj_fn
            cfg.beta_source, cfg.beta_target,
            tc.w_xz, tc.w_zx,
        )
        updates, new_opt = optimizer.update(grads, opt_state, params)
        return optax.apply_updates(params, updates), new_opt, loss_val, aux

    @jax.jit
    def val_step(params, rng, x_test, z_test, e_x, e_z):
        return compute_val_metrics(
            apply_fn, params, rng, x_test, z_test,
            lj_fn, lj_fn,
            cfg.beta_source, cfg.beta_target,
            energy_x_test=e_x, energy_z_test=e_z,
        )

    augment_fn = jax.jit(functools.partial(
        augment_batch, n_particles=N, dimensions=D, box_length=L))

    # ── Training loop ──────────────────────────────────────────────────────────
    print('\nTraining…')
    step = 0
    metrics_log = []

    for epoch in range(1, tc.n_epochs + 1):
        t0 = time.time()
        key, k_epoch = jax.random.split(key)
        batches = make_batches(tgt_train, tc.batch_size, k_epoch)

        ep_loss = ep_xz = ep_zx = 0.0
        n_b = 0
        for x_batch in batches:
            key, k_aug, k_z, k_step = jax.random.split(key, 4)
            x_aug  = augment_fn(x_batch, k_aug)
            z_batch = dynamic_prior.sample(tc.batch_size, k_z, training=True)
            params, opt_state, lv, aux = train_step(
                params, opt_state, k_step, x_aug, z_batch)
            ep_loss += float(lv)
            ep_xz   += float(aux['loss_xz'])
            ep_zx   += float(aux['loss_zx'])
            n_b += 1
            step += 1

        ep_loss /= max(n_b, 1)
        ep_xz   /= max(n_b, 1)
        ep_zx   /= max(n_b, 1)
        dt = time.time() - t0

        # Validation
        if epoch == 1 or epoch % tc.n_dump == 0:
            key, k_v, k_zt = jax.random.split(key, 3)
            x_tc = center_particle(tgt_test, 0, N, D, L)
            z_test, e_z = dynamic_prior.sample_test(
                min(tc.batch_size, tgt_test.shape[0]), k_zt)
            val = val_step(params, k_v,
                           x_tc[:tc.batch_size], z_test,
                           tgt_test_e[:tc.batch_size], e_z)
            lr = float(schedule(step))
            msg = (
                f'epoch {epoch:4d} | '
                f'train xz={ep_xz:.3f} zx={ep_zx:.3f} '
                f'loss={ep_loss:.3f} | '
                f'eval xz={float(val["val_loss_xz"]):.3f} '
                f'zx={float(val["val_loss_zx"]):.3f} '
                f'ress_xz={float(val["ress_xz"]):.4f} '
                f'ress_zx={float(val["ress_zx"]):.4f} | '
                f'lr={lr:.1e} | {dt:.1f}s'
            )
            print(msg, flush=True)
            metrics_log.append({
                'epoch': epoch,
                'train_loss_xz': ep_xz, 'train_loss_zx': ep_zx,
                'val_loss_xz': float(val['val_loss_xz']),
                'val_loss_zx': float(val['val_loss_zx']),
                'ress_xz': float(val['ress_xz']),
                'ress_zx': float(val['ress_zx']),
                'lr': lr,
            })

        # Checkpoint
        if tc.n_save > 0 and epoch % tc.n_save == 0 and epoch != tc.n_epochs:
            ckpt = os.path.join(cfg.save_dir, f'params_epoch_{epoch}.pkl')
            with open(ckpt, 'wb') as f:
                pickle.dump(params, f)
            print(f'  Checkpoint: {ckpt}', flush=True)

    # Final save
    final = os.path.join(cfg.save_dir, 'params_final.pkl')
    with open(final, 'wb') as f:
        pickle.dump(params, f)
    print(f'\nFinal params: {final}')

    log_path = os.path.join(cfg.save_dir, 'train_log.txt')
    with open(log_path, 'w') as f:
        f.write('epoch\txz\tzx\tval_xz\tval_zx\tress_xz\tress_zx\tlr\n')
        for m in metrics_log:
            f.write(f"{m['epoch']}\t{m['train_loss_xz']:.6f}\t"
                    f"{m['train_loss_zx']:.6f}\t{m['val_loss_xz']:.6f}\t"
                    f"{m['val_loss_zx']:.6f}\t{m['ress_xz']:.6f}\t"
                    f"{m['ress_zx']:.6f}\t{m['lr']:.1e}\n")
    print(f'Training log: {log_path}')
    return params, metrics_log
