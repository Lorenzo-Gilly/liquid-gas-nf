"""JAX-backend training pipeline: flow builder, dynamic prior, training loop.

Depends on:
    - lorenzo_models (coupling_flows, bijectors, attention)
    - config.PipelineConfig
    - physics (energy functions, MCMC, utilities)
"""

import functools
import json
import math
import os
import pickle
import time

import jax
import jax.numpy as jnp
import numpy as np
import haiku as hk
import distrax
import optax

from config import PipelineConfig
from physics import (
    make_lj_fn, make_wca_fn, fcc_lattice, run_mcmc,
    generate_dataset, ress, octahedral_transform,
    wrap_pbc, center_particle,
)
from lorenzo_models import bijectors
from lorenzo_models import utils as lm_utils
from lorenzo_models.attention import Transformer
from lorenzo_models.coupling_flows import (
    make_equivariant_conditioner, make_split_coupling_flow,
)


# ---------------------------------------------------------------------------
# a) Flow builder
# ---------------------------------------------------------------------------

def make_correti_coupling_flow(
    event_shape,
    lower,
    upper,
    num_blocks,
    num_bins,
    conditioner,
    use_circular_shift,
    circular_shift_init,
):
    """Create a flow using the Coretti block pattern.

    Each super-block consists of:
        circular_shift, RQS(swap=True), RQS(swap=False),
        circular_shift, RQS(swap=False), RQS(swap=True)

    No particle permutations.  Total coupling layers = num_blocks * 4.

    Uses the same primitives as make_split_coupling_flow (distrax RQS,
    SplitCoupling, CircularShift, equivariant conditioner).
    """
    split_axis = -1
    split_size = event_shape[split_axis]
    split_index = split_size // 2

    def bijector_fn(params):
        return distrax.RationalQuadraticSpline(
            params,
            range_min=lower,
            range_max=upper,
            boundary_slopes='circular',
            min_bin_size=(upper - lower) * 1e-4)

    def _make_coupling(swap):
        shape_transformed = list(event_shape)
        shape_transformed[split_axis] = (
            split_index if swap else split_size - split_index)
        return distrax.SplitCoupling(
            swap=swap,
            split_index=split_index,
            split_axis=split_axis,
            event_ndims=len(event_shape),
            bijector=bijector_fn,
            conditioner=conditioner['constructor'](
                shape_transformed=shape_transformed,
                num_bijector_params=3 * num_bins + 1,
                lower=lower,
                upper=upper,
                **conditioner['kwargs']))

    def _make_shift():
        shift = lm_utils.Parameter(
            name='circular_shift',
            param_name='shift',
            shape=event_shape[-1:],
            init=circular_shift_init)()
        shift_layer = bijectors.CircularShift(
            (upper - lower) * shift, lower, upper)
        return distrax.Block(shift_layer, len(event_shape))

    layers = []
    for _ in range(num_blocks):
        sublayers_a = []
        sublayers_b = []

        if use_circular_shift:
            sublayers_a.append(_make_shift())
        # RQS(swap=True), RQS(swap=False) — transform dim 0 then dim 1
        sublayers_a.append(_make_coupling(swap=True))
        sublayers_a.append(_make_coupling(swap=False))

        if use_circular_shift:
            sublayers_b.append(_make_shift())
        # RQS(swap=False), RQS(swap=True) — transform dim 1 then dim 0
        sublayers_b.append(_make_coupling(swap=False))
        sublayers_b.append(_make_coupling(swap=True))

        layers.append(distrax.Chain(sublayers_a))
        layers.append(distrax.Chain(sublayers_b))

    return distrax.Chain(layers)


def build_flow(cfg: PipelineConfig):
    """Build the normalizing flow as a pair (init_fn, apply_fn).

    The flow maps between physical coordinates (B, N*D) and the latent space.
    Internal pipeline:
        Physical [-L/2, L/2] -> Rescale to [-1, 1]
        -> RemoveOrigin (N,D) -> (N-1)*D flat
        -> Split coupling flow (24 layers, each = permute + shift + 2 couplings)
        -> AddOrigin
        -> Rescale back to [-L/2, L/2]

    Returns:
        (init_fn, apply_fn) from hk.transform.
        apply_fn signature: apply_fn(params, rng, x) -> (y, log_det)
                            apply_fn(params, rng, y, inverse=True) -> (x, log_det)
    """
    N = cfg.system.n_particles
    D = cfg.system.dimensions
    L = cfg.system.box_length
    fc = cfg.flow

    def flow_fn(x, inverse=False):
        """x: (B, N*D). Returns (y, log_det)."""
        # Build bijector components
        rescale_in = bijectors.Rescale(-L / 2, L / 2, -1.0, 1.0)
        remove_origin = bijectors.RemoveOrigin(N, D)
        rescale_out = bijectors.Rescale(-1.0, 1.0, -L / 2, L / 2)

        conditioner_cfg = dict(
            constructor=make_equivariant_conditioner,
            kwargs=dict(
                embedding_size=fc.embedding_size,
                conditioner_constructor=Transformer,
                conditioner_kwargs=dict(
                    num_heads=fc.transformer_heads,
                    num_layers=fc.transformer_depth,
                ),
                num_frequencies=fc.n_freqs,
                w_init_final=hk.initializers.RandomNormal(stddev=fc.init_std),
            ),
        )

        if fc.model_type == "correti":
            coupling_flow = make_correti_coupling_flow(
                event_shape=(N - 1, D),
                lower=-1.0,
                upper=1.0,
                num_blocks=fc.n_blocks,
                num_bins=fc.n_bins,
                conditioner=conditioner_cfg,
                use_circular_shift=fc.use_circular_shift,
                circular_shift_init=jnp.zeros,
            )
        else:
            num_layers = fc.n_blocks * 2
            coupling_flow = make_split_coupling_flow(
                event_shape=(N - 1, D),
                lower=-1.0,
                upper=1.0,
                num_layers=num_layers,
                num_bins=fc.n_bins,
                conditioner=conditioner_cfg,
                permute_variables=fc.permute_variables,
                split_axis=-1,
                use_circular_shift=fc.use_circular_shift,
                prng=hk.next_rng_key(),
                circular_shift_init=jnp.zeros,
            )

        # Manual pipeline (avoids Chain event-ndims mismatch between flat
        # rescale and 2-D RemoveOrigin).
        if not inverse:
            # Forward: z (physical) -> x (physical)
            # 1. Rescale to [-1, 1]  (element-wise on flat vector)
            h = rescale_in.forward(x)
            ld = rescale_in.forward_log_det_jacobian(x).sum(axis=-1)  # scalar, (B,)
            # 2. Reshape to (B, N, D) then RemoveOrigin
            h_nd = h.reshape(x.shape[0], N, D)
            h_stripped, ld2 = remove_origin.forward_and_log_det(h_nd)
            ld = ld + ld2
            # 3. Coupling flow on (B, N-1, D) space
            h_flow = h_stripped.reshape(x.shape[0], N - 1, D)
            h_out, ld3 = coupling_flow.forward_and_log_det(h_flow)
            ld = ld + ld3
            # 4. Inverse RemoveOrigin -> (B, N, D)
            h_flat = h_out.reshape(x.shape[0], (N - 1) * D)
            h_full, ld4 = remove_origin.inverse_and_log_det(h_flat)
            ld = ld + ld4
            # 5. Flatten and rescale back
            h_flat2 = h_full.reshape(x.shape[0], N * D)
            y = rescale_out.forward(h_flat2)
            ld = ld + rescale_out.forward_log_det_jacobian(h_flat2).sum(axis=-1)
            return y, ld
        else:
            # Inverse: x (physical) -> z (physical)
            # 5. Rescale physical -> [-1, 1]
            h = rescale_in.forward(x)
            ld = rescale_in.forward_log_det_jacobian(x).sum(axis=-1)
            # 4. Reshape + RemoveOrigin
            h_nd = h.reshape(x.shape[0], N, D)
            h_stripped, ld2 = remove_origin.forward_and_log_det(h_nd)
            ld = ld + ld2
            # 3. Inverse coupling flow
            h_flow = h_stripped.reshape(x.shape[0], N - 1, D)
            h_out, ld3 = coupling_flow.inverse_and_log_det(h_flow)
            ld = ld + ld3
            # 2. Inverse RemoveOrigin
            h_flat = h_out.reshape(x.shape[0], (N - 1) * D)
            h_full, ld4 = remove_origin.inverse_and_log_det(h_flat)
            ld = ld + ld4
            # 1. Rescale back
            h_flat2 = h_full.reshape(x.shape[0], N * D)
            y = rescale_out.forward(h_flat2)
            ld = ld + rescale_out.forward_log_det_jacobian(h_flat2).sum(axis=-1)
            return y, ld

    # Use hk.transform for pure functional params
    def init_and_forward(x, inverse=False):
        return flow_fn(x, inverse=inverse)

    transformed = hk.transform(init_and_forward)
    return transformed.init, transformed.apply


# ---------------------------------------------------------------------------
# b) Dynamic prior (Python-side mutable cache)
# ---------------------------------------------------------------------------

class DynamicPrior:
    """Manages a cache of WCA configurations refreshed by MCMC.

    The cache lives outside JIT as a numpy/jax array. Each training step:
    1. Extract a batch from the cache.
    2. Run JIT'd MCMC to evolve them.
    3. Write them back into the cache.
    """

    def __init__(self, wca_fn, mcmc_cfg, system_cfg, beta_source,
                 init_configs, key):
        """
        Args:
            wca_fn: JIT'd WCA energy function.
            mcmc_cfg: MCMCConfig.
            system_cfg: SystemConfig.
            beta_source: inverse temperature for source (WCA).
            init_configs: (n_samples, N*D) initial WCA configurations.
            key: PRNG key.
        """
        self.wca_fn = wca_fn
        self.mcmc_cfg = mcmc_cfg
        self.sys = system_cfg
        self.beta = beta_source

        N = system_cfg.n_particles
        D = system_cfg.dimensions
        L = system_cfg.box_length

        # Center all initial configs on particle 0 (matches original
        # MCMC sampler's transform=True which centers particle 0 at origin).
        init_configs = center_particle(init_configs, 0, N, D, L)

        n_test = int(init_configs.shape[0] * mcmc_cfg.test_fraction)
        self.test_data = init_configs[-n_test:]
        self.cache = np.array(init_configs[:-n_test])  # mutable numpy
        self.cache_size = self.cache.shape[0]
        self.key = key

        # Pre-compute test energies (after centering; energy is PBC-invariant)
        self.test_energies = np.array(wca_fn(self.test_data))

        # JIT the MCMC runner (n_cycles baked in as static)
        self._run_mcmc = jax.jit(functools.partial(
            run_mcmc,
            energy_fn=wca_fn,
            beta=beta_source,
            n_cycles=mcmc_cfg.refresh_cycles,
            step_size=mcmc_cfg.step_size,
            box_length=system_cfg.box_length,
            n_particles=system_cfg.n_particles,
            dimensions=system_cfg.dimensions,
        ))

    def sample(self, n, key, training=True):
        """Sample n configurations.

        If training: pick n from cache, evolve via MCMC, update cache, return.
        If eval: pick n from test set (no MCMC).

        Returns:
            (n, N*D) jax array.
        """
        if not training:
            indices = jax.random.choice(key, self.test_data.shape[0], (n,),
                                        replace=n > self.test_data.shape[0])
            return jnp.array(self.test_data[indices])

        # Pick random cache indices
        self.key, k1 = jax.random.split(self.key)
        indices = jax.random.choice(k1, self.cache_size, (n,), replace=False)
        indices_np = np.array(indices)
        batch = jnp.array(self.cache[indices_np])

        # Evolve via MCMC
        self.key, k2 = jax.random.split(self.key)
        evolved, _ = self._run_mcmc(
            configs=batch,
            key=k2,
        )

        # Center on particle 0 (matches original transform=True)
        evolved = center_particle(evolved, 0, self.sys.n_particles,
                                  self.sys.dimensions, self.sys.box_length)

        # Write back
        evolved_np = np.array(evolved)
        self.cache[indices_np] = evolved_np

        return evolved

    def sample_test(self, n, key):
        """Sample n test configurations with energies."""
        indices = jax.random.choice(key, self.test_data.shape[0], (n,),
                                    replace=n > self.test_data.shape[0])
        idx_np = np.array(indices)
        return jnp.array(self.test_data[idx_np]), jnp.array(self.test_energies[idx_np])


# ---------------------------------------------------------------------------
# c) Dataset / augmentation
# ---------------------------------------------------------------------------

def make_batches(data, batch_size, key):
    """Shuffle data and yield batches as a list of jax arrays."""
    n = data.shape[0]
    perm = jax.random.permutation(key, n)
    shuffled = data[perm]
    batches = []
    for i in range(0, n, batch_size):
        batch = shuffled[i:i + batch_size]
        if batch.shape[0] == batch_size:  # drop last incomplete batch
            batches.append(batch)
    return batches


def augment_batch(batch, key, n_particles, dimensions, box_length):
    """Apply data augmentation to a batch of configurations.

    Per sample:
        1. Center on a random particle.
        2. Apply PBC wrapping.
        3. Swap that particle to position 0.
        4. Apply random octahedral transformation.

    Args:
        batch: (B, N*D) configurations.
        key: PRNG key.

    Returns:
        (B, N*D) augmented configurations.
    """
    B = batch.shape[0]
    k1, k2 = jax.random.split(key)

    # Random particle indices per sample
    particle_idx = jax.random.randint(k1, (B,), 0, n_particles)

    def augment_single(x, pidx, rng):
        shaped = x.reshape(n_particles, dimensions)
        # Center on chosen particle
        origin = shaped[pidx]
        shifted = shaped - origin
        wrapped = wrap_pbc(shifted, box_length)
        # Swap particle pidx to position 0
        # Use dynamic indexing: swap rows 0 and pidx
        row0 = wrapped[0]
        rowp = wrapped[pidx]
        swapped = wrapped.at[0].set(rowp)
        swapped = swapped.at[pidx].set(row0)
        # Octahedral transform
        R = octahedral_transform(rng, dimensions)  # (D, D)
        transformed = swapped @ R  # (N, D) @ (D, D) -> (N, D)
        return transformed.reshape(n_particles * dimensions)

    keys = jax.random.split(k2, B)
    return jax.vmap(augment_single)(batch, particle_idx, keys)


def load_or_generate_data(cfg, key):
    """Load dataset from disk or generate via MCMC.

    Returns:
        (lj_data, wca_data) each (n_samples, N*D) as jax arrays.
    """
    os.makedirs(cfg.data_dir, exist_ok=True)
    lj_path = os.path.join(cfg.data_dir, "lj_configs.npy")
    wca_path = os.path.join(cfg.data_dir, "wca_configs.npy")

    lj_fn = make_lj_fn(cfg)
    wca_fn = make_wca_fn(cfg)

    if os.path.exists(lj_path) and os.path.exists(wca_path):
        print("Loading cached datasets...")
        lj_data = jnp.array(np.load(lj_path))
        wca_data = jnp.array(np.load(wca_path))
    else:
        print("Generating LJ dataset via MCMC...")
        key, k1 = jax.random.split(key)
        lj_data = generate_dataset(
            lj_fn, cfg.beta_target, cfg.mcmc.n_samples, cfg, k1)
        np.save(lj_path, np.array(lj_data))

        print("Generating WCA dataset via MCMC...")
        key, k2 = jax.random.split(key)
        wca_data = generate_dataset(
            wca_fn, cfg.beta_source, cfg.mcmc.n_samples, cfg, k2)
        np.save(wca_path, np.array(wca_data))

    print(f"  LJ data: {lj_data.shape}, WCA data: {wca_data.shape}")
    return lj_data, wca_data


# ---------------------------------------------------------------------------
# d) Loss functions
# ---------------------------------------------------------------------------

def loss_xz(apply_fn, params, rng, x, wca_fn, beta_source):
    """NLL loss: x (LJ samples) -> z (WCA latent).

    loss = -mean( -beta_source * U_WCA(z) + log|det J_inv| )

    Args:
        x: (B, N*D) LJ configurations.
    Returns:
        scalar loss.
    """
    z, log_det = apply_fn(params, rng, x, inverse=True)
    log_p_source = -beta_source * wca_fn(z)
    return -jnp.mean(log_p_source + log_det)


def loss_zx(apply_fn, params, rng, z, lj_fn, beta_target):
    """KLD loss: z (WCA samples) -> x (LJ target).

    loss = -mean( -beta_target * U_LJ(x) + log|det J_fwd| )

    Args:
        z: (B, N*D) WCA configurations.
    Returns:
        scalar loss.
    """
    x, log_det = apply_fn(params, rng, z, inverse=False)
    log_p_target = -beta_target * lj_fn(x)
    return -jnp.mean(log_p_target + log_det)


def total_loss(apply_fn, params, rng, x, z,
               wca_fn, lj_fn, beta_source, beta_target,
               w_xz, w_zx):
    """Combined two-sided loss.

    Returns:
        (loss, aux) where aux = dict with individual losses.
    """
    r1, r2 = jax.random.split(rng)
    l_xz = loss_xz(apply_fn, params, r1, x, wca_fn, beta_source)
    l_zx = loss_zx(apply_fn, params, r2, z, lj_fn, beta_target)
    total = w_xz * l_xz + w_zx * l_zx
    return total, {"loss_xz": l_xz, "loss_zx": l_zx}


# ---------------------------------------------------------------------------
# e) Validation / RESS computation
# ---------------------------------------------------------------------------

def compute_val_metrics(apply_fn, params, rng, x_test, z_test,
                        wca_fn, lj_fn, beta_source, beta_target,
                        energy_x_test=None, energy_z_test=None):
    """Compute validation losses and RESS metrics.

    Args:
        x_test: (B, N*D) LJ test configs.
        z_test: (B, N*D) WCA test configs.
        energy_x_test: (B,) pre-computed LJ energies (optional).
        energy_z_test: (B,) pre-computed WCA energies (optional).

    Returns:
        dict with val_loss_xz, val_loss_zx, ress_xz, ress_zx.
    """
    r1, r2 = jax.random.split(rng)

    # x -> z direction
    z_from_x, logdet_xz = apply_fn(params, r1, x_test, inverse=True)
    log_p_source = -beta_source * wca_fn(z_from_x)
    val_nll = -jnp.mean(log_p_source + logdet_xz)

    # Importance weights for RESS_xz
    if energy_x_test is None:
        energy_x_test = lj_fn(x_test)
    log_p_target_x = -beta_target * energy_x_test
    log_w_xz = log_p_source - log_p_target_x + logdet_xz
    ress_xz = ress(log_w_xz)

    # z -> x direction
    x_from_z, logdet_zx = apply_fn(params, r2, z_test, inverse=False)
    log_p_target = -beta_target * lj_fn(x_from_z)
    val_kld = -jnp.mean(log_p_target + logdet_zx)

    # Importance weights for RESS_zx
    if energy_z_test is None:
        energy_z_test = wca_fn(z_test)
    log_p_source_z = -beta_source * energy_z_test
    log_w_zx = log_p_target - log_p_source_z + logdet_zx
    ress_zx = ress(log_w_zx)

    return {
        "val_loss_xz": val_nll,
        "val_loss_zx": val_kld,
        "ress_xz": ress_xz,
        "ress_zx": ress_zx,
    }


# ---------------------------------------------------------------------------
# f) Training loop
# ---------------------------------------------------------------------------

def _cfg_to_dict(cfg: PipelineConfig) -> dict:
    """Serialize PipelineConfig to a plain dict for config.json."""
    bpe = cfg.batches_per_epoch
    return {
        "system": {
            "n_particles": cfg.system.n_particles,
            "dimensions": cfg.system.dimensions,
            "rho": cfg.system.rho,
            "box_length": cfg.system.box_length,
        },
        "physics": {
            "beta_source": cfg.beta_source,
            "beta_target": cfg.beta_target,
        },
        "flow": {
            "model_type": cfg.flow.model_type,
            "n_blocks": cfg.flow.n_blocks,
            "n_rqs_layers": cfg.flow.n_blocks * 2,
            "n_bins": cfg.flow.n_bins,
            "embedding_size": cfg.flow.embedding_size,
            "transformer_depth": cfg.flow.transformer_depth,
            "transformer_heads": cfg.flow.transformer_heads,
            "n_freqs": cfg.flow.n_freqs,
            "init_std": cfg.flow.init_std,
            "use_circular_shift": cfg.flow.use_circular_shift,
            "permute_variables": cfg.flow.permute_variables,
        },
        "training": {
            "n_epochs": cfg.train.n_epochs,
            "batch_size": cfg.train.batch_size,
            "lr": cfg.train.lr,
            "milestones_epochs": list(cfg.train.milestones_epochs),
            "gamma": cfg.train.gamma,
            "batches_per_epoch": bpe,
            "total_steps": cfg.train.n_epochs * bpe,
            "milestone_steps": list(cfg.milestone_steps),
        },
        "mcmc": {
            "n_samples": cfg.mcmc.n_samples,
            "step_size": cfg.mcmc.step_size,
            "n_equilibration": cfg.mcmc.n_equilibration,
            "refresh_cycles": cfg.mcmc.refresh_cycles,
        },
        "seed": cfg.seed,
        "data_dir": cfg.data_dir,
    }


def train(cfg: PipelineConfig):
    """Main JAX training loop."""
    print("=" * 70)
    print("NF Training Pipeline (JAX backend)")
    print("=" * 70)

    os.makedirs(cfg.save_dir, exist_ok=True)

    config_path = os.path.join(cfg.save_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump(_cfg_to_dict(cfg), f, indent=2)
    print(f"Config saved: {config_path}")
    key = jax.random.PRNGKey(cfg.seed)

    # --- Energy functions ---
    lj_fn = make_lj_fn(cfg)
    wca_fn = make_wca_fn(cfg)

    # --- Data ---
    key, k_data = jax.random.split(key)
    lj_data, wca_data = load_or_generate_data(cfg, k_data)

    # Split LJ data into train/test
    n_test = int(lj_data.shape[0] * cfg.mcmc.test_fraction)
    lj_train = lj_data[:-n_test]
    lj_test = lj_data[-n_test:]
    lj_test_energies = lj_fn(lj_test)

    # --- Build flow ---
    print("Building flow...")
    init_fn, apply_fn = build_flow(cfg)

    key, k_init = jax.random.split(key)
    N = cfg.system.n_particles
    D = cfg.system.dimensions
    dummy_x = jnp.zeros((1, N * D))
    params = init_fn(k_init, dummy_x, inverse=False)

    # Count parameters
    n_params = sum(p.size for p in jax.tree.leaves(params))
    print(f"Flow parameters: {n_params:,}")

    # --- Weight initialization: match original Correti init_weights() ---
    # Original PyTorch code (flow_assembler.init_weights) only re-initializes
    # nn.Linear layers with N(0, 0.01).  LayerNorm keeps its defaults
    # (scale=1, offset=0) and circular shifts stay at 0.
    def reinit_like_correti(params, key):
        """Re-init matching original Correti: only Linear w/b get N(0, std).

        - Linear weights ('w') and biases ('b'): N(0, init_std)
        - LayerNorm scale: ones  (PyTorch nn.LayerNorm default)
        - LayerNorm offset: zeros
        - Circular shift: zeros
        - Anything else: keep Haiku default
        """
        new_params = {}
        for module_name, module_params in params.items():
            new_module = {}
            for param_name, param in module_params.items():
                key, subkey = jax.random.split(key)
                if param_name in ('w', 'b'):
                    # Linear weight/bias → N(0, init_std)
                    new_module[param_name] = (
                        cfg.flow.init_std
                        * jax.random.normal(subkey, param.shape))
                elif param_name == 'scale':
                    # LayerNorm scale → 1
                    new_module[param_name] = jnp.ones_like(param)
                elif param_name == 'offset':
                    # LayerNorm offset → 0
                    new_module[param_name] = jnp.zeros_like(param)
                elif param_name == 'shift':
                    # Circular shift → 0
                    new_module[param_name] = jnp.zeros_like(param)
                else:
                    # Unknown param type — keep Haiku init
                    new_module[param_name] = param
            new_params[module_name] = new_module
        return new_params

    key, k_reinit = jax.random.split(key)
    params = reinit_like_correti(params, k_reinit)
    print(f"Params re-initialized (Correti-style): Linear w/b ~ N(0,{cfg.flow.init_std}), "
          f"LayerNorm scale=1/offset=0, shifts=0")

    # --- Optimizer ---
    tc = cfg.train
    bpe = cfg.batches_per_epoch
    total_steps = tc.n_epochs * bpe
    milestone_steps = cfg.milestone_steps
    print(f"Batches/epoch: {bpe}, Total steps: {total_steps}")
    print(f"LR milestones (steps): {milestone_steps}")

    schedule = optax.join_schedules(
        schedules=[
            optax.constant_schedule(tc.lr),
            optax.constant_schedule(tc.lr * tc.gamma),
            optax.constant_schedule(tc.lr * tc.gamma ** 2),
        ],
        boundaries=list(milestone_steps),
    )
    optimizer = optax.adam(schedule)
    opt_state = optimizer.init(params)

    # --- Dynamic prior ---
    print("Setting up dynamic prior...")
    key, k_prior = jax.random.split(key)
    dynamic_prior = DynamicPrior(
        wca_fn=wca_fn,
        mcmc_cfg=cfg.mcmc,
        system_cfg=cfg.system,
        beta_source=cfg.beta_source,
        init_configs=wca_data,
        key=k_prior,
    )

    # --- JIT the training step ---
    @jax.jit
    def train_step(params, opt_state, rng, x_batch, z_batch):
        (loss_val, aux), grads = jax.value_and_grad(total_loss, argnums=1, has_aux=True)(
            apply_fn, params, rng, x_batch, z_batch,
            wca_fn, lj_fn, cfg.beta_source, cfg.beta_target,
            tc.w_xz, tc.w_zx,
        )
        updates, new_opt_state = optimizer.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)
        return new_params, new_opt_state, loss_val, aux

    # JIT the validation step
    @jax.jit
    def val_step(params, rng, x_test, z_test, energy_x, energy_z):
        return compute_val_metrics(
            apply_fn, params, rng, x_test, z_test,
            wca_fn, lj_fn, cfg.beta_source, cfg.beta_target,
            energy_x_test=energy_x, energy_z_test=energy_z,
        )

    # --- Augmentation (JIT'd) ---
    augment_fn = jax.jit(functools.partial(
        augment_batch,
        n_particles=N,
        dimensions=D,
        box_length=cfg.system.box_length,
    ))

    # --- Training loop ---
    print("\nStarting training...")
    step = 0
    metrics_log = []

    for epoch in range(1, tc.n_epochs + 1):
        t0 = time.time()
        key, k_epoch = jax.random.split(key)

        # Shuffle & batch LJ training data
        batches = make_batches(lj_train, tc.batch_size, k_epoch)

        epoch_loss = 0.0
        epoch_loss_xz = 0.0
        epoch_loss_zx = 0.0
        n_batches = 0

        for x_batch in batches:
            key, k_aug, k_z, k_step = jax.random.split(key, 4)

            # Augment x
            x_aug = augment_fn(x_batch, k_aug)

            # Sample z from dynamic prior
            z_batch = dynamic_prior.sample(tc.batch_size, k_z, training=True)

            # Train step
            params, opt_state, loss_val, aux = train_step(
                params, opt_state, k_step, x_aug, z_batch)

            epoch_loss += float(loss_val)
            epoch_loss_xz += float(aux["loss_xz"])
            epoch_loss_zx += float(aux["loss_zx"])
            n_batches += 1
            step += 1

        epoch_loss /= max(n_batches, 1)
        epoch_loss_xz /= max(n_batches, 1)
        epoch_loss_zx /= max(n_batches, 1)
        dt = time.time() - t0

        # --- Validation ---
        if tc.n_dump > 0 and (epoch == 1 or epoch % tc.n_dump == 0):
            key, k_val, k_ztest = jax.random.split(key, 3)

            # Center + wrap test x (center on particle 0, no augmentation)
            x_test_centered = center_particle(
                lj_test, 0, N, D, cfg.system.box_length)

            # Get test z from dynamic prior
            z_test, energy_z_test = dynamic_prior.sample_test(
                min(tc.batch_size, lj_test.shape[0]), k_ztest)

            val = val_step(
                params, k_val,
                x_test_centered[:tc.batch_size],
                z_test,
                lj_test_energies[:tc.batch_size],
                energy_z_test,
            )

            # Current LR
            lr = float(schedule(step))

            msg = (
                f"epoch {epoch:4d} | "
                f"train: xz={epoch_loss_xz:.3f} zx={epoch_loss_zx:.3f} "
                f"loss={epoch_loss:.3f} | "
                f"eval: xz={float(val['val_loss_xz']):.3f} "
                f"zx={float(val['val_loss_zx']):.3f} "
                f"ress_xz={float(val['ress_xz']):.4f} "
                f"ress_zx={float(val['ress_zx']):.4f} | "
                f"lr={lr:.1e} | {dt:.1f}s"
            )
            print(msg)

            metrics_log.append({
                "epoch": epoch,
                "train_loss_xz": epoch_loss_xz,
                "train_loss_zx": epoch_loss_zx,
                "val_loss_xz": float(val["val_loss_xz"]),
                "val_loss_zx": float(val["val_loss_zx"]),
                "ress_xz": float(val["ress_xz"]),
                "ress_zx": float(val["ress_zx"]),
                "lr": lr,
            })

        # --- Save checkpoint ---
        if tc.n_save > 0 and epoch % tc.n_save == 0 and epoch != tc.n_epochs:
            ckpt_path = os.path.join(cfg.save_dir, f"params_epoch_{epoch}.pkl")
            with open(ckpt_path, "wb") as f:
                pickle.dump(params, f)
            print(f"  Saved checkpoint: {ckpt_path}")

    # --- Save final ---
    final_path = os.path.join(cfg.save_dir, "params_final.pkl")
    with open(final_path, "wb") as f:
        pickle.dump(params, f)
    print(f"\nSaved final params: {final_path}")

    # Save metrics log
    log_path = os.path.join(cfg.save_dir, "train_log.txt")
    with open(log_path, "w") as f:
        header = "epoch\tloss_xz\tloss_zx\tval_loss_xz\tval_loss_zx\tress_xz\tress_zx\tLR\n"
        f.write(header)
        for m in metrics_log:
            row = (f"{m['epoch']}\t{m['train_loss_xz']:.6f}\t{m['train_loss_zx']:.6f}\t"
                   f"{m['val_loss_xz']:.6f}\t{m['val_loss_zx']:.6f}\t"
                   f"{m['ress_xz']:.6f}\t{m['ress_zx']:.6f}\t{m['lr']:.1e}\n")
            f.write(row)
    print(f"Saved training log: {log_path}")

    return params, metrics_log
