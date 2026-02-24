"""Energy functions (JAX), MCMC sampler, FCC lattice, and utilities.

All functions are pure JAX, JIT-compatible where noted.
"""

import functools
import math

import jax
import jax.numpy as jnp


# ---------------------------------------------------------------------------
# PBC helpers
# ---------------------------------------------------------------------------

def wrap_pbc(x, box_length):
    """Wrap coordinates into [-L/2, L/2] via minimum image convention."""
    return x - box_length * jnp.round(x / box_length)


def center_particle(configs, particle_idx, n_particles, dimensions, box_length):
    """Center a specific particle at the origin with PBC.

    Args:
        configs: (B, N*D) flat configurations.
        particle_idx: scalar int, which particle to center on.
        n_particles, dimensions: system shape.
        box_length: box side length.

    Returns:
        (B, N*D) centered configs.
    """
    shaped = configs.reshape(configs.shape[0], n_particles, dimensions)
    origin = shaped[:, particle_idx:particle_idx+1, :]  # (B, 1, D)
    shifted = shaped - origin
    wrapped = wrap_pbc(shifted, box_length)
    return wrapped.reshape(configs.shape)


# ---------------------------------------------------------------------------
# Pair-distance computation (JIT-friendly)
# ---------------------------------------------------------------------------

def _pair_distances_sq(x, n_particles, dimensions, box_length):
    """Compute squared pair distances under PBC.

    Args:
        x: (B, N*D) flat configs.

    Returns:
        r2: (B, N*(N-1)/2) squared distances for unique pairs.
    """
    pos = x.reshape(x.shape[0], n_particles, dimensions)  # (B, N, D)
    # Upper-triangle pair indices
    idx_i, idx_j = jnp.triu_indices(n_particles, k=1)
    dr = pos[:, idx_i, :] - pos[:, idx_j, :]  # (B, n_pairs, D)
    dr = wrap_pbc(dr, box_length)
    r2 = jnp.sum(dr ** 2, axis=-1)  # (B, n_pairs)
    return r2


# ---------------------------------------------------------------------------
# LJ energy with cutin linearization, cutoff, optional LRC
# ---------------------------------------------------------------------------

def _lj_pair_energy(r2, epsilon, sigma, cutin, cutoff, tol):
    """LJ pair energy with cutin linearization and cutoff.

    Below cutin: linear extrapolation from the LJ curve at r=cutin.
    Above cutoff: zero (shifted so energy is continuous at cutoff).
    """
    sigma2 = sigma * sigma
    cutin2 = cutin * cutin
    cutoff2 = cutoff * cutoff

    # Clamp r2 for numerical safety
    r2_safe = jnp.clip(r2, tol)

    # Standard LJ: 4*eps*((sig/r)^12 - (sig/r)^6)
    inv_r2 = sigma2 / r2_safe
    inv_r6 = inv_r2 * inv_r2 * inv_r2
    inv_r12 = inv_r6 * inv_r6
    u_lj = 4.0 * epsilon * (inv_r12 - inv_r6)

    # Energy at cutoff (for shift)
    inv_rc2 = sigma2 / cutoff2
    inv_rc6 = inv_rc2 ** 3
    inv_rc12 = inv_rc6 * inv_rc6
    u_cutoff = 4.0 * epsilon * (inv_rc12 - inv_rc6)

    # Shifted LJ
    u_shifted = u_lj - u_cutoff

    # Force at cutin for linearization: F = -dU/dr at r=cutin
    inv_ci2 = sigma2 / cutin2
    inv_ci6 = inv_ci2 ** 3
    inv_ci12 = inv_ci6 * inv_ci6
    u_cutin = 4.0 * epsilon * (inv_ci12 - inv_ci6) - u_cutoff
    # dU/dr = 4*eps*(-12*sig^12/r^13 + 6*sig^6/r^7)
    #       = (1/r) * 4*eps*(-12*(sig/r)^12 + 6*(sig/r)^6)
    r_cutin = jnp.sqrt(cutin2)
    du_dr_cutin = (1.0 / r_cutin) * 4.0 * epsilon * (-12.0 * inv_ci12 + 6.0 * inv_ci6)
    # Linear: u_lin(r) = u_cutin + du_dr * (r - r_cutin)
    r = jnp.sqrt(r2_safe)
    u_linear = u_cutin + du_dr_cutin * (r - r_cutin)

    # Masks
    below_cutin = r2 < cutin2
    above_cutoff = r2 > cutoff2

    # Assemble
    u_pair = jnp.where(below_cutin, u_linear, u_shifted)
    u_pair = jnp.where(above_cutoff, 0.0, u_pair)
    return u_pair


def lj_energy(x, n_particles, dimensions, box_length,
              epsilon=1.0, sigma=1.0, cutin=0.8, cutoff=2.5,
              tol=1e-12, use_lrc=True):
    """Full LJ energy per configuration.

    Args:
        x: (B, N*D) flat configs.

    Returns:
        (B,) total energy per config.
    """
    r2 = _pair_distances_sq(x, n_particles, dimensions, box_length)
    u_pairs = _lj_pair_energy(r2, epsilon, sigma, cutin, cutoff, tol)
    u_total = jnp.sum(u_pairs, axis=-1)  # (B,)

    if use_lrc:
        # 2D LRC: U_lrc = pi * N * rho * eps * sigma^2 *
        #   [ (2/5)*(sigma/rc)^10 - (sigma/rc)^4 ]
        rho = n_particles / (box_length ** dimensions)
        src = sigma / cutoff
        if dimensions == 2:
            lrc = math.pi * n_particles * rho * epsilon * sigma**2 * (
                (2.0 / 5.0) * src**10 - src**4)
        else:
            # 3D LRC
            lrc = (8.0 / 3.0) * math.pi * n_particles * rho * epsilon * sigma**3 * (
                (1.0 / 3.0) * src**9 - src**3)
        u_total = u_total + lrc

    return u_total


def wca_energy(x, n_particles, dimensions, box_length,
               epsilon=1.0, sigma=1.0, cutin=0.8,
               tol=1e-12):
    """WCA energy per configuration (LJ truncated at 2^(1/6)*sigma, no LRC).

    Args:
        x: (B, N*D) flat configs.

    Returns:
        (B,) total energy per config.
    """
    wca_cutoff = sigma * (2.0 ** (1.0 / 6.0))
    r2 = _pair_distances_sq(x, n_particles, dimensions, box_length)
    u_pairs = _lj_pair_energy(r2, epsilon, sigma, cutin, wca_cutoff, tol)
    return jnp.sum(u_pairs, axis=-1)


# Convenience: partially-applied energy functions from config
def make_lj_fn(cfg):
    """Return a JIT-compiled LJ energy function from PipelineConfig."""
    return jax.jit(functools.partial(
        lj_energy,
        n_particles=cfg.system.n_particles,
        dimensions=cfg.system.dimensions,
        box_length=cfg.system.box_length,
        epsilon=cfg.energy.epsilon,
        sigma=cfg.energy.sigma,
        cutin=cfg.energy.cutin,
        cutoff=cfg.energy.lj_cutoff,
        tol=cfg.energy.tol,
        use_lrc=cfg.energy.use_lrc,
    ))


def make_wca_fn(cfg):
    """Return a JIT-compiled WCA energy function from PipelineConfig."""
    return jax.jit(functools.partial(
        wca_energy,
        n_particles=cfg.system.n_particles,
        dimensions=cfg.system.dimensions,
        box_length=cfg.system.box_length,
        epsilon=cfg.energy.epsilon,
        sigma=cfg.energy.sigma,
        cutin=cfg.energy.cutin,
        tol=cfg.energy.tol,
    ))


# ---------------------------------------------------------------------------
# FCC / triangular lattice for 2D
# ---------------------------------------------------------------------------

def fcc_lattice(n_particles, dimensions, box_length):
    """Generate a regular lattice (triangular for 2D, FCC for 3D).

    Returns:
        (N, D) positions centered in [-L/2, L/2].
    """
    if dimensions == 2:
        # Triangular lattice
        n_side = int(math.ceil(math.sqrt(n_particles)))
        positions = []
        for i in range(n_side):
            for j in range(n_side):
                x = i + 0.5 * (j % 2)
                y = j * math.sqrt(3.0) / 2.0
                positions.append([x, y])
                if len(positions) >= n_particles:
                    break
            if len(positions) >= n_particles:
                break
        positions = jnp.array(positions[:n_particles])
        # Scale to box
        span = positions.max(axis=0) - positions.min(axis=0)
        scale = box_length / jnp.maximum(span, 1e-8)
        # Use uniform scaling to preserve lattice shape
        s = jnp.min(scale)
        positions = positions * s
        # Center in box
        center = (positions.max(axis=0) + positions.min(axis=0)) / 2.0
        positions = positions - center
        return positions
    elif dimensions == 3:
        # Simple cubic fallback (for 3D, proper FCC would need unit cell logic)
        n_side = int(math.ceil(n_particles ** (1.0 / 3.0)))
        positions = []
        for i in range(n_side):
            for j in range(n_side):
                for k in range(n_side):
                    positions.append([i, j, k])
                    if len(positions) >= n_particles:
                        break
                if len(positions) >= n_particles:
                    break
            if len(positions) >= n_particles:
                break
        positions = jnp.array(positions[:n_particles], dtype=jnp.float32)
        spacing = box_length / n_side
        positions = positions * spacing + spacing / 2.0 - box_length / 2.0
        return positions
    else:
        raise ValueError(f"Unsupported dimensions={dimensions}")


# ---------------------------------------------------------------------------
# MCMC sampler (JAX, JIT-compatible inner loop)
# ---------------------------------------------------------------------------

def _metropolis_step(carry, _unused, energy_fn, beta, step_size,
                     box_length, n_particles, dimensions):
    """One MC sweep: attempt to move each particle once (vectorized over batch).

    carry = (configs, energies, key)
        configs: (B, N*D)
        energies: (B,)
        key: PRNG key
    """
    configs, energies, key = carry
    B = configs.shape[0]

    key, k_particle, k_disp, k_accept = jax.random.split(key, 4)

    # Pick a random particle for each sample
    particle_idx = jax.random.randint(k_particle, (B,), 0, n_particles)

    # Propose displacement
    disp = step_size * jax.random.normal(k_disp, (B, dimensions))

    # Apply displacement
    shaped = configs.reshape(B, n_particles, dimensions)
    # Scatter the displacement to the chosen particle
    particle_one_hot = jax.nn.one_hot(particle_idx, n_particles)  # (B, N)
    delta = particle_one_hot[:, :, None] * disp[:, None, :]  # (B, N, D)
    new_shaped = shaped + delta
    # PBC wrap
    new_shaped = wrap_pbc(new_shaped, box_length)
    new_configs = new_shaped.reshape(B, n_particles * dimensions)

    # Compute new energies
    new_energies = energy_fn(new_configs)

    # Metropolis criterion
    delta_e = new_energies - energies
    log_accept = -beta * delta_e
    log_u = jnp.log(jax.random.uniform(k_accept, (B,)))
    accept = log_u < log_accept  # (B,)

    configs = jnp.where(accept[:, None], new_configs, configs)
    energies = jnp.where(accept, new_energies, energies)

    return (configs, energies, key), None


def run_mcmc(configs, energy_fn, beta, n_cycles, key, step_size,
             box_length, n_particles, dimensions):
    """Run n_cycles of Metropolis MC (one particle move per cycle per sample).

    Args:
        configs: (B, N*D) initial configurations.
        energy_fn: callable (B, N*D) -> (B,).
        beta: inverse temperature.
        n_cycles: number of MC sweeps.
        key: PRNG key.

    Returns:
        configs: (B, N*D) updated configurations.
        energies: (B,) energies of final configs.
    """
    energies = energy_fn(configs)
    step_fn = functools.partial(
        _metropolis_step,
        energy_fn=energy_fn,
        beta=beta,
        step_size=step_size,
        box_length=box_length,
        n_particles=n_particles,
        dimensions=dimensions,
    )
    (configs, energies, _), _ = jax.lax.scan(
        step_fn, (configs, energies, key), None, length=n_cycles
    )
    return configs, energies


def generate_dataset(energy_fn, beta, n_samples, cfg, key):
    """Full MCMC dataset generation protocol.

    Protocol:
        1. Initialize n_samples replicas from FCC lattice with small noise.
        2. Hot equilibration at 0.2*beta for n_equilibration cycles.
        3. Re-equilibrate at beta for n_equilibration cycles.
        4. Production: n_production_cycles cycles at beta.

    Args:
        energy_fn: callable (B, N*D) -> (B,)
        beta: target inverse temperature.
        n_samples: total number of samples.
        cfg: PipelineConfig.
        key: PRNG key.

    Returns:
        configs: (n_samples, N*D) final configurations.
    """
    N = cfg.system.n_particles
    D = cfg.system.dimensions
    L = cfg.system.box_length
    mc = cfg.mcmc

    key, k_init = jax.random.split(key)
    lattice = fcc_lattice(N, D, L)  # (N, D)
    lattice_flat = lattice.reshape(N * D)
    # Replicate + small noise
    configs = jnp.tile(lattice_flat, (n_samples, 1))
    noise = 0.01 * jax.random.normal(k_init, configs.shape)
    configs = wrap_pbc(configs + noise, L)

    print(f"  MCMC: hot equilibration ({mc.n_equilibration} cycles at beta={0.2*beta:.3f})")
    key, k1 = jax.random.split(key)
    configs, _ = run_mcmc(
        configs, energy_fn, 0.2 * beta, mc.n_equilibration, k1,
        mc.step_size, L, N, D,
    )

    print(f"  MCMC: re-equilibration ({mc.n_equilibration} cycles at beta={beta:.3f})")
    key, k2 = jax.random.split(key)
    configs, _ = run_mcmc(
        configs, energy_fn, beta, mc.n_equilibration, k2,
        mc.step_size, L, N, D,
    )

    print(f"  MCMC: production ({mc.n_production_cycles} cycles at beta={beta:.3f})")
    key, k3 = jax.random.split(key)
    configs, _ = run_mcmc(
        configs, energy_fn, beta, mc.n_production_cycles, k3,
        mc.step_size, L, N, D,
    )

    return configs


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def ress(log_weights):
    """Relative effective sample size.

    RESS = 1 / (N * sum(softmax(log_w)^2))
    Range: [1/N, 1]. Higher is better.

    Args:
        log_weights: (N,) log importance weights.

    Returns:
        Scalar float.
    """
    log_w = log_weights - jax.scipy.special.logsumexp(log_weights)
    w = jnp.exp(log_w)
    return 1.0 / (log_weights.shape[0] * jnp.sum(w ** 2))


def octahedral_transform(key, dimensions):
    """Random element from the hyperoctahedral group (rotations + reflections).

    For 2D: random axis permutation (2!) x random sign flips (2^2) = 8 elements.

    Args:
        key: PRNG key.
        dimensions: spatial dimensionality.

    Returns:
        (D, D) orthogonal matrix.
    """
    k1, k2 = jax.random.split(key)
    # Random permutation matrix
    perm = jax.random.permutation(k1, dimensions)
    P = jnp.eye(dimensions)[perm]
    # Random sign flips
    signs = 2 * jax.random.bernoulli(k2, shape=(dimensions,)).astype(jnp.float32) - 1
    S = jnp.diag(signs)
    return P @ S
