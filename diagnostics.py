"""diagnostics.py — Measurement and visualisation toolkit for 2D LJ coexistence.

All PBC operations use minimum image convention: dr = dr - L*round(dr/L).
Functions accept numpy or JAX arrays and return numpy scalars/arrays unless
noted.  JAX is used only where an energy_fn call is needed.

Public API:
    radial_distribution_function(coords, N, D, L, n_bins=100)
    find_rcut_from_gr(r_vals, gr_vals, r_min=1.0, r_max=2.0)
    largest_cluster_fraction(coords, N, D, L, r_cut)
    mean_coordination(coords, N, D, L, r_cut)
    potential_energy_per_particle(configs, energy_fn, N)
    snapshot_plot(coords, N, D, L, r_cut=None, title="", save_path=None, ax=None)
    multi_panel_diagnostic(configs_batch, N, D, L, r_cut, energy_fn, title, save_path)
    make_liquid_init(N, D, L, rng_key=None, energy_fn=None)
    make_gas_init(N, D, L, rng_key=None, energy_fn=None)
    measure_acceptance_rate(configs, energy_fn, beta, step_size, n_moves, key, N, D, L)
    calibrate_step_size(configs, energy_fn, beta, N, D, L, key, ...)
    compute_op_batch(configs, N, D, L, r_cut)
    ashmans_d(values)
    plot_op_timeseries(op_liq, op_gas, title, save_path)
"""

import warnings
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import signal
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

import jax
import jax.numpy as jnp

from physics import fcc_lattice


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _np(x):
    """Convert JAX / numpy array to plain numpy."""
    return np.asarray(x)


def _pbc_dist_matrix(coords, N, D, L):
    """All N*(N-1)/2 pairwise distances under PBC for a single config.

    Args:
        coords: (N*D,) flat array.
    Returns:
        dists:  (n_pairs,) distances.
        i_idx, j_idx: (n_pairs,) upper-triangle pair indices.
    """
    pos = _np(coords).reshape(N, D)
    i_idx, j_idx = np.triu_indices(N, k=1)
    dr = pos[i_idx] - pos[j_idx]
    dr -= L * np.round(dr / L)
    dists = np.sqrt((dr ** 2).sum(axis=-1))
    return dists, i_idx, j_idx


def _pbc_dist_batch(coords, N, D, L):
    """Pairwise distances for a batch of configs.

    Args:
        coords: (B, N*D).
    Returns:
        dists: (B, n_pairs).
    """
    B = coords.shape[0]
    pos = _np(coords).reshape(B, N, D)
    i_idx, j_idx = np.triu_indices(N, k=1)
    dr = pos[:, i_idx, :] - pos[:, j_idx, :]   # (B, n_pairs, D)
    dr -= L * np.round(dr / L)
    return np.sqrt((dr ** 2).sum(axis=-1))      # (B, n_pairs)


# ---------------------------------------------------------------------------
# 1a) Radial distribution function
# ---------------------------------------------------------------------------

def radial_distribution_function(coords, N, D, L, n_bins=100):
    """g(r) averaged over a batch of 2D configurations.

    2D normalisation:
        g(r) = <n_pairs(r,r+dr)> / [N * rho * 2*pi*r * dr]
    where the average is over the batch and rho = N / L^D.

    Args:
        coords: (B, N*D) or (N*D,) — configurations.
        N, D, L: system parameters.
        n_bins: number of bins in [0, L/2].
    Returns:
        r_vals: (n_bins,) bin centres.
        gr_vals: (n_bins,) g(r).
    """
    coords = _np(coords)
    if coords.ndim == 1:
        coords = coords[None]
    B = coords.shape[0]

    r_max = L / 2.0
    edges = np.linspace(0.0, r_max, n_bins + 1)
    r_vals = 0.5 * (edges[:-1] + edges[1:])
    dr = edges[1] - edges[0]

    dists = _pbc_dist_batch(coords, N, D, L)  # (B, n_pairs)
    hist, _ = np.histogram(dists.ravel(), bins=edges)
    # Average pair count per config per bin
    avg_counts = hist / B

    rho = N / (L ** D)
    if D == 2:
        norm = N * rho * 2.0 * np.pi * r_vals * dr
    else:
        norm = N * rho * 4.0 * np.pi * r_vals ** 2 * dr

    with np.errstate(divide='ignore', invalid='ignore'):
        gr_vals = np.where(norm > 0, avg_counts / norm, 0.0)

    return r_vals, gr_vals


# ---------------------------------------------------------------------------
# 1b) Find r_cut from g(r)
# ---------------------------------------------------------------------------

def find_rcut_from_gr(r_vals, gr_vals, r_min=1.0, r_max=2.0):
    """First minimum of g(r) in [r_min, r_max].

    Args:
        r_vals, gr_vals: output of radial_distribution_function.
        r_min, r_max: search window (sigma units).
    Returns:
        r_cut: float.  Falls back to position of minimum value with a
               warning if no clear local minimum is found.
    """
    mask = (r_vals >= r_min) & (r_vals <= r_max)
    r_w = r_vals[mask]
    g_w = gr_vals[mask]

    if len(r_w) < 3:
        warnings.warn(
            f"find_rcut_from_gr: window [{r_min}, {r_max}] too narrow "
            f"(only {len(r_w)} bins). Returning r_cut=1.5.")
        return 1.5

    minima, _ = signal.find_peaks(-g_w)

    if len(minima) == 0:
        idx = int(np.argmin(g_w))
        r_cut = float(r_w[idx])
        warnings.warn(
            f"find_rcut_from_gr: no clear local minimum in [{r_min}, {r_max}]. "
            f"Using argmin position r_cut={r_cut:.3f}. Verify the g(r) plot.")
        return r_cut

    # Return the first (lowest-r) minimum
    return float(r_w[minima[0]])


# ---------------------------------------------------------------------------
# 1c) Largest cluster fraction
# ---------------------------------------------------------------------------

def largest_cluster_fraction(coords, N, D, L, r_cut):
    """Fraction of particles in the largest connected cluster.

    Connectivity defined by minimum-image distance < r_cut.

    Args:
        coords: (N*D,) or (B, N*D).
    Returns:
        scalar float  if input is (N*D,).
        (B,) ndarray  if input is (B, N*D).
    """
    coords = _np(coords)
    scalar = (coords.ndim == 1)
    if scalar:
        coords = coords[None]
    B = coords.shape[0]

    dists = _pbc_dist_batch(coords, N, D, L)  # (B, n_pairs)
    i_idx, j_idx = np.triu_indices(N, k=1)

    fractions = np.zeros(B)
    for b in range(B):
        bonded = dists[b] < r_cut
        rows = np.concatenate([i_idx[bonded], j_idx[bonded]])
        cols = np.concatenate([j_idx[bonded], i_idx[bonded]])
        data = np.ones(len(rows), dtype=np.float32)
        adj = csr_matrix((data, (rows, cols)), shape=(N, N))
        _, labels = connected_components(adj, directed=False)
        sizes = np.bincount(labels)
        fractions[b] = sizes.max() / N

    return float(fractions[0]) if scalar else fractions


def compute_op_batch(configs, N, D, L, r_cut):
    """Largest cluster fraction for every config in a batch.

    Convenience wrapper around largest_cluster_fraction.

    Args:
        configs: (B, N*D).
    Returns:
        (B,) numpy array.
    """
    return largest_cluster_fraction(configs, N, D, L, r_cut)


# ---------------------------------------------------------------------------
# 1d) Mean coordination number
# ---------------------------------------------------------------------------

def mean_coordination(coords, N, D, L, r_cut):
    """Average number of neighbours per particle within r_cut.

    Args:
        coords: (N*D,) or (B, N*D).
    Returns:
        scalar or (B,) array.
    """
    coords = _np(coords)
    scalar = (coords.ndim == 1)
    if scalar:
        coords = coords[None]

    dists = _pbc_dist_batch(coords, N, D, L)
    within = (dists < r_cut).sum(axis=1)  # (B,) pair counts
    coord = 2.0 * within / N             # each pair counted once → x2

    return float(coord[0]) if scalar else coord.astype(float)


# ---------------------------------------------------------------------------
# 1e) Potential energy per particle
# ---------------------------------------------------------------------------

def potential_energy_per_particle(configs, energy_fn, N):
    """U(x) / N for each configuration.

    Args:
        configs: (B, N*D) array (numpy or JAX).
        energy_fn: callable (B, N*D) -> (B,)  [use_lrc=False].
        N: number of particles.
    Returns:
        (B,) numpy array.
    """
    configs = jnp.array(configs)
    if configs.ndim == 1:
        configs = configs[None]
    return _np(energy_fn(configs)) / N


# ---------------------------------------------------------------------------
# 1f) Snapshot plot
# ---------------------------------------------------------------------------

def snapshot_plot(coords, N, D, L, r_cut=None, title="",
                  save_path=None, ax=None):
    """2D particle scatter.  Coloured by cluster and with bonds if r_cut given.

    Args:
        coords:    (N*D,) flat configuration.
        N, D, L:   system parameters.
        r_cut:     bond/cluster threshold (optional).
        title:     axis title.
        save_path: if given and ax is None, save and close the figure.
        ax:        existing Axes to draw into (optional).
    Returns:
        (fig, ax)
    """
    pos = _np(coords).reshape(N, D)
    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=(4.5, 4.5))
    else:
        fig = ax.figure

    half = L / 2.0
    ax.set_xlim(-half, half)
    ax.set_ylim(-half, half)
    ax.set_aspect('equal')
    ax.set_title(title, fontsize=8)
    ax.set_xlabel('x / σ', fontsize=7)
    ax.set_ylabel('y / σ', fontsize=7)
    ax.tick_params(labelsize=6)

    # Box outline
    rect = plt.Rectangle((-half, -half), L, L,
                         linewidth=1, edgecolor='black', facecolor='none')
    ax.add_patch(rect)

    if r_cut is not None:
        dists, i_idx, j_idx = _pbc_dist_matrix(coords, N, D, L)
        bonded = dists < r_cut

        # Cluster colouring
        rows = np.concatenate([i_idx[bonded], j_idx[bonded]])
        cols = np.concatenate([j_idx[bonded], i_idx[bonded]])
        data = np.ones(len(rows), dtype=np.float32)
        adj = csr_matrix((data, (rows, cols)), shape=(N, N))
        _, labels = connected_components(adj, directed=False)
        cmap = plt.get_cmap('tab20')
        colours = [cmap(labels[i] % 20) for i in range(N)]

        # Draw bonds (split at PBC boundary to avoid cross-box lines)
        for k in range(len(i_idx)):
            if bonded[k]:
                pi, pj = pos[i_idx[k]], pos[j_idx[k]]
                dr = pj - pi
                dr -= L * np.round(dr / L)
                mid = pi + dr / 2.0
                ax.plot([pi[0], mid[0]], [pi[1], mid[1]],
                        '-', color='#aaaaaa', lw=0.4, alpha=0.6, zorder=1)
                ax.plot([pj[0], pj[0] - dr[0] / 2.0],
                        [pj[1], pj[1] - dr[1] / 2.0],
                        '-', color='#aaaaaa', lw=0.4, alpha=0.6, zorder=1)

        ax.scatter(pos[:, 0], pos[:, 1], c=colours, s=60, zorder=2,
                   edgecolors='k', linewidths=0.3)
    else:
        ax.scatter(pos[:, 0], pos[:, 1], c='steelblue', s=60, zorder=2,
                   edgecolors='k', linewidths=0.3)

    if own_fig and save_path is not None:
        fig.savefig(save_path, dpi=100, bbox_inches='tight')
        plt.close(fig)

    return fig, ax


# ---------------------------------------------------------------------------
# 1g) Multi-panel diagnostic figure
# ---------------------------------------------------------------------------

def multi_panel_diagnostic(configs_batch, N, D, L, r_cut, energy_fn,
                           title="", save_path=None):
    """Single figure: 4 snapshots | OP histogram | energy histogram | g(r) | summary.

    Layout: 2 rows × 4 cols.
      Row 0: 4 evenly-spaced snapshots from the batch.
      Row 1: OP histogram | U/N histogram | g(r) | text summary.

    Args:
        configs_batch: (B, N*D).
        N, D, L, r_cut: system parameters.
        energy_fn: callable (B, N*D) -> (B,)  [use_lrc=False].
        title: figure suptitle.
        save_path: path to save the figure.
    Returns:
        fig
    """
    configs_batch = _np(configs_batch)
    B = configs_batch.shape[0]

    fig = plt.figure(figsize=(16, 8))
    fig.suptitle(title, fontsize=11, y=1.01)

    # --- Row 0: snapshots ---
    snap_idx = np.linspace(0, B - 1, 4, dtype=int)
    for col, idx in enumerate(snap_idx):
        ax = fig.add_subplot(2, 4, col + 1)
        snapshot_plot(configs_batch[idx], N, D, L, r_cut=r_cut,
                      title=f'sample {idx}', ax=ax)

    # --- Compute observables ---
    op = compute_op_batch(configs_batch, N, D, L, r_cut)
    u_pp = potential_energy_per_particle(configs_batch, energy_fn, N)
    r_vals, gr_vals = radial_distribution_function(configs_batch, N, D, L)
    coord = mean_coordination(configs_batch, N, D, L, r_cut)

    # --- Row 1, col 0: OP histogram ---
    ax_op = fig.add_subplot(2, 4, 5)
    ax_op.hist(op, bins=30, color='steelblue', edgecolor='white', lw=0.4)
    ax_op.axvline(op.mean(), color='red', lw=1.5, ls='--', label=f'mean={op.mean():.2f}')
    ax_op.set_xlabel('Largest cluster fraction (OP)', fontsize=8)
    ax_op.set_ylabel('Count', fontsize=8)
    ax_op.set_title(f'OP  μ={op.mean():.3f}  σ={op.std():.3f}', fontsize=9)
    ax_op.legend(fontsize=7)

    # --- Row 1, col 1: energy histogram ---
    ax_e = fig.add_subplot(2, 4, 6)
    ax_e.hist(u_pp, bins=30, color='darkorange', edgecolor='white', lw=0.4)
    ax_e.axvline(u_pp.mean(), color='red', lw=1.5, ls='--', label=f'mean={u_pp.mean():.2f}')
    ax_e.set_xlabel('U / N  (ε)', fontsize=8)
    ax_e.set_ylabel('Count', fontsize=8)
    ax_e.set_title(f'U/N  μ={u_pp.mean():.3f}  σ={u_pp.std():.3f}', fontsize=9)
    ax_e.legend(fontsize=7)

    # --- Row 1, col 2: g(r) ---
    ax_gr = fig.add_subplot(2, 4, 7)
    ax_gr.plot(r_vals, gr_vals, 'k-', lw=1.5)
    ax_gr.axhline(1.0, color='gray', lw=1, ls='--', alpha=0.6)
    if r_cut is not None:
        ax_gr.axvline(r_cut, color='red', lw=1.5, ls=':',
                      label=f'r_cut={r_cut:.2f}')
        ax_gr.legend(fontsize=7)
    ax_gr.set_xlabel('r / σ', fontsize=8)
    ax_gr.set_ylabel('g(r)', fontsize=8)
    ax_gr.set_title('Radial distribution function', fontsize=9)
    ax_gr.set_xlim(0, L / 2)

    # --- Row 1, col 3: text summary ---
    ax_t = fig.add_subplot(2, 4, 8)
    ax_t.axis('off')
    coord_mean = coord.mean() if hasattr(coord, 'mean') else coord
    summary = (
        f"N={N}, D={D}\n"
        f"L={L:.3f} σ\n"
        f"B={B} samples\n"
        f"r_cut={r_cut:.3f} σ\n\n"
        f"OP:    {op.mean():.3f} ± {op.std():.3f}\n"
        f"U/N:   {u_pp.mean():.3f} ± {u_pp.std():.3f} ε\n"
        f"Coord: {coord_mean:.2f} neighbours"
    )
    ax_t.text(0.05, 0.95, summary, transform=ax_t.transAxes,
              fontsize=9, va='top', family='monospace',
              bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.6))

    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=100, bbox_inches='tight')
        plt.close(fig)

    return fig


# ---------------------------------------------------------------------------
# 1h) Liquid initialisation
# ---------------------------------------------------------------------------

def make_liquid_init(N, D, L, rng_key=None, energy_fn=None):
    """Dense triangular lattice at rho≈0.75, noise 0.01 σ.

    Uses a sub-box L_tight = sqrt(N/0.75) << L so the lattice sits well
    inside [-L/2, L/2] with no particles near the PBC boundary.  This avoids
    the near-zero PBC pair distances that arise when fcc_lattice is scaled to
    the full simulation box.

    At rho=0.75 the nearest-neighbour spacing is ~1.15 σ (just above the LJ
    minimum at 2^{1/6} ≈ 1.12 σ), giving a negative initial energy suitable
    for liquid-phase equilibration.

    Args:
        N, D, L: system parameters (L is the simulation box; lattice is smaller).
        rng_key: JAX PRNG key for noise (numpy seed 42 if None).
        energy_fn: if given, abort on non-finite energy.
    Returns:
        (N*D,) JAX float32 array.
    """
    L_tight = float(np.sqrt(N / 0.75))          # rho=0.75 sub-box
    lattice = _np(fcc_lattice(N, D, L_tight))    # (N, D), centred in L_tight

    # L_tight < L so all positions are within [-L/2, L/2] — no PBC wrapping
    # needed and no risk of opposite-edge particles colliding under min-image.
    if rng_key is not None:
        noise = 0.01 * _np(jax.random.normal(rng_key, (N, D)))
    else:
        noise = 0.01 * np.random.default_rng(42).standard_normal((N, D))

    pos = (lattice + noise).astype(np.float32)
    flat = pos.ravel()

    if energy_fn is not None:
        e = float(energy_fn(jnp.array(flat)[None])[0])
        if not np.isfinite(e):
            raise RuntimeError(
                f"make_liquid_init: energy not finite ({e}). "
                "This indicates a hard-core overlap in the initial lattice.")

    return jnp.array(flat)


# ---------------------------------------------------------------------------
# 1i) Gas initialisation
# ---------------------------------------------------------------------------

def make_gas_init(N, D, L, rng_key=None, energy_fn=None):
    """Full-box triangular lattice with noise 0.01 σ — gas-phase starting point.

    Mirrors the initialisation in physics.generate_dataset: fcc_lattice scaled
    to the simulation box, plus small noise, wrapped under PBC.  This puts
    particles uniformly throughout the box at the mean density, appropriate
    for gas-phase MCMC.

    The initial energy will be large positive (due to PBC edge particles being
    close under min-image) but the hot-equilibration stage of MCMC rapidly
    thermalises the system away from this state.

    Args:
        N, D, L: system parameters.
        rng_key: JAX PRNG key (numpy seed 123 if None).
        energy_fn: if given, abort on non-finite energy.
    Returns:
        (N*D,) JAX float32 array.
    """
    lattice = _np(fcc_lattice(N, D, L))         # full-box lattice

    if rng_key is not None:
        noise = 0.01 * _np(jax.random.normal(rng_key, (N, D)))
    else:
        noise = 0.01 * np.random.default_rng(123).standard_normal((N, D))

    pos = lattice + noise
    pos = pos - L * np.round(pos / L)           # PBC wrap
    flat = pos.ravel().astype(np.float32)

    if energy_fn is not None:
        e = float(energy_fn(jnp.array(flat)[None])[0])
        if not np.isfinite(e):
            raise RuntimeError(
                f"make_gas_init: energy not finite ({e}). "
                "Hard-core overlap in initial lattice.")

    return jnp.array(flat)


# ---------------------------------------------------------------------------
# Acceptance rate measurement (for step_size calibration)
# ---------------------------------------------------------------------------

def measure_acceptance_rate(configs, energy_fn, beta, step_size,
                            n_moves, key, N, D, L):
    """Run a short MCMC and return the fraction of accepted proposals.

    Uses a Python loop (not jax.lax.scan) so accepts can be counted
    explicitly.  Intended only for short calibration runs.

    Args:
        configs:    (B, N*D) initial configurations.
        energy_fn:  callable (B, N*D) -> (B,).
        beta:       inverse temperature.
        step_size:  Gaussian displacement σ.
        n_moves:    total single-particle moves to attempt.
        key:        JAX PRNG key.
        N, D, L:    system parameters.
    Returns:
        acceptance_rate: float in [0, 1].
    """
    configs = jnp.array(configs)
    B = configs.shape[0]
    energies = energy_fn(configs)

    n_accepted = 0

    for _ in range(n_moves):
        key, k1, k2, k3 = jax.random.split(key, 4)

        particle_idx = jax.random.randint(k1, (B,), 0, N)
        disp = step_size * jax.random.normal(k2, (B, D))

        shaped = configs.reshape(B, N, D)
        one_hot = jax.nn.one_hot(particle_idx, N)           # (B, N)
        delta = one_hot[:, :, None] * disp[:, None, :]      # (B, N, D)
        new_shaped = shaped + delta
        new_shaped = new_shaped - L * jnp.round(new_shaped / L)
        new_configs = new_shaped.reshape(B, N * D)

        new_energies = energy_fn(new_configs)
        log_accept = -beta * (new_energies - energies)
        log_u = jnp.log(jax.random.uniform(k3, (B,)))
        accept = log_u < log_accept

        configs = jnp.where(accept[:, None], new_configs, configs)
        energies = jnp.where(accept, new_energies, energies)
        n_accepted += int(accept.sum())

    return n_accepted / (n_moves * B)


def calibrate_step_size(configs, energy_fn, beta, N, D, L, key,
                        candidates=(0.05, 0.1, 0.2, 0.4),
                        target_rate=0.35, n_test=2000):
    """Choose the step_size from candidates closest to target acceptance rate.

    Runs n_test single-particle moves per candidate on the provided configs.

    Args:
        configs:     (B, N*D) test configurations.
        energy_fn:   callable.
        beta:        inverse temperature.
        N, D, L:     system parameters.
        key:         PRNG key.
        candidates:  step sizes to try.
        target_rate: desired acceptance rate.
        n_test:      moves per candidate.
    Returns:
        best_step: float — the winning step size.
        rates: dict {step_size: measured_rate}.
    """
    rates = {}
    for step in candidates:
        key, subkey = jax.random.split(key)
        rate = measure_acceptance_rate(
            configs, energy_fn, beta, step, n_test, subkey, N, D, L)
        rates[float(step)] = float(rate)

    best = min(candidates, key=lambda s: abs(rates[float(s)] - target_rate))
    return float(best), rates


# ---------------------------------------------------------------------------
# Ashman's D (bimodality statistic)
# ---------------------------------------------------------------------------

def ashmans_d(values):
    """Ashman's D statistic via a 2-component Gaussian mixture.

        D = |μ₁ − μ₂| / sqrt((σ₁² + σ₂²) / 2)

    D > 2 indicates clean bimodality.

    Requires scikit-learn.

    Args:
        values: (N,) 1-D array.
    Returns:
        D:   float (nan if sklearn unavailable or fit fails).
        gmm: fitted GaussianMixture, or None.
    """
    try:
        from sklearn.mixture import GaussianMixture
    except ImportError:
        warnings.warn(
            "scikit-learn not installed; Ashman's D unavailable. "
            "Install with: pip install scikit-learn")
        return float('nan'), None

    values = np.asarray(values).reshape(-1, 1)
    if len(values) < 4:
        warnings.warn("ashmans_d: fewer than 4 samples — returning nan.")
        return float('nan'), None

    try:
        gmm = GaussianMixture(n_components=2, random_state=0, n_init=10)
        gmm.fit(values)
    except Exception as exc:
        warnings.warn(f"ashmans_d: GMM fit failed ({exc}).")
        return float('nan'), None

    mu = gmm.means_.ravel()
    sigma2 = gmm.covariances_.ravel()
    D = abs(mu[0] - mu[1]) / np.sqrt((sigma2[0] + sigma2[1]) / 2.0)
    return float(D), gmm


# ---------------------------------------------------------------------------
# OP time series plot
# ---------------------------------------------------------------------------

def plot_op_timeseries(op_liq, op_gas, title="", save_path=None):
    """OP time series: red = liquid-init chains, blue = gas-init chains.

    Args:
        op_liq: (n_chains, n_steps) OP values from liquid-init runs.
        op_gas: (n_chains, n_steps) OP values from gas-init runs.
        title:  figure title.
        save_path: if given, save and close figure.
    Returns:
        fig
    """
    op_liq = np.asarray(op_liq)
    op_gas = np.asarray(op_gas)

    fig, ax = plt.subplots(figsize=(12, 4))
    for i, ts in enumerate(op_liq):
        ax.plot(ts, color='red', alpha=0.35, lw=0.6,
                label='liquid init' if i == 0 else None)
    for i, ts in enumerate(op_gas):
        ax.plot(ts, color='steelblue', alpha=0.35, lw=0.6,
                label='gas init' if i == 0 else None)

    ax.set_ylim(0, 1)
    ax.axhline(0.5, color='gray', lw=1, ls='--', alpha=0.5)
    ax.set_xlabel('Sample index')
    ax.set_ylabel('Largest cluster fraction (OP)')
    ax.set_title(title)
    ax.legend(loc='upper right', fontsize=8)

    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=100, bbox_inches='tight')
        plt.close(fig)

    return fig
