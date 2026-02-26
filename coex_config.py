"""Configuration for the LJ gas-liquid coexistence normalising flow.

Calibrated from diagnostic runs (2026-02-25):
  Base:   T*=0.50, rho*=0.30  — single-phase fluid, median mixing time 108K moves
  Target: T*=0.36, rho*=0.30  — coexistence; gas-init chains NEVER mix in 500K moves
  N=128, D=2, L=sqrt(N/rho)=20.656, use_lrc=False [A1]
"""
import math
from config import (SystemConfig, EnergyConfig, FlowConfig,
                    TrainConfig, MCMCConfig, PipelineConfig)

N        = 128
D        = 2
RHO      = 0.30
L        = math.sqrt(N / RHO)   # ≈ 20.656σ

T_BASE   = 0.50   # beta_source = 2.0
T_TARGET = 0.36   # beta_target ≈ 2.778

R_CUT_OP = 1.6733  # cluster OP cutoff (from Stage A g(r)); NOT the LJ cutoff


def make_coex_config(
    n_epochs=200,
    batch_size=512,
    n_samples=50_000,
    save_dir='./checkpoints/coex',
    data_dir='./data/coex',
    seed=42,
    w_xz=2.0,    # [A2]: weight NLL higher to prevent mode collapse onto one phase
    w_zx=1.0,
    n_blocks=8,
    model_type='correti',
) -> PipelineConfig:
    """Return PipelineConfig for LJ coexistence NF training."""
    return PipelineConfig(
        system=SystemConfig(
            n_particles=N,
            dimensions=D,
            rho=RHO,
            box_length=L,
        ),
        energy=EnergyConfig(
            use_lrc=False,  # [A1]: LRC assumes uniform density; invalid at coexistence
        ),
        flow=FlowConfig(
            model_type=model_type,
            n_blocks=n_blocks,
            n_bins=16,
            embedding_size=128,
            transformer_depth=1,
            transformer_heads=2,
            n_freqs=8,
            use_circular_shift=True,
            permute_variables=True,
        ),
        train=TrainConfig(
            n_epochs=n_epochs,
            batch_size=batch_size,
            lr=1e-4,
            milestones_epochs=(120, 175),
            gamma=0.1,
            w_xz=w_xz,
            w_zx=w_zx,
            n_dump=5,
            n_save=25,
        ),
        mcmc=MCMCConfig(
            step_size=0.20,
            n_equilibration=10_000,
            n_production_cycles=1_000,
            n_samples=n_samples,
            n_cached=int(n_samples * 0.9),
            test_fraction=0.10,
            refresh_cycles=1_000,
        ),
        backend='jax',
        seed=seed,
        save_dir=save_dir,
        data_dir=data_dir,
        beta_source=1.0 / T_BASE,
        beta_target=1.0 / T_TARGET,
    )
