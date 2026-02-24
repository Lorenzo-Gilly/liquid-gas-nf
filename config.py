"""Shared configuration dataclasses for the NF training pipeline."""

from dataclasses import dataclass, field
import math


@dataclass
class SystemConfig:
    n_particles: int = 32
    dimensions: int = 2
    rho: float = 0.7346
    box_length: float = 6.6  # sqrt(N / rho) for 2D


@dataclass
class EnergyConfig:
    epsilon: float = 1.0
    sigma: float = 1.0
    cutin: float = 0.8
    tol: float = 1e-12
    # WCA cutoff = 2^(1/6) * sigma
    wca_cutoff: float = 1.1224620483093730  # 2**(1/6)
    lj_cutoff: float = 2.5
    use_lrc: bool = True  # long-range correction for LJ


@dataclass
class FlowConfig:
    model_type: str = "lorenzo"  # "lorenzo" or "correti"
    n_blocks: int = 8  # "super-blocks"; total layers depends on model_type
    n_bins: int = 16
    embedding_size: int = 128
    transformer_depth: int = 1
    transformer_heads: int = 2
    n_freqs: int = 8
    init_std: float = 0.01
    use_circular_shift: bool = True
    permute_variables: bool = True


@dataclass
class TrainConfig:
    n_epochs: int = 250
    batch_size: int = 512
    lr: float = 1e-4
    # LR milestones in *epochs* (converted to steps internally)
    milestones_epochs: tuple = (150, 225)
    gamma: float = 0.1  # LR decay factor at each milestone
    w_xz: float = 1.0
    w_zx: float = 1.0
    n_dump: int = 1  # validation frequency (epochs)
    n_save: int = 5  # checkpoint frequency (epochs)


@dataclass
class MCMCConfig:
    step_size: float = 0.2
    n_equilibration: int = 5000
    n_production_cycles: int = 1000
    n_samples: int = 100_000
    n_cached: int = 90_000
    test_fraction: float = 0.1
    # Dynamic prior refresh
    refresh_cycles: int = 1000


@dataclass
class PipelineConfig:
    system: SystemConfig = field(default_factory=SystemConfig)
    energy: EnergyConfig = field(default_factory=EnergyConfig)
    flow: FlowConfig = field(default_factory=FlowConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    mcmc: MCMCConfig = field(default_factory=MCMCConfig)
    backend: str = "jax"
    seed: int = 42
    save_dir: str = "./checkpoints/lorenzo"
    data_dir: str = "./data"
    # Physics
    beta_source: float = 0.5   # WCA: T=2, beta=1/T=0.5
    beta_target: float = 1.0   # LJ:  T=1, beta=1/T=1.0

    @property
    def n_dof(self) -> int:
        """Degrees of freedom after removing the origin particle."""
        return (self.system.n_particles - 1) * self.system.dimensions

    @property
    def batches_per_epoch(self) -> int:
        n_train = int(self.mcmc.n_samples * (1 - self.mcmc.test_fraction))
        return max(n_train // self.train.batch_size, 1)

    @property
    def milestone_steps(self) -> tuple:
        bpe = self.batches_per_epoch
        return tuple(ep * bpe for ep in self.train.milestones_epochs)
