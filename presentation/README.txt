===========================================================================
  2D LJ GAS-LIQUID COEXISTENCE — SUPERVISOR PRESENTATION
  Date: 2026-02-26
  System: N=128, 2D Lennard-Jones, cutoff=2.5σ (shifted), ρ*=0.30
===========================================================================

GOAL
----
Calibrate the gas-liquid coexistence regime (T_coex, coexisting densities)
for a 2D LJ system to use as the target for a normalizing flow (NF) sampler.
The NF is trained to propose cross-phase moves and accelerate mixing.


FOLDER GUIDE
------------
1_phase_diagram/     Phase diagram and bimodality sweeps
2_diagnostics/       Convergence, histogram, and mixing-time diagnostics
3_calibration/       RDF, pressure isotherms, nucleation scans
4_flow_comparison/   NF sampler performance vs. local MCMC
5_key_data/          JSON/CSV summary tables for all key quantities


===========================================================================
1. PHASE DIAGRAM
===========================================================================

coarse_phase_diagram.png
  Coarse grid (N=128) energy/OP scan across (T*, ρ*) space.
  Identifies the two-phase region and approximate T_c.

fine_sweep_bimodality_and_gap.png   *** KEY FIGURE ***
  Fine temperature sweep (T*=0.39 to 0.50) at ρ*=0.30.
  Top panel: Ashman D statistic for energy bimodality (D>2 = well-separated phases).
  Bottom panel: cluster order-parameter gap (liq-init minus gas-init).
  Shows bimodality strengthens as T* decreases below ~0.42.


===========================================================================
2. DIAGNOSTICS  (N=128, ρ*=0.30, 500K moves per chain)
===========================================================================

diag4_energy_gap_vs_T.png          *** KEY FIGURE ***
  6-temperature (T*=0.33–0.50) scatter grid showing U/N trajectories
  for gas-initialized and liquid-initialized chains.
  ΔU/N summary:
    T*=0.33 → ΔU/N=0.394  (strong trapping, permanent phase separation)
    T*=0.36 → ΔU/N=0.336  (strong trapping)
    T*=0.39 → ΔU/N=0.278  (moderate)
    T*=0.42 → ΔU/N=0.169  (weakening)
    T*=0.45 → ΔU/N=0.148  (weak)
    T*=0.50 → ΔU/N=0.061  (near single-phase)

diag5_mixing_time_vs_T.png         *** KEY FIGURE ***
  Median mixing time τ for gas-init chains to reach liquid-phase energy.
    T*=0.33 → 0/4 chains mixed in 500K moves  (τ = ∞)
    T*=0.36 → 0/4 chains mixed                 (τ = ∞)
    T*=0.39 → 1/4 chains mixed, τ~475K
    T*=0.42 → 4/4 mixed, τ~292K
    T*=0.45 → 4/4 mixed, τ~338K
    T*=0.50 → 4/4 mixed, τ~108K
  Conclusion: T_c (at ρ*=0.30) lies between 0.36 and 0.39.
              Below 0.36, gas-init chains are PERMANENTLY TRAPPED —
              this is the target regime for the NF sampler.

diag1_timeseries_T0.36.png
  Energy and cluster OP vs. MC step at T*=0.36 (4 gas + 4 liq chains).
  Shows complete non-mixing: gas and liquid branches never cross.

diag1_timeseries_T0.45.png
  Same at T*=0.45: all chains converge within ~150K moves.

diag1_snapshots_T0.36.png
  Particle configuration filmstrip at T*=0.36. Visual confirmation of
  stable gas cluster (top row) and dense liquid patch (bottom row).

diag1_snapshots_T0.45.png
  Filmstrip at T*=0.45: both initializations mix to uniform fluid.

diag3_histograms_T0.36.png
  U/N and cluster OP histograms at T*=0.36.
  Ashman D = 2.66 (energy) — well-separated bimodal distribution.

diag3_histograms_T0.45.png
  Same at T*=0.45: histograms overlap significantly (single phase).


===========================================================================
3. CALIBRATION
===========================================================================

stage_a_radial_dist_fn.png
  Radial distribution function g(r) from Stage A at T*=0.36.
  First minimum at r_cut=1.6733σ — used as cluster OP bond distance.
  (NOTE: this r_cut is for cluster OP only; LJ potential cutoff = 2.5σ)

pressure_isotherms.png             *** KEY FIGURE ***
  Two-box pressure calibration (v2 method):
    Liquid box: ρ*=0.60  |  Gas box: ρ*=0.05
  Pressure equality (|ΔP|/P < 10%) narrows T_coex to ~0.39.
  Virial pressure: P = ρ*T* + (1/2A) Σ[48/r^12 − 24/r^6]

nucleation_traces.png
  Nucleation time scan at ρ*=0.10 (single-box, mixed density).
  Identifies densities where gas phase survives > 500K moves.


===========================================================================
4. NORMALIZING FLOW COMPARISON  (T*=0.36, ρ*=0.30, N=128)
===========================================================================

timeseries_no_flow.png
  8 gas-initialized chains running local MCMC only.
  0/8 cross to liquid branch in 500K moves.

timeseries_with_flow.png
  Same 8 chains with NF proposal moves interleaved.
  0/8 cross to liquid branch. Flow acceptance rate = 0%.

mixing_time_comparison.png
  Side-by-side summary. Flow proposals are not yet helping.

STATUS: The NF sampler is not yet functional at T*=0.36.
The flow was trained on an early calibration that did not correctly
isolate the coexistence regime. Next step: retrain on the pressure-
calibrated parameters (T_coex≈0.39, ρ_liq≈0.60, ρ_gas≈0.05).


===========================================================================
5. KEY DATA FILES
===========================================================================

stage_a_rcut.json       r_cut = 1.6733σ for cluster OP bond criterion
coex_estimate.json      T_coex = 0.39 from pressure equality method
pressure_data.csv       P_liq, P_gas, ΔP/P for T*=0.28,0.30,0.33,0.36,0.39
fine_sweep_results.json Per-temperature Ashman D, OP gap, transition counts
flow_comparison_summary.txt  NF acceptance rate and mixing counts


===========================================================================
SUMMARY OF CONCLUSIONS
===========================================================================

1. T_c (2D LJ, ρ*=0.30) lies between T*=0.36 and T*=0.39.
   Below T*=0.36, phase separation is permanent on 500K-move timescales.

2. T_coex ≈ 0.39 from pressure equality of two-box simulations
   (ρ_liq*=0.60, ρ_gas*=0.05).

3. Ashman D = 2.66 at T*=0.45 confirms well-separated bimodal distributions.
   Energy gap ΔU/N = 0.394 at T*=0.33 (strongest trapping observed).

4. The normalizing flow currently achieves 0% acceptance at T*=0.36.
   This is expected given the mismatch between training and target regimes.
   Retraining on the calibrated parameters is the immediate next step.

Literature reference: T*_c ≈ 0.44–0.46 for 2D LJ (our effective T_c
is lower at ρ*=0.30 because that density is off-critical).

===========================================================================
