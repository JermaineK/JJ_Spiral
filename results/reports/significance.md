# Significance summary

- Pipeline version: 5.5

## Null models

- Direction randomization: per-point sign flips (50/50) on bidirectional traces; N=300, direction_max_segments=600 (stratified by x_name).
- Phase scramble: randomize FFT phases per segment (amplitude preserved); N=100, phase_all_segments=False, phase_full_pass=False, phase_max_segments=600 (stratified by x_name).
- Parity-bit null handling: eta_V_signed is randomly sign-flipped (50/50) per evaluation to emulate direction-label ambiguity.
- Parity stability null uses per-iteration medians across files (no bootstrap on counts).
- Z-score ranges use bootstrap over null replicates; z_boot=500.

## Observed statistics

- T_eta (median eta_norm): 1.15852
- T_b (median parity_stability): 1
- T_lock (fraction parity locks after knee): 0.996713
- T_lock computed on bidirectional segments with knee_valid=True.

## Z-scores

- direction_randomization: z_eta=18.14 (p5=17.1, p95=19.46), z_b=NA (p5=NA, p95=NA), z_lock=21 (p5=19.59, p95=22.39)
- phase_scramble: z_eta=72.91 (p5=65.86, p95=81.85), z_b=NA (p5=NA, p95=NA), z_lock=25.13 (p5=22.97, p95=28.3)

## Covariance-aware combined statistic

- direction_randomization: Q_obs=844.8, p_emp=0.003322, z_equiv=2.714, n=300
- phase_scramble: Q_obs=7055, p_emp=0.009901, z_equiv=2.33, n=100

## Interpretation

- Under direction-randomized null models, observed odd-channel strength and post-knee parity locking deviate from null expectations at the z~17.1 level.
- Under phase-scrambled null models, observed odd-channel strength and post-knee parity locking deviate from null expectations at the z~23 level.
- Sigma here is algorithmic significance, not a particle-physics discovery claim; null definitions, N, and dependence notes are reported above.
