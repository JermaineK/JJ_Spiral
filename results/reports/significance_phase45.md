# Significance summary

- Pipeline version: 5.5

## Null models

- Direction randomization: per-point sign flips (50/50) on bidirectional traces; N=300.
- Phase scramble: randomize FFT phases per segment (amplitude preserved); N=100, phase_all_segments=False, phase_full_pass=False, phase_max_segments=600 (stratified by x_name).
- Parity-bit null handling: eta_V_signed is randomly sign-flipped (50/50) per evaluation to emulate direction-label ambiguity.
- Parity stability null uses a binomial bootstrap over per-file parity counts; bootstrap_n=200.
- Z-score ranges use bootstrap over null replicates; z_boot=500.

## Observed statistics

- T_eta (median eta_norm): 1.15852
- T_b (median parity_stability): 1
- T_lock (fraction parity locks after knee): 0.996713
- T_lock computed on bidirectional segments with knee_valid=True.

## Z-scores

- direction_randomization: z_eta=56.67 (p5=52.98, p95=60.85), z_b=440.3 (p5=390.1, p95=518.4), z_lock=56.21 (p5=52.4, p95=60.12)
- direction_randomization: z_combined=447.5 (assumes approximate independence)
- phase_scramble: z_eta=77.88 (p5=70.68, p95=88.77), z_b=159.6 (p5=138.9, p95=197.1), z_lock=27.08 (p5=24.39, p95=30.37)
- phase_scramble: z_combined=179.7 (assumes approximate independence)

## Interpretation

- Under direction-randomized null models, observed odd-channel strength and post-knee parity locking deviate from null expectations at the z~52.4 level.
- Under phase-scrambled null models, observed odd-channel strength and post-knee parity locking deviate from null expectations at the z~24.4 level.
- Sigma here is algorithmic significance, not a particle-physics discovery claim; null definitions, N, and dependence notes are reported above.
