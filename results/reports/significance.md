# Significance summary

## Null models

- Direction randomization: per-point sign flips (50/50) on bidirectional traces; N=300.
- Phase scramble: randomize FFT phases per segment (amplitude preserved); N=100, phase_all_segments=False, phase_full_pass=False, phase_max_segments=600 (stratified by x_name).
- Parity-bit null handling: eta_V_signed is randomly sign-flipped (50/50) per evaluation to emulate direction-label ambiguity.
- Parity stability null uses a binomial bootstrap over per-file parity counts; bootstrap_n=200.
- Z-score ranges use bootstrap over null replicates; z_boot=500.

## Observed statistics

- T_eta (median eta_norm): 1.15867
- T_b (median parity_stability): 1
- T_lock (fraction parity locks after knee): 0.996713
- T_lock computed on bidirectional segments with knee_valid=True.

## Z-scores

- direction_randomization: z_eta=59.42 (p5=55.85, p95=64.39), z_b=364.3 (p5=338.5, p95=407.8), z_lock=64.34 (p5=61.09, p95=68.5)
- direction_randomization: z_combined=374.7 (assumes approximate independence)
- phase_scramble: z_eta=72.94 (p5=66.47, p95=82.39), z_b=137.7 (p5=124.2, p95=159.7), z_lock=22.54 (p5=20.06, p95=26.24)
- phase_scramble: z_combined=157.4 (assumes approximate independence)

## Interpretation

- Under direction-randomized null models, observed odd-channel strength and post-knee parity locking deviate from null expectations at the z~55.8 level.
- Under phase-scrambled null models, observed odd-channel strength and post-knee parity locking deviate from null expectations at the z~20.1 level.
- Sigma here is algorithmic significance, not a particle-physics discovery claim; null definitions, N, and dependence notes are reported above.
