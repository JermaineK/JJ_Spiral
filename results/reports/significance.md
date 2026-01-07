# Significance summary

## Null models

- Direction randomization: per-point sign flips (50/50) on bidirectional traces; N=30.
- Phase scramble: randomize FFT phases per segment (amplitude preserved); N=5, phase_all_segments=False, phase_full_pass=False, phase_max_segments=600.
- Parity-bit null handling: eta_V_signed is randomly sign-flipped (50/50) per evaluation to emulate direction-label ambiguity.
- Parity stability null uses a binomial bootstrap over per-file parity counts; bootstrap_n=200.

## Observed statistics

- T_eta (median eta_norm): 1.15867
- T_b (median parity_stability): 1
- T_lock (fraction parity locks after knee): 0.996713

## Z-scores

- direction_randomization: z_eta=55.77, z_b=120.4, z_lock=69.89
- direction_randomization: z_combined=150 (assumes approximate independence)
- phase_scramble: z_eta=88.08, z_b=14.21, z_lock=17.31
- phase_scramble: z_combined=90.89 (assumes approximate independence)

## Interpretation

- Under direction-randomized null models, observed odd-channel strength and post-knee parity locking deviate from null expectations at the z~55.8 level.
- Under phase-scrambled null models, observed odd-channel strength and post-knee parity locking deviate from null expectations at the z~17.3 level.
- Sigma here is algorithmic significance, not a particle-physics discovery claim; null definitions, N, and dependence notes are reported above.
