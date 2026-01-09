# Phase 5.5 Summary

- Pipeline version: 5.5
- Input folder: Data_for_analysis\D4.2
- Total segments: 4813
- Parity-valid segments: 1602

## Primary observables

- delta_coherence_gradient: n=1547, median=0.417348
- knee_alignment_gain_odd_ratio_post: n=1602, median=1.00242
- knee_alignment_gain_coherence_post: n=1547, median=1

## Directional sensitivity

- I_knee_norm: score=None, perm_p=None
- odd_ratio_post: score=None, perm_p=None
- coherence_gradient_post: score=None, perm_p=None

## Null comparisons

- deltaG vs phase_scramble: observed=0.4173, null_mean=0.01717, null_std=0.04216, z=9.49
- knee_alignment_gain vs phase_scramble: observed=1.002, null_mean=1.004, null_std=0.003109, z=-0.533
- knee_alignment_gain_coherence vs phase_scramble: observed=1, null_mean=1.016, null_std=0.02209, z=-0.734
- deltaG vs pair_breaking: observed=0.4173, null_mean=0.4601, null_std=0.005814, z=-7.36
- knee_alignment_gain vs pair_breaking: observed=1.002, null_mean=1.002, null_std=0.003109, z=-0.0177
- knee_alignment_gain_coherence vs pair_breaking: observed=1, null_mean=1, null_std=0, z=nan
- deltaG vs local_shuffle: observed=0.4173, null_mean=0.1839, null_std=0.003933, z=59.4
- knee_alignment_gain vs local_shuffle: observed=1.002, null_mean=1.007, null_std=0.002073, z=-2.13
- knee_alignment_gain_coherence vs local_shuffle: observed=1, null_mean=1, null_std=0, z=nan

## Summary table

Metric	Real	Phase-scramble	Pair-break
deltaG	0.4173	0.01717	0.4601
knee_alignment_gain	1.002	1.004	1.002
knee_alignment_gain_coherence	1	1.016	1

## Interpretation

- Focus is on knee-conditioned coherence change and alignment gain; raw odd_ratio is not treated as diagnostic.
- Nulls include phase scramble, pair-breaking, and local shuffle to separate organized structure from envelope effects.
