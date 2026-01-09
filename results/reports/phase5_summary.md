# Phase 5 Summary

- Pipeline version: 0.5.0
- Input folder: Data_for_analysis\D4.2
- Total files: 1603
- Total segments: 4813
- Parity-valid segments: 1602

## Core checks

- Files with both signs: 1603
- Median sweep monotonicity (file-order): 0.502
- Median overlap fraction: 0.365
- Non-monotonic sweeps: 1603
- Non-monotonic examples: D4.2__Fig 1__IV_data__IV_BS16_C1S1_Vpp_7p500_FRQ_11_AVG_100_Vgain_2E3_Igain_1E5_Rb_0E0_Rser_0E0_____Temp_2.740E-1K_MEASres_1p4E1_Thu,May2,2019_82950PM, D4.2__Fig 2__SQUID_data__IVB_BS30_A1S1_I_coil_+-0.3mA_fine_284mK_191103__IVB__T_1p40K[-0.000000750], D4.2__Fig 2__SQUID_data__IVB_BS30_A1S1_I_coil_+-0.3mA_fine_284mK_191103__IVB__T_1p40K[-0.000001500], D4.2__Fig 2__SQUID_data__IVB_BS30_A1S1_I_coil_+-0.3mA_fine_284mK_191103__IVB__T_1p40K[-0.000002250], D4.2__Fig 2__SQUID_data__IVB_BS30_A1S1_I_coil_+-0.3mA_fine_284mK_191103__IVB__T_1p40K[-0.000003000]
- Spacing outliers: 1
- Spacing outlier examples: D4.2__Fig 2__SQUID_data__IVB_BS30_A1S1_I_coil_+-0.3mA_fine_284mK_191103__IVB__T_1p40K[0.000300000]

## Metrics (medians)

- odd_ratio_L1: n=1602, median=2.3272
- odd_energy_L2: n=1602, median=0.682093
- odd_ratio_pre: n=1602, median=0.132006
- odd_ratio_post: n=1602, median=5.79036
- phi0_proxy: n=1602, median=1.97506
- I_knee_norm: n=1602, median=0.611529

## Condition tests (zero vs nonzero)

- No condition comparisons available.

## Null models (algorithmic significance)

- sign_randomization odd_ratio_post: observed=5.79, null_mean=5.79, null_std=3.571e-15, z=0.995
- sign_randomization parity_lock_fraction: observed=0, null_mean=0, null_std=0, z=NA
- sign_randomization phi0_proxy_abs: observed=1.975, null_mean=1.574, null_std=0.03452, z=11.62
- phase_scramble odd_ratio_post: observed=5.79, null_mean=0.5077, null_std=0.01481, z=356.7
- phase_scramble parity_lock_fraction: observed=0, null_mean=0.1776, null_std=0.00991, z=-17.93
- phase_scramble phi0_proxy_abs: observed=1.975, null_mean=2.579, null_std=0.01286, z=-46.94
- pair_breaking odd_ratio_post: observed=5.79, null_mean=5.694, null_std=0.04009, z=2.393
- pair_breaking parity_lock_fraction: observed=0, null_mean=0.0728, null_std=0.006411, z=-11.36
- pair_breaking phi0_proxy_abs: observed=1.975, null_mean=2.008, null_std=0.003203, z=-10.18

## What was tested

- Even/odd decomposition of V(I) on symmetric current grids.
- Log-log knee detection on |V| vs |I| with pre/post parity metrics.
- Phi0 proxy from odd-integral and parity-lock tests.
- Nulls: sign randomization, phase scrambling, pair-breaking.

## What survived nulls

- odd_ratio_post deviates from phase-scramble null (z=357).
- odd_ratio_post deviates from pair-breaking null (z=2.39).

## What failed or is bounded

- parity_lock_fraction is at or below null expectation (no post-knee locking).
- phi0_proxy_abs is below phase-scramble null (z=-46.9); treat as upper bound.

## Interpretation

- Effects are reported as algorithmic significance (z-scores vs nulls), not as physical discovery claims.
- Metrics that fail parity-valid checks are excluded from parity and knee analyses.
