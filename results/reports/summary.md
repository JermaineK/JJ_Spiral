# Summary

- Pipeline version: 0.4.5

## Dataset capabilities

- Total files: 32310
- Voltage-only files: 32298
- Bidirectional files: 4057
- Cyclic files: 0

## Axis usage

- VbiasSQUID1: 31819
- VbiasSQUID2: 479
- col0: 12

## Incoherence methods

- spectral_flatness: 32298

## Metric distributions

- knee_x: n=32310, median=0.76, p95=0.96
- knee_x_norm: n=32310, median=0.48, p95=0.72
- H_incoh_raw: n=32298, median=0.0325499, p95=0.0756447
- H_incoh: n=32298, median=0.306508, p95=0.750981
- eta_V_L1: n=4045, median=0.788913, p95=0.996902
- eta_V_L2: n=4045, median=0.97842, p95=0.99496
- corr_pm: n=4045, median=-0.990415, p95=-0.952845
- eta_norm: n=4045, median=1.15852, p95=4.69131
- K_max: n=32298, median=1.19588, p95=16.904
- knee_score_norm: n=32298, median=5.3959, p95=202.957
- knee_score_rank: n=32298, median=1.85566, p95=5.31791

## QC checks

- eta_valid: 12.52% of segments
- knee_edge_flag: 0.04% of valid knees
- knee_score_capped: 1.00% of segments
- knee_score_cap (p99): 3.77433e+09
- H_incoh missing: 0.04% of segments
- H_incoh == 0 (raw==0): 0.00% of segments

## Knee position distributions (knee_x_norm)

- all valid: n=32310, p05=0.21, median=0.48, p95=0.72
- top knee_score_rank: n=20, p05=0.25, median=0.67, p95=0.681
- random controls: n=100, p05=0.1, median=0.45, p95=0.68

## Normalization effects

- Suppressed (high incoherence):
- 220726183749_110_0 seg=0 eta=0.9968 eta_norm=1.167 H_incoh=0.854
- 220726183749_110_1 seg=0 eta=0.9968 eta_norm=1.22 H_incoh=0.817
- 220726183749_111_1 seg=0 eta=0.9964 eta_norm=1.218 H_incoh=0.818
- 220726183749_109_0 seg=0 eta=0.9957 eta_norm=1.208 H_incoh=0.824
- 220726183749_111_0 seg=0 eta=0.9948 eta_norm=1.194 H_incoh=0.833
- Promoted (low incoherence):
- 220729121637_086_0_1 seg=0 eta=0.8359 eta_norm=5.19 H_incoh=0.161
- 220729121637_093_1_0 seg=0 eta=0.8455 eta_norm=6.826 H_incoh=0.124
- 220729121637_086_0_0 seg=0 eta=0.8319 eta_norm=4.457 H_incoh=0.187
- 220729121637_163_1_0 seg=0 eta=0.8519 eta_norm=6.299 H_incoh=0.135
- 220729121637_158_2_1 seg=0 eta=0.8483 eta_norm=5.157 H_incoh=0.164

## Parity stability

- parity_stability: n=4045, median=1, p95=1
- Knee parity flips: n=169 / 3651 (fraction=0.04629)
- Parity locks after knee: n=3639 / 3651 (fraction=0.9967)

## Top odd-channel candidates (eta_norm)

- 220729121637_130_7_0 seg=0 eta_norm=8.90084 x=VbiasSQUID1
- 220729121637_129_6_0 seg=0 eta_norm=8.80216 x=VbiasSQUID1
- 220729121637_168_8_0 seg=0 eta_norm=8.54506 x=VbiasSQUID1
- 220729121637_028_8_1 seg=0 eta_norm=8.44405 x=VbiasSQUID1
- 220729121637_089_0_0 seg=0 eta_norm=8.43229 x=VbiasSQUID1
- 220729121637_128_5_1 seg=0 eta_norm=8.076 x=VbiasSQUID1
- 220729121637_159_0_0 seg=0 eta_norm=7.94881 x=VbiasSQUID1
- 220729121637_090_1_0 seg=0 eta_norm=7.92266 x=VbiasSQUID1
- 220729121637_126_3_1 seg=0 eta_norm=7.74951 x=VbiasSQUID1
- 220729121637_159_0_1 seg=0 eta_norm=7.59955 x=VbiasSQUID1
- 220729121637_062_8_1 seg=0 eta_norm=7.51869 x=VbiasSQUID1
- 220729121637_168_8_1 seg=0 eta_norm=7.50532 x=VbiasSQUID1
- 220729121637_129_6_1 seg=0 eta_norm=7.4823 x=VbiasSQUID1
- 220729121637_062_8_0 seg=0 eta_norm=7.41571 x=VbiasSQUID1
- 220729121637_199_3_1 seg=0 eta_norm=7.41012 x=VbiasSQUID1
- 220729121637_026_8_0 seg=0 eta_norm=7.37596 x=VbiasSQUID1
- 220729121637_160_1_0 seg=0 eta_norm=7.27312 x=VbiasSQUID1
- 220729121637_089_0_1 seg=0 eta_norm=7.26899 x=VbiasSQUID1
- 220729121637_168_6_1 seg=0 eta_norm=7.20807 x=VbiasSQUID1
- 220729121637_025_7_1 seg=0 eta_norm=7.15882 x=VbiasSQUID1

## Top knee evidence (knee_score_rank)

- 220804133537_3_3_184_1 seg=0 knee_score_rank=22.0515 x=VbiasSQUID1
- 220726030808_134_3 seg=0 knee_score_rank=22.0515 x=VbiasSQUID1
- 220726030808_135_3 seg=0 knee_score_rank=22.0515 x=VbiasSQUID1
- 220804133537_3_2_059_3 seg=0 knee_score_rank=22.0515 x=VbiasSQUID1
- 220804133537_3_2_059_4 seg=0 knee_score_rank=22.0515 x=VbiasSQUID1
- 220804133537_3_2_059_6 seg=0 knee_score_rank=22.0515 x=VbiasSQUID1
- 220804133537_3_2_060_0 seg=0 knee_score_rank=22.0515 x=VbiasSQUID1
- 220804133537_3_2_060_3 seg=0 knee_score_rank=22.0515 x=VbiasSQUID1
- 220804133537_3_2_061_4 seg=0 knee_score_rank=22.0515 x=VbiasSQUID1
- 220804133537_3_2_088_5 seg=0 knee_score_rank=22.0515 x=VbiasSQUID1
- 220804133537_3_2_061_6 seg=0 knee_score_rank=22.0515 x=VbiasSQUID1
- 220726030808_136_6 seg=0 knee_score_rank=22.0515 x=VbiasSQUID1
- 220726030808_136_7 seg=0 knee_score_rank=22.0515 x=VbiasSQUID1
- 220804133537_3_0_058_5 seg=0 knee_score_rank=22.0515 x=VbiasSQUID1
- 220726030808_084_1 seg=0 knee_score_rank=22.0515 x=VbiasSQUID1
- 220804133537_3_0_058_6 seg=0 knee_score_rank=22.0515 x=VbiasSQUID1
- 220804133537_3_0_059_1 seg=0 knee_score_rank=22.0515 x=VbiasSQUID1
- 220804133537_2_2_135_1 seg=0 knee_score_rank=22.0515 x=VbiasSQUID1
- 220804133537_2_2_133_0 seg=0 knee_score_rank=22.0515 x=VbiasSQUID1
- 220804133537_3_2_036_4 seg=0 knee_score_rank=22.0515 x=VbiasSQUID1

## Interpretation

- Incoherence normalization rescales eta and knee scores, promoting low-incoherence traces while keeping raw values intact.
- Parity stability and knee-parity stats quantify whether odd-channel signals persist across segments.
