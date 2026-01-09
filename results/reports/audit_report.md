# Audit report

- Pipeline version: 5.5

## What looked odd

- High eta_V_L1 median and potential pairing inflation.
- Knee locations clustering late in the sweep.
- Knee score rank saturation near cap.
- Bidirectional mismatch counts.
- Incoherence zeros or missing joins.

## Checks run

- Overlap threshold: min_overlap=0.8
- Denominator threshold: den_min=1e-12
- Alternate eta metrics: eta_V_L2, corr_pm.
- Knee norm/value tracking with edge flags.
- Knee score cap and capped flag.
- Randomized pairing control (cross-file pairing).

## Key outcomes

- Bidirectional segments: 4057
- eta_valid segments: 4045
- eta invalid reasons: insufficient_points=12
- knee_edge_flag (valid knees): 0.04%
- knee_score_capped: 1.00%
- knee_score_cap (p99): 3.77433e+09
- knee_x_norm (valid): median=0.48, p05=0.21, p95=0.72
- knee_x_norm (top knees): median=0.67, p05=0.268, p95=0.715
- f_overlap: median=1, p05=1, p95=1
- H_incoh missing: 0.04%
- H_incoh == 0 (raw==0): 0.00%
- Randomized pairing control: n=4045, eta_median=0.807, eta_p95=0.997, corr_median=-0.894

## High-eta cross-checks

- No high-eta + high-corr conflicts detected.

## What changed

- eta_valid gating now enforces overlap and denominator thresholds.
- Knee position stored as norm/value with edge flags; cap rate exposed.
- Added randomized pairing control and mismatch CSV.

## Remaining limitations

- Overlap threshold may exclude narrow sweeps; review low_overlap cases.
- Randomized pairing control uses same x_name but not device metadata.

## Audit tables

- Bidirectional mismatch list: results/reports/bidirectional_mismatch.csv

