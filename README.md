# JJ Spiral Analysis Pipeline

Reproducible CLI pipeline for parsing heterogeneous `.dat` files, generating QC plots, and extracting JJ/JJA metrics.

## Repo layout

- `data/raw/` original `.dat` files (read-only)
- `results/parsed/` normalized parquet + metadata
- `results/qc_plots/` quicklook figures
- `results/metrics/` extracted metrics tables
- `results/reports/` logs and summary outputs

## How to run ingestion / QC / metrics

If your `.dat` files live under `Data_for_analysis/raw data/raw data`, copy or symlink them into `data/raw/`, or pass that folder via `--in`.

1) Ingest (parse + normalize + metadata)

```bash
python scripts/jj_ingest.py --in data/raw --out results/parsed --glob "**/*.dat"
```

2) QC plots (quicklooks)

```bash
python scripts/jj_qc.py --in data/raw --out results/qc_plots --max-files 20
```

3) Metrics extraction

```bash
python scripts/jj_extract_metrics.py --in data/raw --out results/metrics
```

Optional pairing via CSV (`pair_id, file_left, file_right`):

```bash
python scripts/jj_extract_metrics.py --pairing-csv pairings.csv
```

4) Summary report

```bash
python scripts/jj_report.py --metrics results/metrics/metrics_long.csv --out results/reports/summary.md
```

5) Validation plots (sanity checks)

```bash
python scripts/jj_validation.py --metrics results/metrics/metrics_long.csv --out results/validation_plots
```

6) Phase 3 figures and significance

```bash
python scripts/jj_figures.py --metrics results/metrics/metrics_long.csv --out results/figures
python scripts/jj_significance.py --metrics results/metrics/metrics_long.csv --out results/reports
```

## Dataset capabilities: voltage-only vs IV vs CPR

- Voltage-only: computes V-based knee, hysteresis, and nonreciprocity proxies.
- IV: computes I-V summaries and odd/even asymmetry proxies where bias sweeps exist.
- CPR: fits CPR models only if both I and phase are present; no phi0 claims otherwise.

## How to run on voltage-only dataset

```bash
python scripts/jj_ingest.py --in data/raw
python scripts/jj_qc.py --in data/raw --max-files 20
python scripts/jj_extract_metrics.py --in data/raw
python scripts/jj_report.py
```

## How to add a new dataset

1) Drop new files into `data/raw_new/` (or any folder).
2) Run ingest, QC, and metrics with `--in data/raw_new`.
3) The pipeline auto-detects available signals and enables the relevant metric modules.

## Notes

- Configure defaults in `config.yaml`.
- The parser infers delimiters and captures metadata from header lines.
- Phase units are assumed to be radians unless the range looks like degrees (logged).
