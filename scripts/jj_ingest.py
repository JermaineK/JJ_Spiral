"""Ingest JJ .dat files into normalized parquet + metadata."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from jjparse.config import load_config, seed_everything, setup_logging
from jjparse.io import make_file_id, read_dat
from jjparse.preprocess import apply_unit_conversions, infer_phase_radians
from jjparse.schema import coerce_types, standardize_columns

LOGGER = logging.getLogger("jjparse")


def _iter_files(in_dir: Path, glob_pattern: str) -> list[Path]:
    files = sorted(in_dir.glob(glob_pattern))
    return [path for path in files if path.is_file()]


def _row_from_meta(meta: dict, df: pd.DataFrame | None) -> dict:
    row = {
        "file_id": meta.get("file_id"),
        "source_file": meta.get("source_file"),
        "sha256": meta.get("sha256"),
        "parsed_at": meta.get("parsed_at"),
        "parser_version": meta.get("parser_version"),
        "n_rows": int(meta.get("n_rows", len(df) if df is not None else 0)),
        "n_cols": int(meta.get("n_cols", df.shape[1] if df is not None else 0)),
    }
    for key, value in meta.items():
        if key in row:
            continue
        if isinstance(value, list):
            row[key] = ";".join(str(v) for v in value)
        else:
            row[key] = value
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest JJ .dat files into parquet + metadata.")
    parser.add_argument("--in", dest="in_dir", default="data/raw", help="Input data folder")
    parser.add_argument("--out", dest="out_dir", default="results/parsed", help="Output folder")
    parser.add_argument("--glob", default="**/*.dat", help="Glob pattern for data files")
    parser.add_argument("--max-files", type=int, default=None, help="Limit number of files")
    parser.add_argument("--config", default=None, help="Path to config.yaml")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing outputs")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    args = parser.parse_args()

    config = load_config(args.config)
    seed_everything(args.seed)
    setup_logging("results/reports")

    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir)
    meta_dir = out_dir / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    files = _iter_files(in_dir, args.glob)
    if args.max_files:
        files = files[: args.max_files]

    index_rows = []

    for path in tqdm(files, desc="Ingest"):
        file_id = make_file_id(path, in_dir)
        out_path = out_dir / f"{file_id}.parquet"
        meta_path = meta_dir / f"{file_id}.json"
        if out_path.exists() and not args.overwrite:
            if meta_path.exists():
                with open(meta_path, "r", encoding="utf-8") as handle:
                    cached = json.load(handle)
                index_rows.append(_row_from_meta(cached, None))
            LOGGER.info("Skipping existing %s", out_path)
            continue

        df_raw, meta = read_dat(path)
        if df_raw.empty:
            LOGGER.warning("No data parsed for %s", path)
            continue

        meta["file_id"] = file_id

        df_std = standardize_columns(df_raw, meta)
        df_std = coerce_types(df_std)
        df_std = apply_unit_conversions(df_std, meta, config)
        df_std = infer_phase_radians(df_std, meta, config)

        df_std.to_parquet(out_path, index=False)

        meta["n_rows"] = int(len(df_std))
        meta["n_cols"] = int(df_std.shape[1])
        with open(meta_path, "w", encoding="utf-8") as handle:
            json.dump(meta, handle, indent=2)

        index_rows.append(_row_from_meta(meta, df_std))

    if not index_rows:
        LOGGER.info("No files ingested.")
        return

    index_path = out_dir / "index.csv"
    new_index = pd.DataFrame(index_rows)
    if index_path.exists() and not args.overwrite:
        existing = pd.read_csv(index_path)
        combined = pd.concat([existing, new_index], ignore_index=True)
        combined = combined.drop_duplicates(subset=["file_id"], keep="last")
    else:
        combined = new_index

    combined.to_csv(index_path, index=False)
    LOGGER.info("Wrote index to %s", index_path)


if __name__ == "__main__":
    main()
