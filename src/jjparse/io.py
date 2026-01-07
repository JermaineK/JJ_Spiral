"""File IO and parsing for JJ .dat files."""

from __future__ import annotations

import hashlib
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from jjparse import __version__

LOGGER = logging.getLogger("jjparse")

NUM_RE = re.compile(r"[-+]?((\d+(\.\d*)?)|(\.\d+))([eE][-+]?\d+)?")


def _normalize_key(key: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9]+", "_", key.strip().lower())
    return cleaned.strip("_")


def _is_number(token: str) -> bool:
    try:
        float(token)
        return True
    except ValueError:
        return False


def _split_line(line: str, delimiter: str) -> list[str]:
    if delimiter == " ":
        return [tok for tok in re.split(r"\s+", line.strip()) if tok]
    return [tok for tok in line.split(delimiter) if tok != ""]


def _looks_like_data_candidate(line: str) -> bool:
    return len(NUM_RE.findall(line)) >= 2


def detect_delimiter(lines: Iterable[str], max_lines: int = 200) -> tuple[str, int | None]:
    candidates = [",", "\t", ";", " "]
    counts = {c: [] for c in candidates}

    checked = 0
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        if not _looks_like_data_candidate(stripped):
            continue
        for delim in candidates:
            tokens = _split_line(stripped, delim)
            if len(tokens) < 2:
                continue
            if all(_is_number(tok) for tok in tokens):
                counts[delim].append(len(tokens))
        checked += 1
        if checked >= max_lines:
            break

    best_delim = " "
    best_count = 0
    best_mode = None
    for delim, col_counts in counts.items():
        if not col_counts:
            continue
        mode = max(set(col_counts), key=col_counts.count)
        if len(col_counts) > best_count:
            best_delim = delim
            best_count = len(col_counts)
            best_mode = mode
        elif len(col_counts) == best_count and best_mode is not None:
            if mode > best_mode:
                best_delim = delim
                best_mode = mode

    if best_mode is None:
        return best_delim, None
    return best_delim, best_mode


def _parse_kv_pairs(line: str) -> dict[str, str]:
    pairs: dict[str, str] = {}
    stripped = line.strip()
    if not stripped:
        return pairs

    if stripped.count("=") == 1 and stripped.count(":") == 0:
        key, value = stripped.split("=", 1)
        key = key.strip()
        value = value.strip()
        if key and value:
            pairs[_normalize_key(key)] = value
        return pairs

    if stripped.count(":") == 1 and stripped.count("=") == 0:
        key, value = stripped.split(":", 1)
        key = key.strip()
        value = value.strip()
        if key and value:
            pairs[_normalize_key(key)] = value
        return pairs

    for match in re.finditer(r"([A-Za-z0-9()_\-/]+)\s*:\s*([^\s,]+)", stripped):
        pairs[_normalize_key(match.group(1))] = match.group(2)

    for match in re.finditer(r"([A-Za-z0-9()_\-/]+)\s*=\s*([^\s,]+)", stripped):
        pairs[_normalize_key(match.group(1))] = match.group(2)

    return pairs


def _parse_step_line(line: str) -> dict[str, str]:
    match = re.match(r"\s*step\s*(\d+)\s+([^:]+):\s*(.*)", line, re.IGNORECASE)
    if not match:
        return {}
    step_idx = match.group(1)
    name = match.group(2).strip()
    rest = match.group(3).strip()
    return {
        f"step{step_idx}_name": name,
        f"step{step_idx}_range": rest,
    }


def _parse_meas_line(line: str) -> list[str]:
    match = re.match(r"\s*meas\s*:(.*)", line, re.IGNORECASE)
    if not match:
        return []
    payload = match.group(1).strip()
    tokens = [tok for tok in re.split(r"[\s,]+", payload) if tok]
    return tokens


def _looks_like_header_line(line: str) -> bool:
    if ":" in line or "=" in line:
        return False
    tokens = [tok for tok in re.split(r"[\s,\t;]+", line.strip()) if tok]
    if len(tokens) < 2:
        return False
    return all(not _is_number(tok) for tok in tokens)


def _infer_column_names(lines: Iterable[str], ncols: int | None) -> tuple[list[str] | None, list[str], list[str]]:
    header_names: list[str] | None = None
    step_names: list[str] = []
    meas_names: list[str] = []

    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        if stripped[0] in "#%;":
            continue

        step_meta = _parse_step_line(stripped)
        if step_meta:
            for key, value in step_meta.items():
                if key.endswith("_name"):
                    step_names.append(value)
                    break
            continue

        meas = _parse_meas_line(stripped)
        if meas:
            meas_names = meas
            continue

        if _looks_like_header_line(stripped):
            tokens = [tok for tok in re.split(r"[\s,\t;]+", stripped) if tok]
            if ncols is None or len(tokens) == ncols:
                header_names = tokens
            continue

    return header_names, step_names, meas_names


def _parse_data_lines(
    lines: Iterable[str],
    delimiter: str,
    ncols: int | None,
    colnames: list[str] | None,
) -> pd.DataFrame:
    rows: list[list[float]] = []
    block_ids: list[int] = []
    current_block = 0
    in_data = False
    pending_block = False

    for line in lines:
        stripped = line.strip()
        if not stripped:
            if in_data:
                pending_block = True
            continue

        tokens = _split_line(stripped, delimiter)
        is_data = len(tokens) >= 2 and all(_is_number(tok) for tok in tokens)

        if not is_data:
            if in_data:
                pending_block = True
            continue

        if pending_block:
            current_block += 1
            pending_block = False

        in_data = True
        values: list[float] = []
        for tok in tokens:
            try:
                values.append(float(tok))
            except ValueError:
                values.append(np.nan)

        if ncols is not None and len(values) != ncols:
            if len(values) < ncols:
                values += [np.nan] * (ncols - len(values))
            else:
                values = values[:ncols]
        rows.append(values)
        block_ids.append(current_block)

    if not rows:
        return pd.DataFrame()

    if ncols is None:
        ncols = len(rows[0])

    if not colnames or len(colnames) != ncols:
        colnames = [f"col{i}" for i in range(ncols)]

    df = pd.DataFrame(rows, columns=colnames)
    if current_block > 0:
        df["block_id"] = block_ids
    return df


def _hash_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(8192), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def make_file_id(path: str | Path, root: str | Path | None = None) -> str:
    path = Path(path)
    if root:
        try:
            rel = path.relative_to(Path(root))
            return re.sub(r"[\\/]+", "__", rel.with_suffix("").as_posix())
        except ValueError:
            pass
    return path.stem


def read_dat(path: str | Path) -> tuple[pd.DataFrame, dict]:
    path = Path(path)
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except OSError as exc:
        LOGGER.warning("Failed to read %s: %s", path, exc)
        return pd.DataFrame(), {}

    lines = text.splitlines()
    delimiter, ncols = detect_delimiter(lines)
    header_names, step_names, meas_names = _infer_column_names(lines, ncols)

    meta: dict[str, str] = {}
    header_lines: list[str] = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        if stripped[0] in "#%;":
            header_lines.append(stripped)
            continue

        step_meta = _parse_step_line(stripped)
        if step_meta:
            meta.update({k: v for k, v in step_meta.items()})
            continue

        meas = _parse_meas_line(stripped)
        if meas:
            meta["meas"] = ",".join(meas)
            continue

        pairs = _parse_kv_pairs(stripped)
        if pairs:
            meta.update(pairs)
            continue

        if _looks_like_header_line(stripped):
            header_lines.append(stripped)
            continue

    if not header_names and meas_names:
        if step_names:
            header_names = [step_names[0]] + meas_names
        else:
            header_names = meas_names

    df = _parse_data_lines(lines, delimiter, ncols, header_names)
    if header_names:
        df.attrs["raw_columns"] = header_names
    else:
        df.attrs["raw_columns"] = list(df.columns)

    meta["source_file"] = str(path)
    meta["sha256"] = _hash_file(path)
    meta["parsed_at"] = datetime.utcnow().isoformat() + "Z"
    meta["parser_version"] = __version__
    if header_lines:
        meta["raw_header_lines"] = header_lines[:200]
    meta["raw_columns"] = df.attrs.get("raw_columns", list(df.columns))
    return df, meta
