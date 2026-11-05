"""
data_io.py — Bring your own (x, y) data to the symbolic regression search.

Loads one input column and one output column from CSV, TSV, Excel (.xlsx)
or JSON, validates it, and wraps it in a `DatasetTarget` that `search.run_search`
accepts in place of the built-in synthetic targets.

File schema
-----------
  x  (required)  input values — numbers; column name configurable (--x-col)
  y  (required)  observed outputs — numbers; column name configurable (--y-col)
  any other columns are ignored (e.g. a `note` column)

JSON may be a list of row objects  [{"x": 0.5, "y": 1.2}, ...]
or column arrays                   {"x": [...], "y": [...]}.

Validation
----------
  * both columns must exist (names matched case-insensitively);
  * every non-blank cell must parse as a finite number — errors name the row;
  * rows where x or y is blank are reported and dropped;
  * at least MIN_ROWS rows and MIN_DISTINCT_X distinct x values are needed.
Non-positive x values are allowed but produce a warning, because the
grammar's sqrt/inv/div primitives are undefined there and candidates that
use them get NaN on those points.
"""

from __future__ import annotations

import csv
import io
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

MIN_ROWS = 10
MIN_DISTINCT_X = 5
MAX_POINTS = 2000          # larger files are subsampled (deterministically) for speed

SUPPORTED_SUFFIXES = (".csv", ".tsv", ".xlsx", ".json")


class DataImportError(ValueError):
    """The file cannot be used; `errors` lists row-level problems."""

    def __init__(self, message: str, errors: Optional[List[str]] = None):
        super().__init__(message)
        self.errors = list(errors or [])


@dataclass
class Dataset:
    x: np.ndarray
    y: np.ndarray
    name: str
    source: str = ""
    warnings: List[str] = field(default_factory=list)
    n_rows_read: int = 0


@dataclass
class DatasetTarget:
    """
    Duck-typed stand-in for targets.Target so user data runs through the
    same search loop.  There is no ground truth, so `tokens` is None and
    recovery is not scored — the Pareto front is the answer.
    """
    name: str
    x_vals: np.ndarray
    y_vals: np.ndarray
    description: str = ""
    tokens: Optional[List[str]] = None
    ood: bool = False

    def sample(self, seed: int = 0) -> Tuple[np.ndarray, np.ndarray]:
        return self.x_vals, self.y_vals


# ---------------------------------------------------------------------------
# Readers → list of row dicts (all values as strings)
# ---------------------------------------------------------------------------

def _rows_csv(text: str, delimiter: str) -> Tuple[List[str], List[Dict[str, str]]]:
    reader = csv.DictReader(io.StringIO(text), delimiter=delimiter)
    if not reader.fieldnames:
        raise DataImportError("file is empty — expected a header row such as 'x,y'")
    return list(reader.fieldnames), [dict(r) for r in reader]


def _rows_xlsx(path: Path, sheet: Optional[str]) -> Tuple[List[str], List[Dict[str, str]]]:
    try:
        import openpyxl  # optional, only for .xlsx
    except ImportError:
        raise DataImportError(
            "reading .xlsx needs openpyxl (`python3 -m pip install openpyxl`), "
            "or save the sheet as CSV"
        ) from None
    wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
    if sheet:
        if sheet not in wb.sheetnames:
            raise DataImportError(f"sheet {sheet!r} not found; sheets: {', '.join(wb.sheetnames)}")
        ws = wb[sheet]
    else:
        ws = wb.worksheets[0]
    rows = list(ws.iter_rows(values_only=True))
    wb.close()
    if not rows:
        raise DataImportError("sheet is empty — expected a header row such as 'x,y'")
    header = [str(h).strip() if h is not None else "" for h in rows[0]]
    out = [{h: ("" if v is None else str(v)) for h, v in zip(header, r) if h} for r in rows[1:]]
    return [h for h in header if h], out


def _rows_json(text: str) -> Tuple[List[str], List[Dict[str, str]]]:
    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        raise DataImportError(f"invalid JSON: {e}") from None
    if isinstance(data, dict):
        cols = {k: v for k, v in data.items() if isinstance(v, list)}
        if not cols:
            raise DataImportError("JSON object must map column names to lists, e.g. {\"x\": [...], \"y\": [...]}")
        lengths = {len(v) for v in cols.values()}
        if len(lengths) > 1:
            raise DataImportError(
                "JSON columns have different lengths: "
                + ", ".join(f"{k}={len(v)}" for k, v in cols.items())
            )
        n = lengths.pop()
        rows = [{k: ("" if v[i] is None else str(v[i])) for k, v in cols.items()} for i in range(n)]
        return list(cols), rows
    if isinstance(data, list):
        if not all(isinstance(r, dict) for r in data):
            raise DataImportError("JSON list must contain row objects like {\"x\": 1, \"y\": 2}")
        header: List[str] = []
        for r in data:
            for k in r:
                if k not in header:
                    header.append(k)
        rows = [{k: ("" if v is None else str(v)) for k, v in r.items()} for r in data]
        return header, rows
    raise DataImportError("JSON must be a list of row objects or an object of column lists")


def _find_col(header: List[str], wanted: str) -> Optional[str]:
    for h in header:
        if h.strip().lower() == wanted.strip().lower():
            return h
    return None


def _to_float(s: str) -> Optional[float]:
    s = (s or "").strip()
    if not s:
        return None
    return float(s)     # no locale guessing: "1,5" is reported, not silently misread


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def rows_to_dataset(
    header: List[str],
    rows: List[Dict[str, str]],
    x_col: str = "x",
    y_col: str = "y",
    name: str = "dataset",
    source: str = "",
    max_points: int = MAX_POINTS,
) -> Dataset:
    xh, yh = _find_col(header, x_col), _find_col(header, y_col)
    missing = [c for c, h in ((x_col, xh), (y_col, yh)) if h is None]
    if missing:
        raise DataImportError(
            f"missing column(s) {', '.join(repr(m) for m in missing)}; "
            f"found: {', '.join(header) or 'none'} (use --x-col / --y-col to pick others)"
        )

    errors: List[str] = []
    warnings: List[str] = []
    xs: List[float] = []
    ys: List[float] = []
    n_blank = 0
    for i, r in enumerate(rows, start=2):           # row 1 is the header
        rx, ry = r.get(xh, ""), r.get(yh, "")
        if not (rx or "").strip() and not (ry or "").strip():
            continue                                   # blank line
        try:
            vx = _to_float(rx)
        except ValueError:
            errors.append(f"row {i}: {x_col}={rx!r} is not a number")
            continue
        try:
            vy = _to_float(ry)
        except ValueError:
            errors.append(f"row {i}: {y_col}={ry!r} is not a number")
            continue
        if vx is None or vy is None:
            n_blank += 1
            continue
        if not (math.isfinite(vx) and math.isfinite(vy)):
            errors.append(f"row {i}: values must be finite (got {x_col}={rx}, {y_col}={ry})")
            continue
        xs.append(vx)
        ys.append(vy)

    if errors:
        raise DataImportError(f"{len(errors)} invalid row(s) in {source or name}", errors)
    if n_blank:
        warnings.append(f"dropped {n_blank} row(s) with a blank {x_col} or {y_col}")
    if len(xs) < MIN_ROWS:
        raise DataImportError(f"need at least {MIN_ROWS} complete rows, found {len(xs)}")
    x = np.asarray(xs, dtype=float)
    y = np.asarray(ys, dtype=float)
    if len(np.unique(x)) < MIN_DISTINCT_X:
        raise DataImportError(
            f"need at least {MIN_DISTINCT_X} distinct {x_col} values, found {len(np.unique(x))}"
        )
    if np.any(x <= 0):
        warnings.append(
            f"{int(np.sum(x <= 0))} row(s) have {x_col} <= 0: sqrt/inv/div candidates "
            "are undefined there and are scored on the remaining points"
        )
    n_read = len(x)
    if max_points and len(x) > max_points:
        idx = np.sort(np.random.default_rng(0).choice(len(x), size=max_points, replace=False))
        x, y = x[idx], y[idx]
        warnings.append(f"subsampled {n_read} rows to {max_points} for search speed")
    return Dataset(x=x, y=y, name=name, source=source, warnings=warnings, n_rows_read=n_read)


def load_dataset(
    path: Union[str, Path],
    x_col: str = "x",
    y_col: str = "y",
    sheet: Optional[str] = None,
    max_points: int = MAX_POINTS,
) -> Dataset:
    """Read and validate an (x, y) dataset from .csv/.tsv/.xlsx/.json."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        raise DataImportError(
            f"unsupported file type {suffix!r}; use one of {', '.join(SUPPORTED_SUFFIXES)}"
        )
    if not path.exists():
        raise DataImportError(f"file not found: {path}")
    if suffix == ".xlsx":
        header, rows = _rows_xlsx(path, sheet)
    else:
        text = path.read_text(encoding="utf-8-sig")
        if suffix == ".json":
            header, rows = _rows_json(text)
        else:
            header, rows = _rows_csv(text, "\t" if suffix == ".tsv" else ",")
    return rows_to_dataset(header, rows, x_col, y_col, name=path.stem,
                           source=str(path), max_points=max_points)


def as_target(ds: Dataset) -> DatasetTarget:
    return DatasetTarget(
        name=ds.name,
        x_vals=ds.x,
        y_vals=ds.y,
        description=f"user data {ds.source or ds.name} ({len(ds.x)} points)",
    )
