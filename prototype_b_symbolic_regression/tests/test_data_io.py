"""
test_data_io.py — (x, y) importer, DatasetTarget and the fit.py CLI.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from data_io import (
    DataImportError, DatasetTarget, MIN_ROWS, as_target, load_dataset,
)
import fit

TEMPLATE = Path(__file__).resolve().parents[2] / "data" / "templates" / "regression_template.csv"


def _write_csv(path, header, rows):
    path.write_text(header + "\n" + "\n".join(rows) + "\n")
    return path


class TestLoadDataset:

    def test_template_loads(self):
        ds = load_dataset(TEMPLATE)
        assert len(ds.x) == len(ds.y) >= MIN_ROWS
        assert np.all(np.isfinite(ds.x)) and np.all(np.isfinite(ds.y))

    def test_custom_columns_case_insensitive_extra_ignored(self, tmp_path):
        rows = [f"{i},{i*2.0},foo" for i in range(1, 13)]
        p = _write_csv(tmp_path / "d.csv", "Dose,Response,comment", rows)
        ds = load_dataset(p, x_col="dose", y_col="response")
        assert ds.x[0] == 1.0 and ds.y[-1] == 24.0

    def test_missing_column_message_lists_found(self, tmp_path):
        p = _write_csv(tmp_path / "d.csv", "a,b", [f"{i},{i}" for i in range(12)])
        with pytest.raises(DataImportError, match=r"missing column\(s\) 'x', 'y'; found: a, b"):
            load_dataset(p)

    def test_non_numeric_cells_name_the_row(self, tmp_path):
        rows = [f"{i},{i}" for i in range(1, 12)] + ["7,abc", "1,5,"]
        p = _write_csv(tmp_path / "d.csv", "x,y", rows)
        with pytest.raises(DataImportError) as ei:
            load_dataset(p)
        assert any("row 13: y='abc' is not a number" in e for e in ei.value.errors)

    def test_blank_rows_dropped_with_warning(self, tmp_path):
        rows = [f"{i},{i}" for i in range(1, 13)] + ["3,", ","]
        ds = load_dataset(_write_csv(tmp_path / "d.csv", "x,y", rows))
        assert len(ds.x) == 12
        assert any("dropped 1 row" in w for w in ds.warnings)

    def test_too_few_rows_and_constant_x(self, tmp_path):
        with pytest.raises(DataImportError, match="at least"):
            load_dataset(_write_csv(tmp_path / "a.csv", "x,y", ["1,1", "2,2"]))
        with pytest.raises(DataImportError, match="distinct"):
            load_dataset(_write_csv(tmp_path / "b.csv", "x,y", ["1,1"] * 20))

    def test_infinite_rejected_and_nonpositive_warned(self, tmp_path):
        with pytest.raises(DataImportError):
            load_dataset(_write_csv(tmp_path / "a.csv", "x,y",
                                    [f"{i},{i}" for i in range(12)] + ["1,inf"]))
        ds = load_dataset(_write_csv(tmp_path / "b.csv", "x,y", [f"{i},{i}" for i in range(-2, 12)]))
        assert any("<= 0" in w for w in ds.warnings)

    def test_subsampling_is_deterministic(self, tmp_path):
        rows = [f"{i/10},{i}" for i in range(1, 501)]
        p = _write_csv(tmp_path / "d.csv", "x,y", rows)
        d1, d2 = load_dataset(p, max_points=100), load_dataset(p, max_points=100)
        assert len(d1.x) == 100 and np.array_equal(d1.x, d2.x)
        assert d1.n_rows_read == 500

    def test_json_rows_and_columns(self, tmp_path):
        j1 = tmp_path / "a.json"
        j1.write_text(json.dumps([{"x": i, "y": i * i} for i in range(1, 13)]))
        assert load_dataset(j1).y[2] == 9
        j2 = tmp_path / "b.json"
        j2.write_text(json.dumps({"x": list(range(1, 13)), "y": list(range(12))}))
        assert len(load_dataset(j2).x) == 12
        j3 = tmp_path / "c.json"
        j3.write_text(json.dumps({"x": [1, 2], "y": [1]}))
        with pytest.raises(DataImportError, match="different lengths"):
            load_dataset(j3)

    def test_xlsx_with_sheet(self, tmp_path):
        openpyxl = pytest.importorskip("openpyxl")
        wb = openpyxl.Workbook()
        ws = wb.active
        ws.title = "runs"
        ws.append(["x", "y"])
        for i in range(1, 13):
            ws.append([i * 0.5, i * 1.5])
        p = tmp_path / "d.xlsx"
        wb.save(p)
        assert load_dataset(p, sheet="runs").y[0] == 1.5
        with pytest.raises(DataImportError, match="sheet 'other' not found"):
            load_dataset(p, sheet="other")

    def test_unsupported_and_missing(self, tmp_path):
        with pytest.raises(DataImportError, match="unsupported file type"):
            load_dataset(tmp_path / "d.parquet")
        with pytest.raises(DataImportError, match="file not found"):
            load_dataset(tmp_path / "nope.csv")


class TestTargetAndCli:

    def test_dataset_target_returns_user_data(self, tmp_path):
        ds = load_dataset(TEMPLATE)
        t = as_target(ds)
        assert isinstance(t, DatasetTarget) and t.tokens is None
        x, y = t.sample(seed=123)
        assert np.array_equal(x, ds.x) and np.array_equal(y, ds.y)

    def test_fit_finds_exact_square_quickly(self, tmp_path):
        x = np.linspace(0.2, 2.0, 30)
        rows = [f"{v:.6f},{v*v:.6f}" for v in x]
        p = _write_csv(tmp_path / "sq.csv", "x,y", rows)
        out = tmp_path / "fit.json"
        rc = fit.main(["--file", str(p), "--iterations", "6", "--batch", "40",
                       "--out", str(out), "--quiet"])
        assert rc == 0
        res = json.loads(out.read_text())
        assert res["n_points"] == 30
        front = res["pareto_front"]
        assert front and front == sorted(front, key=lambda r: (r["complexity"], r["mse"]))
        assert min(r["mse"] for r in front) < 1e-3

    def test_fit_csv_output(self, tmp_path):
        out = tmp_path / "fit.csv"
        assert fit.main(["--file", str(TEMPLATE), "--iterations", "2", "--batch", "20",
                         "--out", str(out), "--quiet"]) == 0
        assert out.read_text().startswith("complexity,mse,r2,expression")

    def test_fit_bad_file_exit_code(self, tmp_path, capsys):
        p = _write_csv(tmp_path / "d.csv", "a,b", ["1,2"])
        assert fit.main(["--file", str(p), "--quiet"]) == 2
        assert "missing column" in capsys.readouterr().err
