"""
test_theorem_io.py — Parser, exact truth check, file import and prove.py CLI.
"""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from expressions import Add, Mul, Neg, C, a, b, c
from kernel import KERNEL
from theorem_io import (
    TheoremImportError, TheoremParseError, identity_counterexample,
    load_theorems, make_theorem, parse_expr, parse_statement, total_degree,
)
import prove

TEMPLATE = Path(__file__).resolve().parents[2] / "data" / "templates" / "theorems_template.csv"


class TestParser:

    def test_precedence_and_subtraction(self):
        assert parse_expr("a + b*c") == Add(a, Mul(b, c))
        assert parse_expr("a - b") == Add(a, Neg(b))
        assert parse_expr("-a*b") == Mul(Neg(a), b)

    def test_parentheses_and_constants(self):
        assert parse_expr("2*(a+0)") == Mul(C(2), Add(a, C(0)))

    def test_power_expands_to_products(self):
        assert parse_expr("a^3") == Mul(Mul(a, a), a)
        assert parse_expr("a**2") == Mul(a, a)
        assert parse_expr("b^0") == C(1)

    def test_equation_becomes_difference(self):
        assert parse_statement("a*b = b*a") == Add(Mul(a, b), Neg(Mul(b, a)))

    @pytest.mark.parametrize("bad, fragment", [
        ("x + 1", "unknown variable 'x'"),
        ("a + ", "ends unexpectedly"),
        ("(a + b", "missing ')'"),
        ("a $ b", "unexpected character '$'"),
        ("a = b = c", "at most one '='"),
        ("a^b", "non-negative integer"),
        ("a^99", "above the limit"),
        ("", "empty"),
        ("= a", "both sides"),
    ])
    def test_errors_are_specific(self, bad, fragment):
        with pytest.raises(TheoremParseError, match=None) as ei:
            parse_statement(bad)
        assert fragment in str(ei.value)


class TestTruthCheck:

    def test_true_identities_have_no_counterexample(self):
        for s in ["(a+b)^2 = a^2 + 2*a*b + b^2",
                  "(a+b)*(a-b) = a^2 - b^2",
                  "a*(b*c) = (a*b)*c"]:
            assert identity_counterexample(parse_statement(s)) is None

    def test_false_statement_gives_witness(self):
        e = parse_statement("(a+b)^2 = a^2 + b^2")
        env = identity_counterexample(e)
        assert env is not None and e.eval(env) != 0

    def test_degree(self):
        assert total_degree(parse_expr("a*b*c + a^2")) == 3

    def test_make_theorem_rejects_false(self):
        with pytest.raises(TheoremParseError, match="not a ring identity"):
            make_theorem("a + b = a")


class TestImport:

    def test_template_loads_and_every_row_is_provable(self):
        report = load_theorems(TEMPLATE)
        assert report.ok and len(report.theorems) >= 3
        results = prove.run(report.theorems, None, max_nodes=2000, beam_width=8, verbose=False)
        assert all(r["status"] == "proved" for r in results)

    def test_csv_row_errors_are_reported_not_fatal(self, tmp_path):
        p = tmp_path / "t.csv"
        p.write_text("id,statement\nok,a*b = b*a\nbad,a + q\nfalse,a = b\nok,a = a\n,\n")
        report = load_theorems(p)
        assert [t.id for t in report.theorems] == ["ok"]
        joined = "\n".join(report.errors)
        assert "row 2 (bad)" in joined and "unknown variable 'q'" in joined
        assert "not a ring identity" in joined
        assert "duplicate id 'ok'" in joined

    def test_strict_raises_with_all_errors(self, tmp_path):
        p = tmp_path / "t.csv"
        p.write_text("statement\na + q\na = b\n")
        with pytest.raises(TheoremImportError) as ei:
            load_theorems(p, strict=True)
        assert len(ei.value.errors) == 2

    def test_missing_statement_column(self, tmp_path):
        p = tmp_path / "t.csv"
        p.write_text("id,formula\nT1,a = a\n")
        with pytest.raises(TheoremImportError, match="missing required column 'statement'"):
            load_theorems(p)

    def test_json_and_txt(self, tmp_path):
        j = tmp_path / "t.json"
        j.write_text(json.dumps({"theorems": ["a*1 = a", {"id": "N", "statement": "-(-b) = b"}]}))
        assert [t.id for t in load_theorems(j).theorems] == ["T1", "N"]
        t = tmp_path / "t.txt"
        t.write_text("# my theorems\n\na + 0 = a\n")
        assert len(load_theorems(t).theorems) == 1

    def test_xlsx(self, tmp_path):
        openpyxl = pytest.importorskip("openpyxl")
        wb = openpyxl.Workbook()
        ws = wb.active
        ws.append(["id", "statement", "notes"])
        ws.append(["X1", "a*(b+c) = a*b + a*c", "from excel"])
        p = tmp_path / "t.xlsx"
        wb.save(p)
        report = load_theorems(p)
        assert report.theorems[0].id == "X1" and report.theorems[0].notes == "from excel"

    def test_bad_file_type_and_missing_file(self, tmp_path):
        with pytest.raises(TheoremImportError, match="unsupported file type"):
            load_theorems(tmp_path / "t.docx")
        with pytest.raises(TheoremImportError, match="file not found"):
            load_theorems(tmp_path / "nope.csv")


class TestProveCli:

    def test_cli_writes_certified_proofs(self, tmp_path, capsys):
        out = tmp_path / "proofs.json"
        rc = prove.main(["--statement", "a*(b+c) = a*b + a*c", "--out", str(out), "--quiet"])
        assert rc == 0
        data = json.loads(out.read_text())
        assert data[0]["status"] == "proved"
        # replay the reported rule sequence through the kernel: it must reach 0
        from kernel import RULE_NAMES
        expr = make_theorem("a*(b+c) = a*b + a*c").expr
        for s in data[0]["steps"]:
            expr = KERNEL.constant_fold(KERNEL.apply(RULE_NAMES.index(s["rule"]), s["path"], expr))
        assert KERNEL.is_zero(expr)

    def test_cli_csv_output_and_uniform(self, tmp_path):
        out = tmp_path / "proofs.csv"
        rc = prove.main(["--file", str(TEMPLATE), "--uniform", "--out", str(out), "--quiet"])
        assert rc == 0
        assert out.read_text().splitlines()[0].startswith("id,statement,status")

    def test_cli_rejects_when_nothing_valid(self, capsys):
        assert prove.main(["--statement", "a = b", "--quiet"]) == 2
        assert "not a ring identity" in capsys.readouterr().err

    def test_cli_missing_policy_file(self, tmp_path):
        assert prove.main(["--statement", "a = a", "--policy", str(tmp_path / "w.npz"), "--quiet"]) == 2

    def test_cli_demo(self):
        assert prove.main(["--demo", "2", "--quiet"]) == 0
