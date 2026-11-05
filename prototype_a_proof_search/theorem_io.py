"""
theorem_io.py — Bring your own theorems to the proof search.

Parses human-written ring statements such as

    a*(b+c) = a*b + a*c
    (a+b)^2 - (a^2 + 2*a*b + b^2)
    -(-a) - a

into the immutable `Expr` trees used by the kernel, checks that each one is a
genuine identity of the commutative ring Z[a, b, c] (so the search is never
asked to prove something false), and loads batches of them from CSV, Excel,
JSON or plain-text files.

Statement syntax
----------------
  variables : a, b, c                (the kernel's variable set)
  integers  : 0, 1, 2, 17, ...
  operators : +  -  *  ^ (or **) with a non-negative integer exponent
  grouping  : ( ... )
  equation  : `lhs = rhs` is proved as `lhs - rhs = 0`;
              a statement with no `=` is proved as `statement = 0`.

`x - y` becomes `x + (-y)` and `x^n` becomes the product x*x*...*x, exactly
as the kernel's own constructors build them.

File schema (CSV / XLSX / JSON)
-------------------------------
  statement  (required)  the identity to prove, syntax above
  id         (optional)  unique label; defaults to T<row number>
  notes      (optional)  free text, carried through to the output

Plain-text files hold one statement per line; blank lines and lines starting
with `#` are skipped.

Truth check
-----------
`identity_counterexample()` is exact, not a numerical heuristic: a non-zero
polynomial in three variables whose total degree is D cannot vanish on the
whole grid {0..D}^3, so evaluating the statement on that grid (with Python's
exact integers) decides whether it is an identity.
"""

from __future__ import annotations

import csv
import io
import json
import re
from dataclasses import dataclass, field
from itertools import product
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple, Union

from expressions import Expr, Const, Var, Add, Mul, Neg

VARIABLES = ("a", "b", "c")
MAX_EXPONENT = 8        # x^n with n above this is rejected (tree blow-up)
MAX_SIZE = 250          # node budget per statement; larger trees stall the search
MAX_DEGREE = 24         # bounds the exact truth-check grid at (D+1)^3 points

SUPPORTED_SUFFIXES = (".csv", ".tsv", ".xlsx", ".json", ".txt")


class TheoremParseError(ValueError):
    """A statement could not be parsed; the message points at the problem."""


class TheoremImportError(ValueError):
    """A file could not be imported; `errors` lists every row-level problem."""

    def __init__(self, message: str, errors: Optional[List[str]] = None):
        super().__init__(message)
        self.errors = list(errors or [])


# ---------------------------------------------------------------------------
# Tokeniser + recursive-descent parser (no eval)
# ---------------------------------------------------------------------------

_TOKEN_RE = re.compile(r"\s*(?:(\d+)|([A-Za-z_][A-Za-z_0-9]*)|(\*\*|[-+*^()=]))")


def _tokenise(text: str) -> List[Tuple[str, str, int]]:
    tokens: List[Tuple[str, str, int]] = []
    pos = 0
    text = text.rstrip()
    while pos < len(text):
        m = _TOKEN_RE.match(text, pos)
        if not m:
            j = pos
            while j < len(text) and text[j].isspace():
                j += 1
            raise TheoremParseError(f"unexpected character {text[j]!r} at column {j + 1}")
        num, name, op = m.groups()
        col = m.start(m.lastindex) + 1
        if num is not None:
            tokens.append(("int", num, col))
        elif name is not None:
            if name not in VARIABLES:
                raise TheoremParseError(
                    f"unknown variable {name!r} at column {col} — only "
                    f"{', '.join(VARIABLES)} are supported by the kernel"
                )
            tokens.append(("var", name, col))
        else:
            tokens.append(("op", "^" if op == "**" else op, col))
        pos = m.end()
    return tokens


class _Parser:
    def __init__(self, tokens: List[Tuple[str, str, int]]):
        self.toks = tokens
        self.i = 0

    def peek(self) -> Optional[Tuple[str, str, int]]:
        return self.toks[self.i] if self.i < len(self.toks) else None

    def take(self) -> Tuple[str, str, int]:
        tok = self.peek()
        if tok is None:
            raise TheoremParseError("statement ends unexpectedly")
        self.i += 1
        return tok

    def at_op(self, *ops: str) -> bool:
        tok = self.peek()
        return tok is not None and tok[0] == "op" and tok[1] in ops

    # expr := term (('+' | '-') term)*
    def expr(self) -> Expr:
        node = self.term()
        while self.at_op("+", "-"):
            op = self.take()[1]
            rhs = self.term()
            node = Add(node, rhs) if op == "+" else Add(node, Neg(rhs))
        return node

    # term := unary ('*' unary)*
    def term(self) -> Expr:
        node = self.unary()
        while self.at_op("*"):
            self.take()
            node = Mul(node, self.unary())
        return node

    # unary := '-' unary | power
    def unary(self) -> Expr:
        if self.at_op("-"):
            self.take()
            return Neg(self.unary())
        if self.at_op("+"):
            self.take()
            return self.unary()
        return self.power()

    # power := atom ('^' INT)?
    def power(self) -> Expr:
        base = self.atom()
        if self.at_op("^"):
            _, _, col = self.take()
            tok = self.take()
            if tok[0] != "int":
                raise TheoremParseError(
                    f"exponent at column {col} must be a non-negative integer"
                )
            n = int(tok[1])
            if n > MAX_EXPONENT:
                raise TheoremParseError(
                    f"exponent {n} at column {col} is above the limit of {MAX_EXPONENT}"
                )
            if n == 0:
                return Const(1)
            node = base
            for _ in range(n - 1):
                node = Mul(node, base)
            return node
        return base

    # atom := INT | VAR | '(' expr ')'
    def atom(self) -> Expr:
        tok = self.take()
        kind, val, col = tok
        if kind == "int":
            return Const(int(val))
        if kind == "var":
            return Var(val)
        if val == "(":
            node = self.expr()
            if not self.at_op(")"):
                nxt = self.peek()
                where = f"column {nxt[2]}" if nxt else "end of statement"
                raise TheoremParseError(f"missing ')' to close '(' at column {col} (found {where})")
            self.take()
            return node
        raise TheoremParseError(f"unexpected {val!r} at column {col}")


def parse_expr(text: str) -> Expr:
    """Parse a single ring expression (no '=')."""
    if text is None or not str(text).strip():
        raise TheoremParseError("statement is empty")
    p = _Parser(_tokenise(str(text)))
    node = p.expr()
    if p.peek() is not None:
        kind, val, col = p.peek()
        raise TheoremParseError(f"unexpected {val!r} at column {col}")
    return node


def parse_statement(text: str) -> Expr:
    """
    Parse `lhs = rhs` (returned as lhs + (-rhs)) or a bare expression
    (returned as-is, to be proved equal to 0).
    """
    if text is None or not str(text).strip():
        raise TheoremParseError("statement is empty")
    parts = str(text).split("=")
    if len(parts) > 2:
        raise TheoremParseError("a statement may contain at most one '='")
    if len(parts) == 2:
        lhs_s, rhs_s = parts
        if not lhs_s.strip() or not rhs_s.strip():
            raise TheoremParseError("both sides of '=' must be non-empty")
        try:
            lhs = parse_expr(lhs_s)
        except TheoremParseError as e:
            raise TheoremParseError(f"left side: {e}") from None
        try:
            rhs = parse_expr(rhs_s)
        except TheoremParseError as e:
            raise TheoremParseError(f"right side: {e}") from None
        return Add(lhs, Neg(rhs))
    return parse_expr(parts[0])


# ---------------------------------------------------------------------------
# Exact truth check
# ---------------------------------------------------------------------------

def total_degree(expr: Expr) -> int:
    """Upper bound on the total polynomial degree of expr."""
    if isinstance(expr, Const):
        return 0
    if isinstance(expr, Var):
        return 1
    if isinstance(expr, Neg):
        return total_degree(expr.child)
    if isinstance(expr, Add):
        return max(total_degree(expr.left), total_degree(expr.right))
    if isinstance(expr, Mul):
        return total_degree(expr.left) + total_degree(expr.right)
    raise TypeError(f"Unknown Expr subtype: {type(expr)}")


def identity_counterexample(expr: Expr) -> Optional[Dict[str, int]]:
    """
    Return None if expr == 0 is an identity of Z[a, b, c]; otherwise an
    assignment {a, b, c} where expr evaluates to a non-zero integer.
    """
    d = max(1, total_degree(expr))
    if d > MAX_DEGREE:
        raise TheoremParseError(
            f"polynomial degree {d} is above the limit of {MAX_DEGREE}"
        )
    pts = range(d + 1)
    for va, vb, vc in product(pts, pts, pts):
        env = {"a": va, "b": vb, "c": vc}
        if expr.eval(env) != 0:
            return env
    return None


# ---------------------------------------------------------------------------
# Theorem records + file import
# ---------------------------------------------------------------------------

@dataclass
class Theorem:
    id: str
    statement: str
    expr: Expr
    notes: str = ""


@dataclass
class ImportReport:
    theorems: List[Theorem] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors


def make_theorem(statement: str, theorem_id: str = "T1", notes: str = "") -> Theorem:
    """Parse, size-check and truth-check one statement; raise TheoremParseError if unusable."""
    expr = parse_statement(statement)
    if expr.size() > MAX_SIZE:
        raise TheoremParseError(
            f"statement expands to {expr.size()} nodes; the limit is {MAX_SIZE}"
        )
    cex = identity_counterexample(expr)
    if cex is not None:
        val = expr.eval(cex)
        raise TheoremParseError(
            "not a ring identity — at "
            + ", ".join(f"{k}={v}" for k, v in cex.items())
            + f" the two sides differ by {val}"
        )
    return Theorem(id=str(theorem_id), statement=str(statement).strip(), expr=expr, notes=notes)


def _rows_from_csv_text(text: str, delimiter: str) -> List[Dict[str, str]]:
    reader = csv.DictReader(io.StringIO(text), delimiter=delimiter)
    if reader.fieldnames is None:
        raise TheoremImportError("file is empty — expected a header row with a 'statement' column")
    return [dict(r) for r in reader]


def _rows_from_xlsx(path: Path) -> List[Dict[str, str]]:
    try:
        import openpyxl  # optional dependency, only needed for .xlsx
    except ImportError:
        raise TheoremImportError(
            "reading .xlsx needs openpyxl (`python3 -m pip install openpyxl`), "
            "or save the sheet as CSV"
        ) from None
    wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
    ws = wb.worksheets[0]
    rows = list(ws.iter_rows(values_only=True))
    wb.close()
    if not rows:
        raise TheoremImportError("sheet is empty — expected a header row with a 'statement' column")
    header = [str(h).strip() if h is not None else "" for h in rows[0]]
    out = []
    for r in rows[1:]:
        out.append({h: ("" if v is None else str(v)) for h, v in zip(header, r) if h})
    return out


def _rows_from_json(text: str) -> List[Dict[str, str]]:
    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        raise TheoremImportError(f"invalid JSON: {e}") from None
    if isinstance(data, dict):
        data = data.get("theorems")
    if not isinstance(data, list):
        raise TheoremImportError(
            "JSON must be a list of statements / objects, or {\"theorems\": [...]}"
        )
    rows = []
    for item in data:
        if isinstance(item, str):
            rows.append({"statement": item})
        elif isinstance(item, dict):
            rows.append({k: ("" if v is None else str(v)) for k, v in item.items()})
        else:
            rows.append({"statement": ""})
    return rows


def _rows_from_txt(text: str) -> List[Dict[str, str]]:
    rows = []
    for line in text.splitlines():
        s = line.strip()
        rows.append({"statement": s, "_skip": "1" if (not s or s.startswith("#")) else ""})
    return rows


def read_rows(path: Union[str, Path]) -> List[Dict[str, str]]:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        raise TheoremImportError(
            f"unsupported file type {suffix!r}; use one of {', '.join(SUPPORTED_SUFFIXES)}"
        )
    if not path.exists():
        raise TheoremImportError(f"file not found: {path}")
    if suffix == ".xlsx":
        return _rows_from_xlsx(path)
    text = path.read_text(encoding="utf-8-sig")
    if suffix == ".json":
        return _rows_from_json(text)
    if suffix == ".txt":
        return _rows_from_txt(text)
    return _rows_from_csv_text(text, "\t" if suffix == ".tsv" else ",")


def theorems_from_rows(rows: Iterable[Dict[str, str]], strict: bool = False) -> ImportReport:
    """Validate rows (dicts with 'statement' and optional 'id', 'notes')."""
    rows = list(rows)
    report = ImportReport()
    if rows and not any("statement" in {k.strip().lower() for k in r} for r in rows):
        found = sorted({k for r in rows for k in r if k and not k.startswith("_")})
        raise TheoremImportError(
            f"missing required column 'statement' (found: {', '.join(found) or 'none'})"
        )
    seen: Dict[str, int] = {}
    for n, raw in enumerate(rows, start=1):
        if raw.get("_skip"):
            continue
        row = {str(k).strip().lower(): (v or "").strip() for k, v in raw.items() if k}
        label = f"row {n}"
        stmt = row.get("statement", "")
        if not stmt:
            if not any(row.values()):
                continue                       # fully blank line
            report.errors.append(f"{label}: 'statement' is empty")
            continue
        tid = row.get("id") or f"T{n}"
        if tid in seen:
            report.errors.append(f"{label}: duplicate id {tid!r} (first used on row {seen[tid]})")
            continue
        try:
            thm = make_theorem(stmt, tid, row.get("notes", ""))
        except TheoremParseError as e:
            report.errors.append(f"{label} ({tid}): {e}")
            continue
        seen[tid] = n
        report.theorems.append(thm)
    if strict and report.errors:
        raise TheoremImportError(
            f"{len(report.errors)} invalid row(s)", report.errors
        )
    return report


def load_theorems(path: Union[str, Path], strict: bool = False) -> ImportReport:
    """Load and validate theorems from a .csv/.tsv/.xlsx/.json/.txt file."""
    return theorems_from_rows(read_rows(path), strict=strict)
