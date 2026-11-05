"""
fit.py — Discover a closed-form expression for your own (x, y) data.

Usage (from prototype_b_symbolic_regression/):
    python fit.py --file ../data/templates/regression_template.csv
    python fit.py --file measurements.xlsx --x-col dose --y-col response
    python fit.py --file data.csv --iterations 40 --batch 80 --out results/fit.json
    python fit.py --demo square_plus_one        # a built-in synthetic target

Runs the same neuro-symbolic loop as demo.py (grammar-masked transformer ->
simplify -> fit constants -> Pareto front over (MSE, complexity) -> train)
on the imported data and prints the Pareto front: the simplest expression
at each level of fit.  With no ground truth there is no "recovered" verdict;
pick from the front using your domain knowledge.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from data_io import DataImportError, as_target, load_dataset
from evaluator import eval_node
from search import run_search
from targets import ALL_TARGETS
from transformer import TinyTransformer


def front_rows(pool, x: np.ndarray, y: np.ndarray, top: int) -> List[Dict]:
    """Pareto-front candidates, simplest first, with MSE and R^2."""
    front = sorted(pool.pareto_front(), key=lambda c: (c.complexity, c.fit_mse))[:top]
    var_y = float(np.var(y)) if len(y) else 0.0
    rows = []
    for c in front:
        pred = eval_node(c.node, x, c.constants)
        mask = np.isfinite(pred)
        r2 = None
        if var_y > 0 and mask.sum() >= 2:
            r2 = 1.0 - float(np.mean((pred[mask] - y[mask]) ** 2)) / var_y
        rows.append({
            "expression": c.infix_str,
            "prefix": c.prefix_str,
            "constants": [round(float(v), 6) for v in c.constants],
            "mse": float(c.fit_mse),
            "r2": None if r2 is None else round(r2, 6),
            "complexity": int(c.complexity),
        })
    return rows


def fit_target(target, iterations: int, batch: int, train_steps: int,
               seed: int, top: int, verbose: bool) -> Dict:
    model = TinyTransformer(seed=seed)
    pool, metrics = run_search(target, model, n_iterations=iterations,
                               batch_size=batch, train_steps=train_steps,
                               seed=seed, verbose=verbose)
    x, y = target.sample(seed=seed)
    best = pool.best()
    return {
        "dataset": target.name,
        "description": target.description,
        "n_points": int(len(x)),
        "iterations": iterations,
        "best": None if best is None else {
            "expression": best.infix_str, "mse": float(best.fit_mse),
            "complexity": int(best.complexity),
        },
        "pareto_front": front_rows(pool, x, y, top),
        "metrics": metrics,
    }


def write_result(result: Dict, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.suffix.lower() == ".csv":
        with out.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["complexity", "mse", "r2", "expression", "prefix", "constants"])
            for r in result["pareto_front"]:
                w.writerow([r["complexity"], r["mse"], "" if r["r2"] is None else r["r2"],
                            r["expression"], r["prefix"], " ".join(map(str, r["constants"]))])
    else:
        out.write_text(json.dumps(result, indent=2), encoding="utf-8")


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--file", help="CSV/TSV/XLSX/JSON with x and y columns (see README)")
    src.add_argument("--demo", choices=[t.name for t in ALL_TARGETS],
                     help="run on a built-in synthetic target instead")
    ap.add_argument("--x-col", default="x", help="input column name (default x)")
    ap.add_argument("--y-col", default="y", help="output column name (default y)")
    ap.add_argument("--sheet", help="worksheet name for .xlsx (default: first sheet)")
    ap.add_argument("--iterations", type=int, default=25)
    ap.add_argument("--batch", type=int, default=60, help="candidates per iteration")
    ap.add_argument("--train-steps", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--top", type=int, default=10, help="Pareto-front rows to show")
    ap.add_argument("--out", help="write results to this .json or .csv file")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)

    if args.file:
        try:
            ds = load_dataset(args.file, args.x_col, args.y_col, sheet=args.sheet)
        except DataImportError as e:
            print(f"error: {e}", file=sys.stderr)
            for line in e.errors[:20]:
                print(f"  - {line}", file=sys.stderr)
            if len(e.errors) > 20:
                print(f"  ... and {len(e.errors) - 20} more", file=sys.stderr)
            return 2
        for w in ds.warnings:
            print(f"warning: {w}", file=sys.stderr)
        target = as_target(ds)
    else:
        target = next(t for t in ALL_TARGETS if t.name == args.demo)

    print(f"Searching for f({args.x_col}) on {target.description} "
          f"({args.iterations} iterations x {args.batch} candidates)")
    result = fit_target(target, args.iterations, args.batch, args.train_steps,
                        args.seed, args.top, verbose=not args.quiet)

    print("\nPareto front (simplest first; x is your input column):")
    print(f"  {'cplx':>4}  {'MSE':>11}  {'R^2':>8}  expression")
    for r in result["pareto_front"]:
        r2 = "-" if r["r2"] is None else f"{r['r2']:.4f}"
        consts = f"   C={r['constants']}" if r["constants"] else ""
        print(f"  {r['complexity']:>4}  {r['mse']:>11.6g}  {r2:>8}  {r['expression']}{consts}")
    if args.out:
        write_result(result, Path(args.out))
        print(f"\nResults written to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
