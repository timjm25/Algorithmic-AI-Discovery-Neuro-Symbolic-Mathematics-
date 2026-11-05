"""
prove.py — Run the neural-guided proof search on your own ring identities.

Usage (from prototype_a_proof_search/):
    python prove.py --file ../data/templates/theorems_template.csv
    python prove.py --statement "a*(b+c) = a*b + a*c"
    python prove.py --demo 5                      # built-in generated theorems
    python prove.py --file my.csv --out results/my_proofs.json   (.json or .csv)

Every row is parsed, checked to be a true identity of Z[a, b, c] (exactly, see
theorem_io.py) and then handed to best-first search.  Each step of a returned
proof was certified by the kernel.  "not proved" means the statement is true
but the 17 kernel rules did not reach 0 within the node budget — the rules
only rewrite in one direction (e.g. they expand a*(b+c) but never factor),
so some true identities are out of reach.

Policy: by default the trained weights in results/policy_weights.npz are
loaded if present (written by demo.py); pass --uniform for the no-learning
baseline.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).parent))

from policy import MLPPolicy
from search import best_first_search, uniform_search, ProofResult
from theorem_io import (
    Theorem, ImportReport, TheoremImportError, TheoremParseError,
    load_theorems, make_theorem,
)
from theorems import generate_training_theorems

DEFAULT_WEIGHTS = Path(__file__).parent / "results" / "policy_weights.npz"


def load_policy(weights: Optional[str], uniform: bool):
    """Return (policy_or_None, description).  None means uniform search."""
    if uniform:
        return None, "uniform baseline"
    path = Path(weights) if weights else DEFAULT_WEIGHTS
    if path.exists():
        pol = MLPPolicy(seed=0)
        pol.load(str(path))
        return pol, f"trained MLP policy ({path.name})"
    if weights:
        raise FileNotFoundError(f"policy weights not found: {path}")
    return None, "uniform baseline (no trained weights found — run demo.py to train)"


def prove_one(thm: Theorem, policy, max_nodes: int, beam_width: int) -> Dict:
    if policy is None:
        res: ProofResult = uniform_search(thm.expr, max_nodes=max_nodes, beam_width=beam_width)
    else:
        res = best_first_search(thm.expr, policy, max_nodes=max_nodes, beam_width=beam_width)
    return {
        "id": thm.id,
        "statement": thm.statement,
        "notes": thm.notes,
        "goal": f"{thm.expr!r} = 0",
        "status": "proved" if res.found else "not_proved",
        "proof_length": res.proof_length if res.found else None,
        "nodes_expanded": res.nodes_expanded,
        "steps": [
            {"rule": s.rule_name, "path": s.path, "result": repr(s.expr_after)}
            for s in res.steps
        ],
    }


def run(theorems: List[Theorem], policy, max_nodes: int, beam_width: int,
        verbose: bool = True) -> List[Dict]:
    results = []
    for thm in theorems:
        r = prove_one(thm, policy, max_nodes, beam_width)
        results.append(r)
        if verbose:
            print(f"\n[{r['id']}] {r['statement']}")
            print(f"  goal  : {r['goal']}")
            if r["status"] == "proved":
                for i, s in enumerate(r["steps"], 1):
                    print(f"  step {i}: {s['rule']:<13} at {s['path']!s:<10} -> {s['result']}")
                print(f"  QED   ({r['proof_length']} kernel-certified steps, "
                      f"{r['nodes_expanded']} nodes expanded)")
            else:
                print(f"  not proved within {max_nodes} nodes "
                      f"(true identity; try a larger --max-nodes or --beam)")
    return results


def write_results(results: List[Dict], out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.suffix.lower() == ".csv":
        with out.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["id", "statement", "status", "proof_length", "nodes_expanded", "proof"])
            for r in results:
                proof = " ; ".join(f"{s['rule']}@{s['path']}" for s in r["steps"])
                w.writerow([r["id"], r["statement"], r["status"], r["proof_length"] or "",
                            r["nodes_expanded"], proof])
    else:
        out.write_text(json.dumps(results, indent=2), encoding="utf-8")


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--file", help="CSV/TSV/XLSX/JSON/TXT of statements (see README for the schema)")
    src.add_argument("--statement", action="append", help="a single statement; repeatable")
    src.add_argument("--demo", type=int, metavar="N", help="prove N built-in generated theorems")
    ap.add_argument("--policy", help=f"policy weights .npz (default {DEFAULT_WEIGHTS.name} if present)")
    ap.add_argument("--uniform", action="store_true", help="use the uniform (no-learning) baseline")
    ap.add_argument("--max-nodes", type=int, default=2000, help="search budget per theorem (default 2000)")
    ap.add_argument("--beam", type=int, default=8, help="moves expanded per node (default 8)")
    ap.add_argument("--strict", action="store_true", help="stop if any row is invalid")
    ap.add_argument("--out", help="write results to this .json or .csv file")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)

    report = ImportReport()
    try:
        if args.file:
            report = load_theorems(args.file, strict=args.strict)
        elif args.statement:
            for i, s in enumerate(args.statement, 1):
                try:
                    report.theorems.append(make_theorem(s, f"S{i}"))
                except TheoremParseError as e:
                    report.errors.append(f"statement {i}: {e}")
        else:
            for i, e in enumerate(generate_training_theorems(args.demo, seed=123), 1):
                report.theorems.append(Theorem(id=f"DEMO{i}", statement=repr(e), expr=e,
                                               notes="generated demo theorem"))
    except TheoremImportError as e:
        print(f"error: {e}", file=sys.stderr)
        for line in e.errors:
            print(f"  - {line}", file=sys.stderr)
        return 2

    for line in report.errors:
        print(f"skipped {line}", file=sys.stderr)
    if not report.theorems:
        print("error: no valid statements to prove", file=sys.stderr)
        return 2

    try:
        policy, desc = load_policy(args.policy, args.uniform)
    except FileNotFoundError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    if not args.quiet:
        print(f"Proving {len(report.theorems)} statement(s) with {desc}; "
              f"budget {args.max_nodes} nodes, beam {args.beam}")

    results = run(report.theorems, policy, args.max_nodes, args.beam, verbose=not args.quiet)
    n_ok = sum(r["status"] == "proved" for r in results)
    print(f"\nProved {n_ok}/{len(results)}"
          + (f"; {len(report.errors)} row(s) skipped as invalid" if report.errors else ""))
    if args.out:
        write_results(results, Path(args.out))
        print(f"Results written to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
