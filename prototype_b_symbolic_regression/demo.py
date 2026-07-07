"""
demo.py — Runnable demo for Prototype B: Hybrid Symbolic Regression.

Usage:
    cd prototype_b_symbolic_regression
    python demo.py

What this does
--------------
  For each of 6 target identities (4 training, 2 OOD):
    1. Run the neuro-symbolic search loop (25 iterations, 80 candidates/iter).
    2. Check whether the recovered expression is symbolically equivalent
       to the ground truth (SymPy oracle).
    3. Print results and save to results/.

  Then:
    4. Run a pure-random baseline (no transformer — uniform sampling) on the
       same targets to show the neural component is load-bearing.
    5. Print ablation table.

Reproducing: fixed seeds, re-run produces identical numbers.
"""

import sys
import json
import time
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from grammar import ALL_TOKENS, VOCAB_SIZE, is_valid_prefix
from transformer import TinyTransformer
from pareto import CandidatePool
from targets import ALL_TARGETS, TRAIN_TARGETS, OOD_TARGETS
from search import run_search, evaluate_recovery
from evaluator import eval_node, mse
from grammar import parse

RESULTS_DIR = str(Path(__file__).parent / "results")
SEED = 42

BANNER = "=" * 70


def random_baseline_search(target, n_candidates=2000, seed=0):
    """
    Pure-random candidate generation — no transformer.
    Uniform sampling from the grammar: generates random valid prefix sequences.
    """
    from grammar import BINARY_OPS, UNARY_OPS, TERMINALS, arity, expected_remaining
    import random

    rng   = random.Random(seed)
    pool  = CandidatePool(max_size=500)
    x_vals, y_vals = target.sample(seed=seed)

    expandable_tokens = BINARY_OPS + UNARY_OPS + TERMINALS

    for _ in range(n_candidates):
        # Build a random valid prefix sequence
        tokens = []
        need   = 1
        for _ in range(15):
            if need == 0:
                break
            # Only consider tokens that keep completion possible
            candidates_t = []
            for tok in expandable_tokens:
                a = arity(tok)
                new_need = need - 1 + a
                remaining = 14 - len(tokens)
                if 0 <= new_need <= remaining:
                    candidates_t.append(tok)
            if not candidates_t:
                break
            tok = rng.choice(candidates_t)
            tokens.append(tok)
            need = need - 1 + arity(tok)

        if need == 0 and is_valid_prefix(tokens):
            pool.add(tokens, x_vals, y_vals)

    pool.compute_pareto()
    return pool


def main():
    np.random.seed(SEED)
    Path(RESULTS_DIR).mkdir(parents=True, exist_ok=True)

    print(BANNER)
    print("Prototype B — Hybrid Symbolic Regression")
    print("  Neural  : autoregressive transformer (grammar-masked sampling)")
    print("  Symbolic: from-scratch simplifier + SymPy equivalence oracle")
    print("  Signal  : Pareto front over (MSE, symbolic complexity)")
    print(BANNER)

    # ------------------------------------------------------------------
    # Interlock sanity check
    # ------------------------------------------------------------------
    print("\n[0] Interlock sanity check")
    print("  Neural → Symbolic: grammar mask constrains transformer output")
    print("  Symbolic → Neural: Pareto advantage weights train transformer")

    model = TinyTransformer(seed=SEED)
    # Generate 5 samples and verify grammar mask works
    for i in range(5):
        toks = model.generate(temperature=1.2, grammar_mask=True, seed=i)
        valid = is_valid_prefix(toks)
        print(f"    sample {i+1}: {' '.join(toks):<30} valid={valid}")

    # ------------------------------------------------------------------
    # Search on all 6 targets
    # ------------------------------------------------------------------
    all_results = []
    all_metrics = {}

    for target in ALL_TARGETS:
        label = "OOD" if target.ood else "TRAIN"
        print(f"\n[{'OOD' if target.ood else 'TRAIN'}] Target: {target.description}")
        print(f"  Ground truth tokens: {' '.join(target.tokens)}")

        pool, metrics = run_search(
            target, model,
            n_iterations=25,
            batch_size=80,
            train_steps=5,
            temperature=1.0,
            seed=SEED,
            verbose=True,
        )

        result = evaluate_recovery(target, pool, seed=SEED + 1)
        all_results.append(result)
        all_metrics[target.name] = metrics

        status = "✓ RECOVERED" if result["recovered"] else "✗ NOT FOUND"
        print(f"  → {status}")
        print(f"     Ground truth : {target.description}")
        print(f"     Best found   : {result['best_expr']}")
        print(f"     Best MSE     : {result['best_mse']:.6f}" if result['best_mse'] else "")

    # ------------------------------------------------------------------
    # Recovery summary
    # ------------------------------------------------------------------
    print(f"\n{'='*70}")
    print("Recovery summary")
    print(f"{'='*70}")
    n_train_ok = sum(1 for r in all_results if not r["ood"] and r["recovered"])
    n_ood_ok   = sum(1 for r in all_results if r["ood"]  and r["recovered"])
    n_train    = sum(1 for r in all_results if not r["ood"])
    n_ood      = sum(1 for r in all_results if r["ood"])

    print(f"  {'Target':<20} {'Split':>6}  {'MSE':>10}  {'Recovered':>10}  Expression")
    print(f"  {'-'*20} {'-'*6}  {'-'*10}  {'-'*10}  {'-'*20}")
    for r in all_results:
        split   = "OOD" if r["ood"] else "TRAIN"
        mse_s   = f"{r['best_mse']:.4f}" if r["best_mse"] else "—"
        rec_s   = "YES" if r["recovered"] else "NO"
        expr_s  = (r["best_expr"] or "—")[:30]
        print(f"  {r['target']:<20} {split:>6}  {mse_s:>10}  {rec_s:>10}  {expr_s}")

    print(f"\n  Training: {n_train_ok}/{n_train} recovered")
    print(f"  OOD:      {n_ood_ok}/{n_ood} recovered")
    print(f"  Total:    {n_train_ok+n_ood_ok}/{n_train+n_ood} recovered")
    print(f"\n  Falsification target: ≥4/6 exact recoveries")
    total_ok = n_train_ok + n_ood_ok
    print(f"  Result: {'PASSED' if total_ok >= 4 else 'FAILED'} ({total_ok}/6)")

    # ------------------------------------------------------------------
    # Ablation: random baseline vs trained transformer
    # ------------------------------------------------------------------
    print(f"\n[Ablation] Random baseline (uniform grammar sampling, no transformer)")
    ablation_results = []
    for target in ALL_TARGETS:
        pool_rand = random_baseline_search(target, n_candidates=2000, seed=SEED)
        result_rand = evaluate_recovery(target, pool_rand, seed=SEED + 1)
        ablation_results.append(result_rand)
        label = "OOD" if target.ood else "TRAIN"
        status = "✓" if result_rand["recovered"] else "✗"
        mse_s  = f"{result_rand['best_mse']:.4f}" if result_rand["best_mse"] else "—"
        print(f"  [{label}] {target.name:<20}  mse={mse_s}  recovered={status}")

    n_rand_ok = sum(1 for r in ablation_results if r["recovered"])
    print(f"\n  Random baseline total: {n_rand_ok}/6 recovered")
    print(f"  Transformer total:     {n_train_ok+n_ood_ok}/6 recovered")

    # ------------------------------------------------------------------
    # Save all results
    # ------------------------------------------------------------------
    print(f"\n[Saving results to {RESULTS_DIR}/]")
    out = {
        "recovery": all_results,
        "ablation": ablation_results,
    }
    (Path(RESULTS_DIR) / "recovery_results.json").write_text(json.dumps(out, indent=2))
    (Path(RESULTS_DIR) / "training_metrics.json").write_text(
        json.dumps(all_metrics, indent=2)
    )
    print("  Done.")
    print(f"\n{BANNER}")


if __name__ == "__main__":
    main()
