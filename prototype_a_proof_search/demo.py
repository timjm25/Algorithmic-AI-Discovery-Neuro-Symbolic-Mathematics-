"""
demo.py — Runnable demo for Prototype A: Neural-guided proof search.

Usage:
    cd prototype_a_proof_search
    python demo.py

What this script does
---------------------
  1. Initialises a fresh MLPPolicy (random weights).
  2. Runs 12 expert-iteration cycles (each: generate theorems → search →
     collect traces → train).
  3. Runs the ablation: trained policy vs. uniform baseline on 100 training
     theorems and 100 OOD theorems.
  4. Prints a worked proof of a held-out theorem step by step.
  5. Saves all metrics to results/.

Reproducing:  fixed seeds throughout; re-run produces identical numbers.
"""

import sys
import json
import numpy as np
from pathlib import Path

# Ensure imports work when run from within the prototype dir
sys.path.insert(0, str(Path(__file__).parent))

from expressions import Expr, Add, Mul, Neg, C, V, a, b, c
from kernel import KERNEL
from policy import MLPPolicy
from search import best_first_search, uniform_search
from train import expert_iteration, ablation, save_results
from theorems import generate_training_theorems, generate_ood_theorems

RESULTS_DIR = str(Path(__file__).parent / "results")
SEED = 7

BANNER = "=" * 65


def print_proof(theorem: Expr, result) -> None:
    """Pretty-print a certified proof trace."""
    print(f"  Start : {theorem!r}")
    for i, step in enumerate(result.steps, 1):
        print(f"  Step {i}: apply '{step.rule_name}' at path {step.path}")
        print(f"         → {step.expr_after!r}")
    print(f"  QED   ({len(result.steps)} steps, {result.nodes_expanded} nodes expanded)")


def main():
    np.random.seed(SEED)

    print(BANNER)
    print("Prototype A — Neural-guided proof search")
    print("  Symbolic kernel : ring axioms (17 rules)")
    print("  Neural policy   : 2-layer MLP (NumPy, hand-rolled backprop)")
    print("  Training        : expert iteration (behavioural cloning)")
    print(BANNER)

    # -------------------------------------------------------------------
    # Sanity-check the kernel on one step before training anything
    # -------------------------------------------------------------------
    print("\n[0] Kernel sanity check")
    # a*(b+c) - (a*b + a*c)  →  apply dist_l at path [0]  →  a*b+a*c - (a*b+a*c)
    #                         →  apply add_inv at root     →  0
    theorem_demo = Add(Mul(a, Add(b, c)), Neg(Add(Mul(a, b), Mul(a, c))))
    print(f"  Theorem : {theorem_demo!r}")
    step1 = KERNEL.apply(4, [0], theorem_demo)        # dist_l at left child
    print(f"  After dist_l at [0] : {step1!r}")
    step2 = KERNEL.apply(12, [], step1)               # add_inv at root
    step2 = KERNEL.constant_fold(step2)
    print(f"  After add_inv at [] : {step2!r}")
    assert KERNEL.is_zero(step2), "Kernel sanity check failed!"
    print("  Kernel certified proof: OK")

    # -------------------------------------------------------------------
    # Expert iteration training
    # -------------------------------------------------------------------
    print(f"\n[1] Expert iteration ({12} cycles, 60 theorems/cycle)")
    policy = MLPPolicy(lr=0.005, seed=SEED)
    metrics = expert_iteration(
        policy,
        n_iterations=12,
        batch_size=60,
        train_epochs=5,
        seed=SEED,
        verbose=True,
    )

    # -------------------------------------------------------------------
    # Show learning curve summary
    # -------------------------------------------------------------------
    print("\n[2] Learning curve")
    print(f"  {'Iter':>4}  {'Proof rate':>11}  {'Mean nodes':>10}")
    for row in metrics:
        nodes_str = f"{row['mean_nodes']}" if row["mean_nodes"] is not None else "—"
        print(f"  {row['iteration']:>4}  {row['proof_rate']:>10.1%}  {nodes_str:>10}")

    # -------------------------------------------------------------------
    # Ablation: trained policy vs. uniform baseline
    # -------------------------------------------------------------------
    print("\n[3] Ablation: trained policy vs. uniform baseline (100 theorems each)")
    abl = ablation(policy, n_theorems=100, seed=42, verbose=True)

    # Summarise node reduction
    for split in ("train", "ood"):
        p_n = abl[split]["policy_mean_nodes"]
        u_n = abl[split]["uniform_mean_nodes"]
        if p_n and u_n:
            reduction = (u_n - p_n) / u_n * 100
            print(f"  [{split}] node reduction: {reduction:.1f}% (policy={p_n}, uniform={u_n})")

    # -------------------------------------------------------------------
    # Prove a held-out OOD theorem step by step
    # -------------------------------------------------------------------
    print("\n[4] Worked proof of a held-out OOD theorem")
    ood = generate_ood_theorems(10, seed=0)
    proved = None
    for t in ood:
        r = best_first_search(t, policy, max_nodes=300, beam_width=8)
        if r.found:
            proved = (t, r)
            break

    if proved:
        print_proof(*proved)
    else:
        print("  (no OOD theorem solved — increase max_nodes)")

    # -------------------------------------------------------------------
    # Verify summation theorem on first OOD proof
    # -------------------------------------------------------------------
    if proved:
        print("\n[5] Correctness audit: every step re-verified by kernel")
        expr = proved[0]
        for i, step in enumerate(proved[1].steps):
            try:
                rechecked = KERNEL.apply(step.rule_idx, step.path, expr)
                rechecked = KERNEL.constant_fold(rechecked)
                assert repr(rechecked) == repr(step.expr_after), \
                    f"Step {i+1} mismatch!"
                expr = step.expr_after
            except Exception as e:
                print(f"  Step {i+1} FAILED re-verification: {e}")
                break
        else:
            print(f"  All {len(proved[1].steps)} steps pass kernel re-verification: OK")

    # -------------------------------------------------------------------
    # Save results
    # -------------------------------------------------------------------
    print("\n[6] Saving results")
    save_results(metrics, abl, RESULTS_DIR)

    # Save policy weights
    policy.save(f"{RESULTS_DIR}/policy_weights.npz")
    print(f"  Policy weights saved to {RESULTS_DIR}/policy_weights.npz")

    print(f"\n{BANNER}")
    print("Done.")


if __name__ == "__main__":
    main()
