"""
train.py — Expert iteration training loop.

Algorithm
---------
  Repeat for N iterations:
    1. Generate a batch of training theorems (scrambled known-zeros).
    2. Run best_first_search with the current policy on each theorem.
    3. Collect winning proof traces (sequence of expert actions).
    4. Train the MLP policy on those traces via cross-entropy
       (behavioural cloning / imitation learning).
    5. Log proof rate and mean nodes-to-proof.

  At the end, evaluate on both training theorems and OOD theorems
  and compare against the uniform (no-learning) baseline.

INTERLOCK in the training loop
--------------------------------
  Symbolic → Neural (reward/supervision signal):
    The kernel's is_zero() is the sole oracle for "proof found."
    Only kernel-certified traces are used as training data.
    The policy can only improve by finding moves that lead to
    kernel-verified proofs — it cannot cheat.

  Neural → Symbolic (search guidance):
    The trained policy steers search by lowering the priority of
    unproductive branches.  After each iteration the policy should
    expand fewer nodes to reach QED.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from policy import MLPPolicy
from search import best_first_search, uniform_search, ProofResult
from theorems import generate_training_theorems, generate_ood_theorems

# Search hyper-parameters
MAX_NODES   = 300
BEAM_WIDTH  = 8

# Expert-iteration hyper-parameters
N_ITERATIONS    = 12
BATCH_SIZE      = 60   # theorems per iteration
TRAIN_EPOCHS    = 5    # gradient steps per example batch


def _collect_examples(
    results: List[ProofResult],
) -> List[Tuple]:
    """
    Flatten successful proof traces into (expr, legal_moves, expert_idx) tuples
    for supervised training.

    For each step in a proof, the "expert action" is the (rule_idx, path) the
    search took.  We re-compute legal_moves at that state to form the
    classification problem.
    """
    from kernel import KERNEL
    examples = []
    for res in results:
        if not res.found:
            continue
        for step in res.steps:
            expr = step.expr_before
            moves = KERNEL.legal_moves(expr)
            if not moves:
                continue
            # Find the index of the expert action
            try:
                idx = moves.index((step.rule_idx, step.path))
            except ValueError:
                continue   # move not found in list (shouldn't happen)
            examples.append((expr, moves, idx))
    return examples


def run_batch(
    policy: MLPPolicy,
    theorems: List,
    max_nodes: int = MAX_NODES,
    beam_width: int = BEAM_WIDTH,
) -> Tuple[List[ProofResult], float, float]:
    """
    Run search on a batch of theorems with the given policy.
    Returns (results, proof_rate, mean_nodes_on_success).
    """
    results = [
        best_first_search(t, policy, max_nodes=max_nodes, beam_width=beam_width)
        for t in theorems
    ]
    solved  = [r for r in results if r.found]
    rate    = len(solved) / len(results)
    mean_n  = float(np.mean([r.nodes_expanded for r in solved])) if solved else float("nan")
    return results, rate, mean_n


def expert_iteration(
    policy: MLPPolicy,
    n_iterations: int = N_ITERATIONS,
    batch_size: int = BATCH_SIZE,
    train_epochs: int = TRAIN_EPOCHS,
    seed: int = 0,
    verbose: bool = True,
) -> List[Dict]:
    """
    Main training loop.  Returns a list of per-iteration metric dicts.
    """
    metrics: List[Dict] = []

    for it in range(n_iterations):
        t0 = time.time()

        # 1. Generate theorem batch
        theorems = generate_training_theorems(
            batch_size, max_scramble=4, seed=seed + it * 1000
        )

        # 2. Search with current policy → collect proof traces
        results, rate, mean_nodes = run_batch(policy, theorems)

        # 3. Build supervised training examples from winning traces
        examples = _collect_examples(results)

        # 4. Train policy (SYMBOLIC → NEURAL: only kernel-verified traces used)
        losses = []
        for _ in range(train_epochs):
            if examples:
                loss = policy.train_step(examples)
                losses.append(loss)

        mean_loss = float(np.mean(losses)) if losses else float("nan")
        elapsed = time.time() - t0

        row = {
            "iteration":    it + 1,
            "proof_rate":   round(rate, 3),
            "mean_nodes":   round(mean_nodes, 1) if not np.isnan(mean_nodes) else None,
            "n_examples":   len(examples),
            "mean_loss":    round(mean_loss, 4) if not np.isnan(mean_loss) else None,
            "elapsed_s":    round(elapsed, 1),
        }
        metrics.append(row)

        if verbose:
            print(
                f"  iter {it+1:2d} | "
                f"proof_rate={rate:.2%}  "
                f"mean_nodes={row['mean_nodes']}  "
                f"n_examples={len(examples)}  "
                f"loss={row['mean_loss']}  "
                f"({elapsed:.1f}s)"
            )

    return metrics


def ablation(
    policy: MLPPolicy,
    n_theorems: int = 100,
    seed: int = 42,
    verbose: bool = True,
) -> Dict:
    """
    Compare trained policy vs uniform baseline on both in-distribution
    and OOD theorem sets.
    """
    train_theorems = generate_training_theorems(n_theorems, seed=seed)
    ood_theorems   = generate_ood_theorems(n_theorems, seed=seed + 1)

    results_dict = {}

    for label, theorems in [("train", train_theorems), ("ood", ood_theorems)]:
        # Trained policy
        _, rate_pol, nodes_pol = run_batch(policy, theorems)
        # Uniform baseline
        uni_results = [uniform_search(t, MAX_NODES, BEAM_WIDTH) for t in theorems]
        solved_uni  = [r for r in uni_results if r.found]
        rate_uni    = len(solved_uni) / len(uni_results)
        nodes_uni   = (
            float(np.mean([r.nodes_expanded for r in solved_uni]))
            if solved_uni else float("nan")
        )

        results_dict[label] = {
            "policy_proof_rate":    round(rate_pol, 3),
            "policy_mean_nodes":    round(nodes_pol, 1) if not np.isnan(nodes_pol) else None,
            "uniform_proof_rate":   round(rate_uni, 3),
            "uniform_mean_nodes":   round(nodes_uni, 1) if not np.isnan(nodes_uni) else None,
        }

        if verbose:
            print(
                f"  [{label}]  "
                f"policy  rate={rate_pol:.2%}  nodes={results_dict[label]['policy_mean_nodes']}   "
                f"uniform rate={rate_uni:.2%}  nodes={results_dict[label]['uniform_mean_nodes']}"
            )

    return results_dict


def save_results(metrics: List[Dict], ablation_data: Dict, out_dir: str) -> None:
    p = Path(out_dir)
    p.mkdir(parents=True, exist_ok=True)
    (p / "training_metrics.json").write_text(json.dumps(metrics, indent=2))
    (p / "ablation.json").write_text(json.dumps(ablation_data, indent=2))
    print(f"  Results written to {p}/")
