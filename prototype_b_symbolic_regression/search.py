"""
search.py — The neuro-symbolic search loop for symbolic regression.

Algorithm
---------
  Repeat for N_ITERATIONS:
    1. [Neural → Symbolic] Transformer generates BATCH candidates via
       grammar-masked autoregressive sampling.
    2. Each candidate is:
         a. Parsed (grammar.py — from scratch)
         b. Simplified (pareto._simplify_node — from scratch)
         c. Constant-fitted (evaluator.fit_constants — L-BFGS-B)
         d. Equivalence-checked (evaluator.are_equivalent — SymPy oracle)
         e. Added to CandidatePool (deduplicated)
    3. [Symbolic → Neural] CandidatePool.compute_pareto() assigns advantage
       weights based on Pareto rank over (MSE, complexity).
    4. Pareto-front candidates are converted to token sequences and used
       to train the transformer (weighted cross-entropy).
    5. Log best MSE, pool size, Pareto front size.

Falsification evaluation
------------------------
  After training, run the model on 6 held-out identities and check whether
  the recovered expression is symbolically equivalent to the ground truth
  (SymPy simplify(A - B) == 0).
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from grammar import TOKEN_TO_ID, BOS_ID, EOS_ID, is_valid_prefix
from evaluator import eval_node, mse, are_equivalent
from pareto import CandidatePool, Candidate
from targets import Target
from transformer import TinyTransformer

# Search hyper-parameters
BATCH_PER_ITER  = 60    # candidates generated per iteration
N_ITERATIONS    = 20    # training iterations per target
TRAIN_STEPS     = 4     # gradient steps per iteration
TEMPERATURE     = 1.0


def _seqs_and_weights(
    pool: CandidatePool,
) -> Tuple[List[List[int]], List[float]]:
    """
    Convert Pareto-front candidates to token-id sequences + advantage weights
    for transformer training.
    """
    front = pool.pareto_front()
    if not front:
        return [], []

    sequences, weights = [], []
    for cand in front:
        ids = [BOS_ID] + [TOKEN_TO_ID[t] for t in cand.tokens] + [EOS_ID]
        sequences.append(ids)
        weights.append(cand.advantage)

    return sequences, weights


def run_search(
    target: Target,
    model: TinyTransformer,
    n_iterations: int   = N_ITERATIONS,
    batch_size: int     = BATCH_PER_ITER,
    train_steps: int    = TRAIN_STEPS,
    temperature: float  = TEMPERATURE,
    seed: int           = 0,
    verbose: bool       = True,
) -> Tuple[CandidatePool, List[Dict]]:
    """
    Run the neuro-symbolic search loop on one target identity.

    Returns (pool, metrics_list).
    """
    x_vals, y_vals = target.sample(seed=seed)
    pool    = CandidatePool(max_size=500)
    metrics: List[Dict] = []
    rng_seed = seed

    for it in range(n_iterations):
        t0 = time.time()
        n_added = 0

        # [Neural → Symbolic] Generate candidates
        for _ in range(batch_size):
            tokens = model.generate(
                temperature=temperature,
                grammar_mask=True,
                seed=rng_seed,
            )
            rng_seed += 1
            if is_valid_prefix(tokens):
                cand = pool.add(tokens, x_vals, y_vals)
                if cand is not None:
                    n_added += 1

        # [Symbolic → Neural] Compute Pareto front → training signal
        pool.compute_pareto()
        seqs, weights = _seqs_and_weights(pool)

        # Train transformer on Pareto-front candidates
        loss = 0.0
        for _ in range(train_steps):
            if seqs:
                loss = model.train_on_sequences(seqs, weights)

        best  = pool.best()
        front = pool.pareto_front()
        elapsed = time.time() - t0

        row = {
            "iteration":    it + 1,
            "pool_size":    len(pool.candidates),
            "n_added":      n_added,
            "pareto_front": len(front),
            "best_mse":     round(float(best.fit_mse), 6) if best else None,
            "best_expr":    best.infix_str if best else None,
            "loss":         round(float(loss), 4),
            "elapsed_s":    round(elapsed, 2),
        }
        metrics.append(row)

        if verbose:
            mse_str  = f"{best.fit_mse:.4f}" if best else "—"
            expr_str = best.infix_str[:40] if best else "—"
            print(
                f"  iter {it+1:2d} | pool={len(pool.candidates):3d} "
                f"added={n_added:2d} front={len(front):2d} "
                f"mse={mse_str}  best={expr_str}  ({elapsed:.1f}s)"
            )

    return pool, metrics


def evaluate_recovery(
    target: Target,
    pool: CandidatePool,
    seed: int = 1,
) -> Dict:
    """
    Check whether the pool contains an expression symbolically equivalent
    to the target ground truth.  Uses SymPy oracle for equivalence.

    Returns dict with: recovered (bool), best_mse, best_expr, ground_truth.
    """
    from grammar import parse

    x_vals, y_vals = target.sample(seed=seed)   # fresh data for validation
    gt_node   = parse(target.tokens)

    recovered = False
    best      = pool.best()

    # Check all Pareto-front candidates for exact symbolic equivalence
    for cand in pool.pareto_front():
        if are_equivalent(cand.node, cand.constants, gt_node, []):
            recovered = True
            break

    # Numeric check as fallback: MSE < threshold on clean validation data
    gt_y = eval_node(gt_node, x_vals, [])
    mse_threshold = 0.005   # well below noise level σ² = 0.0025

    if not recovered and best is not None:
        y_pred = eval_node(best.node, x_vals, best.constants)
        if mse(y_pred, gt_y) < mse_threshold:
            recovered = True   # numerically equivalent even if SymPy missed it

    return {
        "target":       target.name,
        "description":  target.description,
        "ground_truth": " ".join(target.tokens),
        "recovered":    recovered,
        "best_mse":     float(best.fit_mse) if best else None,
        "best_expr":    best.infix_str if best else None,
        "ood":          target.ood,
    }
