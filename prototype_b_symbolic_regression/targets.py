"""
targets.py — Ground-truth identities for the falsification test.

Six known mathematical identities sampled at x ∈ (0, 2] with Gaussian noise.
The model must recover the exact symbolic form (verified by SymPy equivalence).

Training / OOD split:
  Training identities (4): used to warm-start the transformer via
    a small set of ground-truth sequences.
  OOD identities (2): never seen during training; model must generalise
    the learned compositional structure.

Noise level σ = 0.05 on all identities.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List
import numpy as np

N_POINTS = 200
SIGMA    = 0.05
X_LO, X_HI = 0.1, 2.0    # avoid x=0 for inv/sqrt/div


@dataclass
class Target:
    name: str
    tokens: List[str]      # ground-truth prefix token sequence
    description: str
    ood: bool = False      # True = not shown during training

    def sample(self, seed: int = 0) -> tuple:
        """Return (x_vals, y_vals) with Gaussian noise."""
        rng = np.random.default_rng(seed)
        x   = rng.uniform(X_LO, X_HI, N_POINTS)
        from grammar import parse
        from evaluator import eval_node
        node  = parse(self.tokens)
        y_clean = eval_node(node, x, [])
        y     = y_clean + rng.normal(0, SIGMA, N_POINTS)
        return x, y


# All identities expressible in our grammar with no free constants (C tokens)
ALL_TARGETS = [
    # ---- Training identities ----
    Target(
        name="square",
        tokens=["pow2", "x"],
        description="f(x) = x²",
        ood=False,
    ),
    Target(
        name="inv_x",
        tokens=["inv", "x"],
        description="f(x) = 1/x",
        ood=False,
    ),
    Target(
        name="x_plus_inv",
        tokens=["add", "x", "inv", "x"],
        description="f(x) = x + 1/x",
        ood=False,
    ),
    Target(
        name="square_plus_one",
        tokens=["add", "pow2", "x", "1"],
        description="f(x) = x² + 1",
        ood=False,
    ),
    # ---- OOD identities (not used in training) ----
    Target(
        name="x_minus_inv",
        tokens=["sub", "x", "inv", "x"],
        description="f(x) = x - 1/x",
        ood=True,
    ),
    Target(
        name="sqrt_x",
        tokens=["sqrt", "x"],
        description="f(x) = √x",
        ood=True,
    ),
]

TRAIN_TARGETS = [t for t in ALL_TARGETS if not t.ood]
OOD_TARGETS   = [t for t in ALL_TARGETS if t.ood]
