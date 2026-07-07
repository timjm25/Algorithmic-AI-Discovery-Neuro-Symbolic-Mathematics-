"""
theorems.py — Theorem generator for the proof-search prototype.

A "theorem" here is an expression E such that E = 0 is provable using the
ring axioms in kernel.py.  We generate theorems by:

  1. Start with a known-zero expression (trivially = 0 by one or two rules).
  2. Apply k random rewrites using the kernel (any legal move is valid
     because all rules are equalities).
  3. The resulting scrambled expression is still = 0, but now requires
     k+ steps to prove.

Two families:
  Training family   — distributivity, commutativity, identity templates.
  OOD family        — associativity-of-mul, mixed identity chains, and
                      right-distributivity templates (same rules, but
                      structures the policy never saw in training).
"""

from __future__ import annotations

import random
from typing import List

from expressions import Expr, Const, Add, Mul, Neg, C, V, a, b, c
from kernel import KERNEL


# ---------------------------------------------------------------------------
# Seed expressions (provably = 0 with 1–2 rule applications)
# ---------------------------------------------------------------------------

def _training_seeds() -> List[Expr]:
    """Expressions used to generate training theorems."""
    return [
        # Left distributivity identity: a*(b+c) - a*b - a*c = 0
        Add(Mul(a, Add(b, c)), Neg(Add(Mul(a, b), Mul(a, c)))),
        # Commutativity of +: (a+b) - (b+a) = 0
        Add(Add(a, b), Neg(Add(b, a))),
        # Commutativity of *: a*b - b*a = 0
        Add(Mul(a, b), Neg(Mul(b, a))),
        # Additive identity: (a+0) - a = 0
        Add(Add(a, C(0)), Neg(a)),
        # Multiplicative identity: a*1 - a = 0
        Add(Mul(a, C(1)), Neg(a)),
        # Left zero: 1*a - a = 0
        Add(Mul(C(1), a), Neg(a)),
        # Additive inverse: a + (-a) = 0
        Add(a, Neg(a)),
        # Double negation: -(-a) - a = 0
        Add(Neg(Neg(a)), Neg(a)),
        # Negation of sum: -(a+b) + a + b = 0
        Add(Add(Neg(Add(a, b)), a), b),
        # Zero annihilation: a*0 = 0
        Mul(a, C(0)),
    ]


def _ood_seeds() -> List[Expr]:
    """
    OOD seeds — same rules, but different structural templates.
    The policy was never trained on theorems derived from these.
    """
    return [
        # Right distributivity: (a+b)*c - a*c - b*c = 0
        Add(Mul(Add(a, b), c), Neg(Add(Mul(a, c), Mul(b, c)))),
        # Associativity of *: (a*b)*c - a*(b*c) = 0
        Add(Mul(Mul(a, b), c), Neg(Mul(a, Mul(b, c)))),
        # Mixed identity chain: (a+0)*1 - a = 0
        Add(Mul(Add(a, C(0)), C(1)), Neg(a)),
        # Nested commutativity: a*(b+a) - a*(a+b) = 0
        Add(Mul(a, Add(b, a)), Neg(Mul(a, Add(a, b)))),
        # Assoc + with identity: ((a+0)+b) - (a+b) = 0
        Add(Add(Add(a, C(0)), b), Neg(Add(a, b))),
    ]


# ---------------------------------------------------------------------------
# Scrambler
# ---------------------------------------------------------------------------

def scramble(expr: Expr, rng: random.Random, n_steps: int) -> Expr:
    """
    Apply n_steps random kernel rewrites to expr.
    Every rewrite is certified by the kernel, so the result is still = 0.
    """
    current = expr
    for _ in range(n_steps):
        moves = KERNEL.legal_moves(current)
        if not moves:
            break
        rule_idx, path = rng.choice(moves)
        try:
            current = KERNEL.apply(rule_idx, path, current)
        except Exception:
            pass   # shouldn't happen; skip if it does
    return current


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_training_theorems(n: int, max_scramble: int = 4, seed: int = 0) -> List[Expr]:
    """
    Generate n training theorems.
    Scramble depth drawn uniformly from [2, max_scramble].
    """
    rng = random.Random(seed)
    seeds = _training_seeds()
    out: List[Expr] = []

    for _ in range(n):
        base = rng.choice(seeds)
        steps = rng.randint(2, max_scramble)
        theorem = scramble(base, rng, steps)
        # Ensure not trivially zero already
        if KERNEL.is_zero(theorem):
            theorem = scramble(base, rng, 2)
        out.append(theorem)

    return out


def generate_ood_theorems(n: int, max_scramble: int = 3, seed: int = 99) -> List[Expr]:
    """
    Generate n OOD theorems from a disjoint template family.
    max_scramble kept lower so they remain tractable for evaluation.
    """
    rng = random.Random(seed)
    seeds = _ood_seeds()
    out: List[Expr] = []

    for _ in range(n):
        base = rng.choice(seeds)
        steps = rng.randint(1, max_scramble)
        theorem = scramble(base, rng, steps)
        if KERNEL.is_zero(theorem):
            theorem = base
        out.append(theorem)

    return out
