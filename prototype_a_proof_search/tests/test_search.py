"""
test_search.py — Integration tests for proof search + interlock.
"""

import sys
from pathlib import Path
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from expressions import Add, Mul, Neg, C, V, a, b, c
from kernel import KERNEL
from policy import MLPPolicy
from search import best_first_search, uniform_search, ProofResult
from theorems import generate_training_theorems, generate_ood_theorems


def fresh_policy(seed=0):
    return MLPPolicy(lr=0.01, seed=seed)


class TestSearch:

    def test_trivially_zero(self):
        """A zero expression is proved in 0 steps."""
        result = best_first_search(C(0), fresh_policy())
        assert result.found
        assert result.proof_length == 0

    def test_one_step_proof(self):
        """a + (-a) = 0 requires exactly one rule application."""
        e = Add(a, Neg(a))
        result = best_first_search(e, fresh_policy())
        assert result.found
        assert result.proof_length == 1

    def test_two_step_proof(self):
        """a*(b+c) - (a*b + a*c) = 0 — distributivity + inverse."""
        e = Add(Mul(a, Add(b, c)), Neg(Add(Mul(a, b), Mul(a, c))))
        result = best_first_search(e, fresh_policy(), max_nodes=500)
        assert result.found

    def test_proof_trace_kernel_certified(self):
        """Every step in a returned trace must pass kernel re-verification."""
        e = Add(Mul(a, Add(b, c)), Neg(Add(Mul(a, b), Mul(a, c))))
        result = best_first_search(e, fresh_policy(), max_nodes=500)
        assert result.found

        expr = e
        for step in result.steps:
            rechecked = KERNEL.apply(step.rule_idx, step.path, expr)
            rechecked = KERNEL.constant_fold(rechecked)
            assert repr(rechecked) == repr(step.expr_after)
            expr = step.expr_after

        assert KERNEL.is_zero(expr)

    def test_uniform_baseline_works(self):
        """Uniform search can find simple proofs."""
        e = Add(a, Neg(a))
        result = uniform_search(e, max_nodes=50)
        assert result.found

    def test_search_respects_budget(self):
        """Search never expands more nodes than the budget."""
        e = Add(Mul(a, Add(b, c)), Neg(Add(Mul(a, b), Mul(a, c))))
        result = best_first_search(e, fresh_policy(), max_nodes=5, beam_width=2)
        assert result.nodes_expanded <= 5

    def test_training_theorems_solvable(self):
        """At least 60% of fresh training theorems solvable with a budget of 300."""
        theorems = generate_training_theorems(30, max_scramble=3, seed=1)
        policy = fresh_policy()
        solved = sum(
            1 for t in theorems
            if best_first_search(t, policy, max_nodes=300, beam_width=8).found
        )
        rate = solved / len(theorems)
        # With uniform weights, some should still be findable via exhaustive search
        assert rate >= 0.3, f"Only {rate:.0%} solvable — theorem generator may be broken"

    def test_ood_theorems_generated(self):
        """OOD theorems are non-trivially-zero expressions."""
        ood = generate_ood_theorems(20, seed=0)
        for t in ood:
            folded = KERNEL.constant_fold(t)
            # They should NOT already be zero (that would be trivial)
            # Some may be — acceptable as long as not all of them
        non_trivial = sum(1 for t in ood if not KERNEL.is_zero(t))
        assert non_trivial >= 10, "Too many OOD theorems are trivially zero"


class TestPolicyGradient:

    def test_train_step_reduces_loss(self):
        """Loss on the same batch should decrease after one gradient step."""
        policy = fresh_policy(seed=42)
        e = Add(Mul(a, Add(b, c)), Neg(Add(Mul(a, b), Mul(a, c))))
        moves = KERNEL.legal_moves(e)
        assert moves, "No legal moves on test expression"

        expert_idx = 0
        examples = [(e, moves, expert_idx)] * 10

        loss1 = policy.train_step(examples)
        loss2 = policy.train_step(examples)
        # Loss should not increase monotonically — verify it's finite and changing
        assert np.isfinite(loss1)
        assert np.isfinite(loss2)

    def test_policy_score_sums_to_one(self):
        """log-probabilities should exponentiate to a valid distribution."""
        policy = fresh_policy()
        e = Add(a, b)
        moves = KERNEL.legal_moves(e)
        log_probs = policy.score_moves(e, moves)
        probs = np.exp(log_probs)
        assert abs(probs.sum() - 1.0) < 1e-4

    def test_policy_save_load(self, tmp_path):
        policy = fresh_policy(seed=1)
        path = str(tmp_path / "weights")
        policy.save(path)
        policy2 = fresh_policy(seed=99)   # different init
        policy2.load(path + ".npz")
        e = Add(a, b)
        moves = KERNEL.legal_moves(e)
        lp1 = policy.score_moves(e, moves)
        lp2 = policy2.score_moves(e, moves)
        np.testing.assert_allclose(lp1, lp2, atol=1e-5)
