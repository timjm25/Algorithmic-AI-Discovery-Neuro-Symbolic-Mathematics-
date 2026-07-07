"""
test_kernel.py — Correctness and adversarial tests for the symbolic kernel.

ADVERSARIAL INVARIANT:
  For every rule R and every expression E where R's pattern does NOT match,
  KERNEL.apply(R, ..., E) must raise KernelRejectError.

SOUNDNESS INVARIANT:
  For every successful KERNEL.apply() call, the returned expression must
  evaluate to the same integer as the input under all variable assignments.
  (Checked on a grid of assignments.)
"""

import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from expressions import Expr, Add, Mul, Neg, Const, C, V, a, b, c
from kernel import KERNEL, KernelRejectError, N_RULES, RULE_NAMES, _match, _RULE_TABLE


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

ENV_GRID = [
    {"a": v_a, "b": v_b, "c": v_c}
    for v_a in (-1, 0, 1, 2)
    for v_b in (-1, 0, 1, 2)
    for v_c in (-1, 0, 1, 2)
]


def check_sound(before: Expr, after: Expr) -> None:
    """Assert that before and after evaluate identically on all grid points."""
    for env in ENV_GRID:
        assert before.eval(env) == after.eval(env), (
            f"Soundness violation: {before!r} vs {after!r} on {env}"
        )


# ---------------------------------------------------------------------------
# Structural tests
# ---------------------------------------------------------------------------

class TestExpressions:
    def test_add_eval(self):
        assert Add(a, b).eval({"a": 3, "b": 5, "c": 0}) == 8

    def test_mul_eval(self):
        assert Mul(a, b).eval({"a": 3, "b": 4, "c": 0}) == 12

    def test_neg_eval(self):
        assert Neg(a).eval({"a": 7, "b": 0, "c": 0}) == -7

    def test_expr_equality(self):
        assert Add(a, b) == Add(V("a"), V("b"))
        assert Add(a, b) != Add(b, a)

    def test_get_at_root(self):
        e = Add(a, b)
        assert e.get_at([]) is e

    def test_get_at_child(self):
        e = Add(Mul(a, b), c)
        assert e.get_at([0]) == Mul(a, b)
        assert e.get_at([0, 0]) == a

    def test_replace_at(self):
        e = Add(a, b)
        e2 = e.replace_at([1], c)
        assert e2 == Add(a, c)

    def test_all_nodes_count(self):
        # Add(Mul(a,b), c) has 5 nodes: Add, Mul, a, b, c
        e = Add(Mul(a, b), c)
        nodes = e.all_nodes()
        assert len(nodes) == 5

    def test_size(self):
        assert Add(Mul(a, b), c).size() == 5
        assert C(0).size() == 1

    def test_depth(self):
        assert Add(Mul(a, b), c).depth() == 2
        assert C(0).depth() == 0


# ---------------------------------------------------------------------------
# Kernel correctness tests
# ---------------------------------------------------------------------------

class TestKernelCorrectness:

    def test_comm_add(self):
        e = Add(a, b)
        result = KERNEL.apply(0, [], e)
        assert result == Add(b, a)
        check_sound(e, result)

    def test_comm_mul(self):
        e = Mul(a, b)
        result = KERNEL.apply(1, [], e)
        assert result == Mul(b, a)
        check_sound(e, result)

    def test_dist_l(self):
        e = Mul(a, Add(b, c))
        result = KERNEL.apply(4, [], e)
        assert result == Add(Mul(a, b), Mul(a, c))
        check_sound(e, result)

    def test_dist_r(self):
        e = Mul(Add(a, b), c)
        result = KERNEL.apply(5, [], e)
        assert result == Add(Mul(a, c), Mul(b, c))
        check_sound(e, result)

    def test_add_id_r(self):
        e = Add(a, C(0))
        result = KERNEL.apply(6, [], e)
        assert result == a
        check_sound(e, result)

    def test_add_id_l(self):
        e = Add(C(0), b)
        result = KERNEL.apply(7, [], e)
        assert result == b
        check_sound(e, result)

    def test_mul_id_r(self):
        e = Mul(a, C(1))
        result = KERNEL.apply(8, [], e)
        assert result == a
        check_sound(e, result)

    def test_mul_zero_r(self):
        e = Mul(a, C(0))
        result = KERNEL.apply(10, [], e)
        assert result == C(0)
        check_sound(e, result)

    def test_add_inv(self):
        e = Add(a, Neg(a))
        result = KERNEL.apply(12, [], e)
        assert result == C(0)
        check_sound(e, result)

    def test_double_neg(self):
        e = Neg(Neg(a))
        result = KERNEL.apply(13, [], e)
        assert result == a
        check_sound(e, result)

    def test_neg_sum(self):
        e = Neg(Add(a, b))
        result = KERNEL.apply(14, [], e)
        assert result == Add(Neg(a), Neg(b))
        check_sound(e, result)

    def test_apply_at_subpath(self):
        # Apply comm_add inside a larger expression
        e = Mul(Add(a, b), c)
        result = KERNEL.apply(0, [0], e)  # swap a+b at path [0]
        assert result == Mul(Add(b, a), c)
        check_sound(e, result)

    def test_constant_fold(self):
        e = Add(C(3), C(4))
        assert KERNEL.constant_fold(e) == C(7)

    def test_is_zero(self):
        assert KERNEL.is_zero(C(0))
        assert KERNEL.is_zero(Add(C(2), C(-2)))
        assert not KERNEL.is_zero(a)


# ---------------------------------------------------------------------------
# ADVERSARIAL TESTS
# Core invariant: apply() on a non-matching expression MUST raise KernelRejectError.
# If the kernel lets any of these through, it can certify a false step.
# ---------------------------------------------------------------------------

class TestAdversarial:

    def test_wrong_rule_type(self):
        """comm_add (rule 0) must reject a Mul node."""
        e = Mul(a, b)
        with pytest.raises(KernelRejectError):
            KERNEL.apply(0, [], e)

    def test_wrong_rule_type_2(self):
        """dist_l (rule 4) must reject an Add root."""
        e = Add(a, b)
        with pytest.raises(KernelRejectError):
            KERNEL.apply(4, [], e)

    def test_add_inv_different_subtrees(self):
        """add_inv (rule 12) binds X consistently: a + (-b) must not match X+(-X)."""
        e = Add(a, Neg(b))
        with pytest.raises(KernelRejectError):
            KERNEL.apply(12, [], e)

    def test_double_neg_on_neg(self):
        """double_neg (rule 13) requires -(-X); plain -X should fail."""
        e = Neg(a)
        with pytest.raises(KernelRejectError):
            KERNEL.apply(13, [], e)

    def test_out_of_range_rule(self):
        """Rule indices outside [0, N_RULES) must always fail."""
        e = Add(a, b)
        with pytest.raises(KernelRejectError):
            KERNEL.apply(N_RULES, [], e)
        with pytest.raises(KernelRejectError):
            KERNEL.apply(-1, [], e)

    def test_all_rules_rejected_on_var(self):
        """A bare Var matches no rule pattern — every rule must reject it."""
        e = a
        for i in range(N_RULES):
            with pytest.raises(KernelRejectError):
                result = KERNEL.apply(i, [], e)
                pytest.fail(f"Rule {i} ({RULE_NAMES[i]}) accepted a bare Var → {result!r}")

    def test_soundness_on_all_rules(self):
        """
        For every rule that CAN apply to a test expression, the output
        must evaluate identically to the input on all ENV_GRID points.
        """
        test_exprs = [
            Add(a, b), Mul(a, b), Add(Add(a, b), c),
            Mul(a, Add(b, c)), Add(a, C(0)), Mul(a, C(1)),
            Add(a, Neg(a)), Neg(Neg(a)), Neg(Add(a, b)),
            Mul(Mul(a, b), c),
        ]
        for e in test_exprs:
            for path, _ in e.all_nodes():
                for i in range(N_RULES):
                    try:
                        result = KERNEL.apply(i, path, e)
                        check_sound(e, result)
                    except KernelRejectError:
                        pass   # rejection is fine


# ---------------------------------------------------------------------------
# Two-step certified proof tests
# ---------------------------------------------------------------------------

class TestProofs:

    def test_distributivity_proof(self):
        """a*(b+c) - (a*b + a*c)  →  0  in 2 steps."""
        e = Add(Mul(a, Add(b, c)), Neg(Add(Mul(a, b), Mul(a, c))))
        step1 = KERNEL.apply(4, [0], e)           # dist_l at left child
        step2 = KERNEL.apply(12, [], step1)        # add_inv at root
        step2 = KERNEL.constant_fold(step2)
        assert KERNEL.is_zero(step2)

    def test_commutativity_proof(self):
        """(a+b) - (b+a)  →  0."""
        e = Add(Add(a, b), Neg(Add(b, a)))
        step1 = KERNEL.apply(0, [0], e)            # comm_add at [0]
        step2 = KERNEL.apply(12, [], step1)         # add_inv at root
        assert KERNEL.is_zero(KERNEL.constant_fold(step2))

    def test_identity_proof(self):
        """(a + 0) - a  →  0."""
        e = Add(Add(a, C(0)), Neg(a))
        step1 = KERNEL.apply(6, [0], e)            # add_id_r at [0]
        step2 = KERNEL.apply(12, [], step1)         # add_inv at root
        assert KERNEL.is_zero(KERNEL.constant_fold(step2))
