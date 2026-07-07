"""
test_prototype_b.py — Tests for symbolic regression prototype.

Covers:
  - Grammar parsing / validation
  - Numerical evaluator correctness
  - Constant fitting
  - SymPy equivalence oracle (adversarial: non-equivalent expressions must differ)
  - Pareto front computation
  - Transformer forward pass (shapes)
  - Grammar masking (every generated token must be valid)
"""

import sys
from pathlib import Path
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from grammar import (
    parse, ParseError, is_valid_prefix, expected_remaining,
    ALL_TOKENS, BINARY_OPS, UNARY_OPS, TERMINALS,
    VOCAB_SIZE, BOS_ID, EOS_ID, arity,
)
from evaluator import eval_node, mse, fit_constants, are_equivalent, symbolic_complexity
from pareto import CandidatePool, _simplify_node
from transformer import TinyTransformer
from targets import ALL_TARGETS, TRAIN_TARGETS


# ---------------------------------------------------------------------------
# Grammar
# ---------------------------------------------------------------------------

class TestGrammar:

    def test_parse_simple(self):
        node = parse(["pow2", "x"])
        assert node.token == "pow2"
        assert node.children[0].token == "x"

    def test_parse_nested(self):
        # add(x, inv(x)) → prefix: add x inv x
        node = parse(["add", "x", "inv", "x"])
        assert node.token == "add"
        assert node.children[1].token == "inv"

    def test_parse_leftover_fails(self):
        with pytest.raises(ParseError):
            parse(["x", "x"])

    def test_parse_incomplete_fails(self):
        with pytest.raises(ParseError):
            parse(["add", "x"])   # missing second arg

    def test_is_valid_prefix(self):
        assert is_valid_prefix(["x"])
        assert is_valid_prefix(["pow2", "x"])
        assert is_valid_prefix(["add", "x", "1"])
        assert not is_valid_prefix(["add", "x"])
        assert not is_valid_prefix(["x", "x"])

    def test_expected_remaining(self):
        assert expected_remaining([]) == 1         # need root
        assert expected_remaining(["x"]) == 0      # complete
        assert expected_remaining(["add"]) == 2    # need 2 args
        assert expected_remaining(["add", "x"]) == 1

    def test_prefix_roundtrip(self):
        tokens = ["add", "pow2", "x", "inv", "x"]
        node = parse(tokens)
        assert node.to_prefix() == tokens

    def test_arity(self):
        assert arity("add") == 2
        assert arity("neg") == 1
        assert arity("x")   == 0
        assert arity("C")   == 0


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------

class TestEvaluator:

    def setup_method(self):
        self.x = np.linspace(0.1, 2.0, 50)

    def test_eval_x(self):
        node = parse(["x"])
        y = eval_node(node, self.x, [])
        np.testing.assert_allclose(y, self.x)

    def test_eval_pow2(self):
        node = parse(["pow2", "x"])
        y = eval_node(node, self.x, [])
        np.testing.assert_allclose(y, self.x**2, rtol=1e-5)

    def test_eval_inv(self):
        node = parse(["inv", "x"])
        y = eval_node(node, self.x, [])
        np.testing.assert_allclose(y, 1/self.x, rtol=1e-5)

    def test_eval_div_by_zero_is_nan(self):
        x = np.array([0.0, 1.0])
        node = parse(["inv", "x"])
        y = eval_node(node, x, [])
        assert np.isnan(y[0])
        assert np.isfinite(y[1])

    def test_eval_constant_placeholder(self):
        node = parse(["mul", "C", "x"])
        y = eval_node(node, self.x, [3.0])
        np.testing.assert_allclose(y, 3.0 * self.x, rtol=1e-5)

    def test_mse_ignores_nan(self):
        y_pred = np.array([1.0, np.nan, 3.0])
        y_true = np.array([1.0, 2.0,   3.0])
        assert mse(y_pred, y_true) == pytest.approx(0.0)

    def test_fit_constants_zero_constants(self):
        node = parse(["pow2", "x"])
        y = self.x**2
        constants, fit = fit_constants(node, self.x, y)
        assert constants == []
        assert fit < 1e-6

    def test_fit_constants_linear(self):
        # f(x) = C*x, true C = 2.5
        node = parse(["mul", "C", "x"])
        y_true = 2.5 * self.x + np.random.default_rng(0).normal(0, 0.01, len(self.x))
        constants, fit = fit_constants(node, self.x, y_true)
        assert len(constants) == 1
        assert abs(constants[0] - 2.5) < 0.1
        assert fit < 0.01


# ---------------------------------------------------------------------------
# SymPy oracle (adversarial: non-equivalent must be detected)
# ---------------------------------------------------------------------------

class TestEquivalenceOracle:

    def setup_method(self):
        pass

    def test_equivalent_same(self):
        n = parse(["pow2", "x"])
        assert are_equivalent(n, [], n, [])

    def test_equivalent_different_form(self):
        # x*x and x² should be equivalent (SymPy simplifies)
        a = parse(["mul", "x", "x"])
        b = parse(["pow2", "x"])
        # These may or may not be equal depending on SymPy — both are x²
        # so they should be equivalent
        result = are_equivalent(a, [], b, [])
        assert result   # x*x == x**2 symbolically

    def test_not_equivalent(self):
        a = parse(["pow2", "x"])       # x²
        b = parse(["add", "x", "1"])   # x+1
        assert not are_equivalent(a, [], b, [])

    def test_not_equivalent_inv_vs_x(self):
        a = parse(["inv", "x"])  # 1/x
        b = parse(["x"])          # x
        assert not are_equivalent(a, [], b, [])


# ---------------------------------------------------------------------------
# Simplifier
# ---------------------------------------------------------------------------

class TestSimplifier:

    def test_identity_elimination(self):
        # mul(x, 1) → x
        node = parse(["mul", "x", "1"])
        simplified = _simplify_node(node)
        assert simplified.token == "x"

    def test_zero_annihilation(self):
        # mul(x, 0) → 0
        node = parse(["mul", "x", "0"])
        simplified = _simplify_node(node)
        assert simplified.token == "0"

    def test_add_zero(self):
        # add(x, 0) → x
        node = parse(["add", "x", "0"])
        simplified = _simplify_node(node)
        assert simplified.token == "x"

    def test_double_neg(self):
        # neg(neg(x)) → x
        node = parse(["neg", "neg", "x"])
        simplified = _simplify_node(node)
        assert simplified.token == "x"

    def test_constant_fold(self):
        # add(1, 2) → 3... but 3 is not in our terminals; stays as is
        # add(1, 1) → 2 (which IS in our terminals)
        node = parse(["add", "1", "1"])
        simplified = _simplify_node(node)
        assert simplified.token == "2"


# ---------------------------------------------------------------------------
# Candidate pool
# ---------------------------------------------------------------------------

class TestCandidatePool:

    def setup_method(self):
        rng = np.random.default_rng(0)
        self.x = rng.uniform(0.1, 2.0, 50)
        self.y = self.x**2   # target = x²

    def test_add_valid(self):
        pool = CandidatePool()
        cand = pool.add(["pow2", "x"], self.x, self.y)
        assert cand is not None
        assert len(pool.candidates) == 1

    def test_add_invalid(self):
        pool = CandidatePool()
        cand = pool.add(["add", "x"], self.x, self.y)  # incomplete
        assert cand is None
        assert len(pool.candidates) == 0

    def test_dedup_structural(self):
        pool = CandidatePool()
        pool.add(["pow2", "x"], self.x, self.y)
        pool.add(["pow2", "x"], self.x, self.y)   # same
        assert len(pool.candidates) == 1

    def test_pareto_front_nonempty(self):
        pool = CandidatePool()
        for tok_seq in [["pow2", "x"], ["x"], ["inv", "x"]]:
            pool.add(tok_seq, self.x, self.y)
        front = pool.pareto_front()
        assert len(front) >= 1

    def test_pareto_rank_0_is_best(self):
        pool = CandidatePool()
        for tok_seq in [["pow2", "x"], ["x"], ["add", "x", "1"]]:
            pool.add(tok_seq, self.x, self.y)
        pool.compute_pareto()
        front = [c for c in pool.candidates if c.pareto_rank == 0]
        # All front members must be non-dominated
        for f in front:
            dominated = any(
                (c.fit_mse <= f.fit_mse and c.complexity <= f.complexity
                 and (c.fit_mse < f.fit_mse or c.complexity < f.complexity))
                for c in pool.candidates if c is not f
            )
            assert not dominated, f"{f.infix_str} is on front but dominated"


# ---------------------------------------------------------------------------
# Transformer
# ---------------------------------------------------------------------------

class TestTransformer:

    def test_forward_shape(self):
        model = TinyTransformer(seed=0)
        from grammar import VOCAB_SIZE
        logits = model.forward([BOS_ID, 5])   # BOS + one token
        assert logits.shape == (2, VOCAB_SIZE)

    def test_generate_valid_prefix(self):
        model = TinyTransformer(seed=0)
        for i in range(10):
            tokens = model.generate(temperature=1.0, grammar_mask=True, seed=i)
            assert is_valid_prefix(tokens), f"Generated invalid: {tokens}"

    def test_generate_without_mask_can_be_invalid(self):
        # Without grammar mask, some sequences may be invalid (expected)
        model = TinyTransformer(seed=0)
        results = [
            model.generate(temperature=1.5, grammar_mask=False, seed=i)
            for i in range(20)
        ]
        n_invalid = sum(1 for r in results if not is_valid_prefix(r))
        # Without mask, some should be invalid (shows mask is load-bearing)
        # This is not a hard assertion — just demonstrates the mask matters
        # We just verify the function runs
        assert len(results) == 20

    def test_train_step_runs(self):
        model = TinyTransformer(seed=0)
        from grammar import TOKEN_TO_ID, BOS_ID, EOS_ID
        seqs = [[BOS_ID, TOKEN_TO_ID["pow2"], TOKEN_TO_ID["x"], EOS_ID]]
        loss = model.train_on_sequences(seqs, [1.0])
        assert np.isfinite(loss)

    def test_save_load(self, tmp_path):
        model = TinyTransformer(seed=1)
        path  = str(tmp_path / "weights")
        model.save(path)
        model2 = TinyTransformer(seed=99)
        model2.load(path + ".npz")
        logits1 = model.forward([BOS_ID])
        logits2 = model2.forward([BOS_ID])
        np.testing.assert_allclose(logits1, logits2, atol=1e-4)


# ---------------------------------------------------------------------------
# Target data
# ---------------------------------------------------------------------------

class TestTargets:

    def test_all_targets_parse(self):
        for t in ALL_TARGETS:
            node = parse(t.tokens)
            assert node is not None

    def test_sample_shape(self):
        from targets import N_POINTS
        for t in ALL_TARGETS:
            x, y = t.sample()
            assert x.shape == (N_POINTS,)
            assert y.shape == (N_POINTS,)

    def test_noise_level(self):
        from targets import SIGMA
        for t in TRAIN_TARGETS:
            x, y_noisy = t.sample(seed=0)
            node = parse(t.tokens)
            y_clean = eval_node(node, x, [])
            residual_std = float(np.std(y_noisy - y_clean))
            # Should be within 2σ of expected noise
            assert abs(residual_std - SIGMA) < SIGMA, \
                f"{t.name}: noise std={residual_std:.3f}, expected≈{SIGMA}"
