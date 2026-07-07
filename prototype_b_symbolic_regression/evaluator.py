"""
evaluator.py — Evaluate a Node tree numerically and symbolically.

Numerical evaluation
--------------------
  eval_node(node, x_vals, constants) → np.ndarray
  Evaluates the expression at each x value using the given constant vector.
  Returns NaN/Inf for undefined operations (div by zero, sqrt of negative, etc.)
  instead of raising — so that the optimiser can penalise these.

Constant fitting
----------------
  fit_constants(node, x_vals, y_vals) → (constants, mse)
  Fit the C-placeholders in the expression to data by least-squares
  (scipy.optimize.minimize with L-BFGS-B, warm-started at 1.0).
  Returns the fitted constants and the achieved MSE.

Symbolic interface (SymPy oracle — arm's-length use only)
---------------------------------------------------------
  to_sympy(node, constants) → sympy expression
  are_equivalent(node_a, node_b, constants_a, constants_b) → bool
    Uses SymPy simplify to check symbolic equality — this is the ONLY
    place SymPy is called.  All search / selection logic is our code.
  symbolic_complexity(node) → int
    Node count in the simplified SymPy expression tree — used as the
    complexity axis of the Pareto front.
"""

from __future__ import annotations
import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import minimize

from grammar import Node, BINARY_OPS, UNARY_OPS

# ---------------------------------------------------------------------------
# Numerical evaluation
# ---------------------------------------------------------------------------

def eval_node(
    node: Node,
    x_vals: np.ndarray,
    constants: List[float],
    _c_idx: Optional[List[int]] = None,
) -> np.ndarray:
    """
    Evaluate the expression tree at all x values.

    constants: ordered list of values for C-placeholders (one per 'C' in tree,
               left-to-right / depth-first order).
    Returns float array of shape (len(x_vals),).
    """
    # Use a mutable counter to consume constants left-to-right
    c_counter = [0]

    def _eval(n: Node) -> np.ndarray:
        t = n.token
        if t == "x":    return x_vals.astype(float)
        if t == "C":
            val = constants[c_counter[0]] if c_counter[0] < len(constants) else 1.0
            c_counter[0] += 1
            return np.full_like(x_vals, val, dtype=float)
        if t == "0":    return np.zeros_like(x_vals, dtype=float)
        if t == "1":    return np.ones_like(x_vals, dtype=float)
        if t == "2":    return np.full_like(x_vals, 2.0, dtype=float)
        if t == "-1":   return np.full_like(x_vals, -1.0, dtype=float)

        ch = [_eval(c) for c in n.children]

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if t == "add":  return ch[0] + ch[1]
            if t == "mul":  return ch[0] * ch[1]
            if t == "sub":  return ch[0] - ch[1]
            if t == "div":
                denom = ch[1].copy()
                denom[np.abs(denom) < 1e-8] = np.nan
                return ch[0] / denom
            if t == "pow2": return ch[0] ** 2
            if t == "sin":  return np.sin(ch[0])
            if t == "exp":
                arg = np.clip(ch[0], -50, 50)  # avoid overflow
                return np.exp(arg)
            if t == "sqrt":
                arg = ch[0].copy()
                arg[arg < 0] = np.nan
                return np.sqrt(arg)
            if t == "inv":
                denom = ch[0].copy()
                denom[np.abs(denom) < 1e-8] = np.nan
                return 1.0 / denom
            if t == "neg":  return -ch[0]

        raise ValueError(f"Unknown token '{t}'")

    result = _eval(node)
    return np.where(np.isfinite(result), result, np.nan)


def mse(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """MSE ignoring NaN positions."""
    mask = np.isfinite(y_pred) & np.isfinite(y_true)
    if mask.sum() < 2:
        return np.inf
    return float(np.mean((y_pred[mask] - y_true[mask]) ** 2))


# ---------------------------------------------------------------------------
# Constant fitting
# ---------------------------------------------------------------------------

def fit_constants(
    node: Node,
    x_vals: np.ndarray,
    y_vals: np.ndarray,
    n_restarts: int = 2,
) -> Tuple[List[float], float]:
    """
    Fit C-placeholders in node to data (x_vals, y_vals).

    Strategy (fast path first):
      0 constants → direct evaluation
      1 constant  → coarse grid search then Brent refinement (fast, no NaN issues)
      2+ constants → rejected upstream (too slow; callers should skip these)

    Returns (fitted_constants, mse).
    """
    n_c = node.n_constants()
    if n_c == 0:
        y_pred = eval_node(node, x_vals, [])
        return [], mse(y_pred, y_vals)

    if n_c >= 2:
        # Skip expensive multi-constant fitting in the search loop
        return [1.0] * n_c, np.inf

    # --- Single-constant case: grid search + Brent's method ---
    # Grid over a wide range
    grid = np.concatenate([
        np.linspace(-5, 5, 40),
        np.array([-10, -0.5, -0.1, 0.1, 0.5, 10, 100]),
    ])
    best_c = 1.0
    best_mse_val = np.inf

    for c_val in grid:
        y_pred = eval_node(node, x_vals, [float(c_val)])
        m = mse(y_pred, y_vals)
        if m < best_mse_val:
            best_mse_val = m
            best_c = float(c_val)

    # Refine with scipy minimize_scalar around best grid point
    import warnings as _warnings
    try:
        with _warnings.catch_warnings():
            _warnings.simplefilter("ignore")
            res = minimize(
                lambda c: mse(eval_node(node, x_vals, [float(c[0])]), y_vals),
                x0=np.array([best_c]),
                method="L-BFGS-B",
                options={"maxiter": 30, "ftol": 1e-8},
            )
        if np.isfinite(res.fun) and res.fun < best_mse_val:
            best_mse_val = float(res.fun)
            best_c = float(res.x[0])
    except Exception:
        pass

    return [best_c], best_mse_val


# ---------------------------------------------------------------------------
# SymPy oracle (arm's-length — search / selection never calls this directly)
# ---------------------------------------------------------------------------

def to_sympy(node: Node, constants: List[float]):
    """Convert Node + fitted constants to a SymPy expression."""
    import sympy as sp
    x = sp.Symbol("x")
    c_counter = [0]

    def _conv(n: Node):
        t = n.token
        if t == "x":  return x
        if t == "C":
            val = constants[c_counter[0]] if c_counter[0] < len(constants) else 1.0
            c_counter[0] += 1
            return sp.Float(round(val, 6))
        if t == "0":  return sp.Integer(0)
        if t == "1":  return sp.Integer(1)
        if t == "2":  return sp.Integer(2)
        if t == "-1": return sp.Integer(-1)

        ch = [_conv(c) for c in n.children]
        if t == "add":  return ch[0] + ch[1]
        if t == "mul":  return ch[0] * ch[1]
        if t == "sub":  return ch[0] - ch[1]
        if t == "div":  return ch[0] / ch[1]
        if t == "pow2": return ch[0] ** 2
        if t == "sin":  return sp.sin(ch[0])
        if t == "exp":  return sp.exp(ch[0])
        if t == "sqrt": return sp.sqrt(ch[0])
        if t == "inv":  return sp.Integer(1) / ch[0]
        if t == "neg":  return -ch[0]
        raise ValueError(t)

    return _conv(node)


def are_equivalent(
    node_a: Node, constants_a: List[float],
    node_b: Node, constants_b: List[float],
) -> bool:
    """
    Check symbolic equivalence using SymPy simplify as oracle.
    Returns True if simplify(A - B) == 0.
    This is the ONLY SymPy call in the pipeline — all selection logic is ours.
    """
    try:
        import sympy as sp
        expr_a = to_sympy(node_a, constants_a)
        expr_b = to_sympy(node_b, constants_b)
        diff   = sp.simplify(expr_a - expr_b)
        return diff == 0
    except Exception:
        return False


def symbolic_complexity(node: Node, constants: List[float]) -> int:
    """
    Complexity = number of nodes in the SymPy-simplified expression tree.
    Used as the 'C' axis in Pareto(MSE, complexity).
    Falls back to node.size() if SymPy fails.
    """
    try:
        import sympy as sp
        expr   = to_sympy(node, constants)
        simple = sp.simplify(expr)
        return sp.count_ops(simple) + 1   # +1 so constants are never 0-complexity
    except Exception:
        return node.size()
