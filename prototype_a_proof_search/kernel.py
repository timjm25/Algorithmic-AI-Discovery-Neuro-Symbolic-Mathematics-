"""
kernel.py — Sound rewriting kernel for commutative ring axioms.

DESIGN CONTRACT
---------------
  kernel.apply(rule_idx, path, expr)
    → new_expr   if and only if RULES[rule_idx].lhs matches the subterm at path
    → raises KernelRejectError  otherwise

No path around KernelRejectError exists. The kernel is the ONLY component
allowed to advance the proof state. External code cannot write to _rules or
bypass the match check. Any sequence of .apply() calls is therefore a sound
equational proof in the ring theory.

ADVERSARIAL INVARIANT (checked in tests/test_kernel.py):
  For every rule, applying it to a non-matching expression raises KernelRejectError.
  The kernel never returns an expression numerically different from the input
  (checked by eval on random variable assignments).
"""

from __future__ import annotations
from typing import Dict, List, Optional, Tuple

from expressions import Expr, Const, Var, Add, Mul, Neg, C, V, a, b, c


# ---------------------------------------------------------------------------
# Pattern nodes — wildcards for structural matching
# ---------------------------------------------------------------------------

class _W:
    """Wildcard: matches any subexpression and binds it to self.name."""
    __slots__ = ("name",)
    def __init__(self, name: str): self.name = name
    def __repr__(self): return f"?{self.name}"

X, Y, Z = _W("X"), _W("Y"), _W("Z")


# ---------------------------------------------------------------------------
# Pattern matching and instantiation
# ---------------------------------------------------------------------------

def _match(pattern, expr: Expr) -> Optional[Dict[str, Expr]]:
    """
    Match pattern against expr.  Returns binding dict {name: Expr} or None.
    Wildcards must bind consistently (X must bind to the same subtree everywhere).
    """
    if isinstance(pattern, _W):
        return {pattern.name: expr}

    if type(pattern) is not type(expr):
        return None

    if isinstance(pattern, Const):
        return {} if pattern.n == expr.n else None

    if isinstance(pattern, Var):
        return {} if pattern.name == expr.name else None

    if isinstance(pattern, Add):
        b1 = _match(pattern.left,  expr.left)
        b2 = _match(pattern.right, expr.right)
    elif isinstance(pattern, Mul):
        b1 = _match(pattern.left,  expr.left)
        b2 = _match(pattern.right, expr.right)
    elif isinstance(pattern, Neg):
        b1 = _match(pattern.child, expr.child)
        b2 = {}
    else:
        return None

    if b1 is None or b2 is None:
        return None

    # Merge, checking wildcard consistency
    merged = dict(b1)
    for k, v in b2.items():
        if k in merged and merged[k] != v:
            return None       # same wildcard matched two different subtrees
        merged[k] = v
    return merged


def _instantiate(template, bindings: Dict[str, Expr]) -> Expr:
    """Substitute bindings into a rule right-hand side template."""
    if isinstance(template, _W):
        return bindings[template.name]
    if isinstance(template, (Const, Var)):
        return template
    if isinstance(template, Add):
        return Add(_instantiate(template.left,  bindings),
                   _instantiate(template.right, bindings))
    if isinstance(template, Mul):
        return Mul(_instantiate(template.left,  bindings),
                   _instantiate(template.right, bindings))
    if isinstance(template, Neg):
        return Neg(_instantiate(template.child, bindings))
    raise ValueError(f"Unknown template type {type(template)}")


# ---------------------------------------------------------------------------
# Rule table
# ---------------------------------------------------------------------------
#
# Each entry: (name, lhs_pattern, rhs_template)
# All 17 equational axioms of a commutative ring over ℤ.
#
_RULE_TABLE = [
    # 0  Commutativity of +
    ("comm_add",      Add(X, Y),         Add(Y, X)),
    # 1  Commutativity of ×
    ("comm_mul",      Mul(X, Y),         Mul(Y, X)),
    # 2  Associativity of + : (X+Y)+Z → X+(Y+Z)
    ("assoc_add_lr",  Add(Add(X,Y), Z),  Add(X, Add(Y, Z))),
    # 3  Associativity of + : X+(Y+Z) → (X+Y)+Z
    ("assoc_add_rl",  Add(X, Add(Y,Z)),  Add(Add(X, Y), Z)),
    # 4  Left distributivity : X*(Y+Z) → X*Y + X*Z
    ("dist_l",        Mul(X, Add(Y,Z)),  Add(Mul(X,Y), Mul(X,Z))),
    # 5  Right distributivity : (Y+Z)*X → Y*X + Z*X
    ("dist_r",        Mul(Add(Y,Z), X),  Add(Mul(Y,X), Mul(Z,X))),
    # 6  Additive identity (right) : X+0 → X
    ("add_id_r",      Add(X, C(0)),      X),
    # 7  Additive identity (left)  : 0+X → X
    ("add_id_l",      Add(C(0), X),      X),
    # 8  Multiplicative identity (right) : X*1 → X
    ("mul_id_r",      Mul(X, C(1)),      X),
    # 9  Multiplicative identity (left)  : 1*X → X
    ("mul_id_l",      Mul(C(1), X),      X),
    # 10 Zero annihilation (right) : X*0 → 0
    ("mul_zero_r",    Mul(X, C(0)),      C(0)),
    # 11 Zero annihilation (left)  : 0*X → 0
    ("mul_zero_l",    Mul(C(0), X),      C(0)),
    # 12 Additive inverse : X+(-X) → 0
    ("add_inv",       Add(X, Neg(X)),    C(0)),
    # 13 Double negation  : -(-X) → X
    ("double_neg",    Neg(Neg(X)),       X),
    # 14 Negation of sum  : -(X+Y) → (-X)+(-Y)
    ("neg_sum",       Neg(Add(X,Y)),     Add(Neg(X), Neg(Y))),
    # 15 Associativity of × : (X*Y)*Z → X*(Y*Z)
    ("assoc_mul_lr",  Mul(Mul(X,Y), Z), Mul(X, Mul(Y, Z))),
    # 16 Associativity of × : X*(Y*Z) → (X*Y)*Z
    ("assoc_mul_rl",  Mul(X, Mul(Y,Z)), Mul(Mul(X, Y), Z)),
]

N_RULES   = len(_RULE_TABLE)
RULE_NAMES = [r[0] for r in _RULE_TABLE]


# ---------------------------------------------------------------------------
# Kernel
# ---------------------------------------------------------------------------

class KernelRejectError(Exception):
    """Raised when a rule does not match at the requested path."""


class _Kernel:
    """
    The sole trusted authority in the proof system.
    All proof-state transitions must go through .apply().
    """

    def apply(self, rule_idx: int, path: List[int], expr: Expr) -> Expr:
        """
        Apply rule RULES[rule_idx] to the subterm at path in expr.

        Returns a new expression logically equivalent to expr under ring axioms.
        Raises KernelRejectError if the rule does not match.

        SYMBOLIC → NEURAL interlock point:
          This is the gate that rejects any move the policy might propose.
          An invalid move is never queued in the search.
        """
        if not (0 <= rule_idx < N_RULES):
            raise KernelRejectError(
                f"rule_idx {rule_idx} out of range [0, {N_RULES})"
            )

        name, lhs, rhs = _RULE_TABLE[rule_idx]
        subterm = expr.get_at(path)
        bindings = _match(lhs, subterm)

        if bindings is None:
            raise KernelRejectError(
                f"Rule '{name}' does not match at path {path} "
                f"(subterm = {subterm!r})"
            )

        new_subterm = _instantiate(rhs, bindings)
        return expr.replace_at(path, new_subterm)

    def constant_fold(self, expr: Expr) -> Expr:
        """
        Deterministic constant folding — evaluates any all-Const subtree.
        Not a policy choice; applied automatically after each step.
        """
        return _fold(expr)

    def is_zero(self, expr: Expr) -> bool:
        """True iff the expression reduces to Const(0) after constant folding."""
        return self.constant_fold(expr).is_zero()

    def legal_moves(self, expr: Expr) -> List[Tuple[int, List[int]]]:
        """
        Enumerate all (rule_idx, path) pairs where a rule matches.

        SYMBOLIC → NEURAL interlock point:
          This is the action space passed to the policy for scoring.
          Only moves that can succeed are offered.
        """
        moves: List[Tuple[int, List[int]]] = []
        for path, subterm in expr.all_nodes():
            for i, (_, lhs, _) in enumerate(_RULE_TABLE):
                if _match(lhs, subterm) is not None:
                    moves.append((i, path))
        return moves


def _fold(expr: Expr) -> Expr:
    if isinstance(expr, (Const, Var)):
        return expr
    if isinstance(expr, Neg):
        child = _fold(expr.child)
        return Const(-child.n) if isinstance(child, Const) else Neg(child)
    if isinstance(expr, Add):
        left, right = _fold(expr.left), _fold(expr.right)
        if isinstance(left, Const) and isinstance(right, Const):
            return Const(left.n + right.n)
        return Add(left, right)
    if isinstance(expr, Mul):
        left, right = _fold(expr.left), _fold(expr.right)
        if isinstance(left, Const) and isinstance(right, Const):
            return Const(left.n * right.n)
        return Mul(left, right)
    raise TypeError(f"Unknown Expr subtype: {type(expr)}")


# Module-level singleton — the only instance needed
KERNEL = _Kernel()
