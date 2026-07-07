"""
pareto.py — Candidate management, deduplication, and Pareto-front selection.

This module is the core SYMBOLIC → NEURAL feedback mechanism in Prototype B.

Pipeline
--------
  1. Transformer generates N candidate token sequences.
  2. Each candidate is parsed, simplified (from scratch), constants fitted.
  3. Two-stage deduplication — no SymPy in the hot path:
       a. Structural: repr(simplified node) — identical expression trees
       b. Numerical: 5-point probe signature — algebraically equivalent forms
          (e.g. mul(x,x) and pow2(x) produce identical probe values)
  4. Surviving candidates are placed on a Pareto front over (MSE, complexity).
  5. Pareto rank → advantage weight → gradient signal back to transformer.

SymPy is NOT called here. It is reserved for the final falsification check
in search.evaluate_recovery().

The Pareto front is the interlock:
  - Neural: transformer proposes diverse structures
  - Symbolic: fast dedup collapses redundancy; Pareto over
    (fit, complexity) selects which candidates improve the policy
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

from grammar import Node, parse, ParseError, is_valid_prefix
from evaluator import fit_constants, mse, eval_node, are_equivalent, symbolic_complexity


# ---------------------------------------------------------------------------
# Candidate
# ---------------------------------------------------------------------------

@dataclass
class Candidate:
    tokens: List[str]
    node: Node
    constants: List[float]
    fit_mse: float
    complexity: int
    pareto_rank: int = 0          # 0 = Pareto-optimal (front 1)
    advantage: float = 0.0        # training weight (Symbolic → Neural signal)

    @property
    def prefix_str(self) -> str:
        return " ".join(self.tokens)

    @property
    def infix_str(self) -> str:
        return repr(self.node)


# ---------------------------------------------------------------------------
# From-scratch simplification (not SymPy — SymPy is only the equivalence oracle)
# ---------------------------------------------------------------------------

def _simplify_node(node: Node) -> Node:
    """
    Lightweight structural simplification:
      - Constant-fold integer-only sub-expressions
      - x * 0 → 0,  x + 0 → x,  x * 1 → x,  1 * x → x
      - neg(neg(x)) → x
    Returns a simplified Node (may be a different object).
    """
    # Simplify children first
    new_children = [_simplify_node(c) for c in node.children]
    n = Node(node.token, new_children)

    t  = n.token
    ch = n.children

    # Constant folding on known integer literals
    INT_VALS = {"0": 0, "1": 1, "2": 2, "-1": -1}
    if t in ("add", "mul", "sub") and len(ch) == 2:
        if ch[0].token in INT_VALS and ch[1].token in INT_VALS:
            a, b = INT_VALS[ch[0].token], INT_VALS[ch[1].token]
            result = (a + b if t == "add" else a * b if t == "mul" else a - b)
            if result in (-1, 0, 1, 2):
                return Node(str(result))

    # Identity rules
    if t == "mul":
        if ch[0].token == "0" or ch[1].token == "0": return Node("0")
        if ch[0].token == "1": return ch[1]
        if ch[1].token == "1": return ch[0]
    if t == "add":
        if ch[0].token == "0": return ch[1]
        if ch[1].token == "0": return ch[0]
    if t == "sub":
        if ch[1].token == "0": return ch[0]
    if t == "neg" and ch[0].token == "neg":
        return ch[0].children[0]   # neg(neg(x)) → x

    return n


# ---------------------------------------------------------------------------
# Candidate pool
# ---------------------------------------------------------------------------

class CandidatePool:
    """
    Maintains a deduplicated pool of candidates scored on (MSE, complexity).

    SYMBOLIC → NEURAL: the Pareto front computed here is the training signal.
    """

    # Fixed probe points for fast numerical dedup
    _PROBE_X = np.array([0.2, 0.5, 1.0, 1.5, 2.0], dtype=np.float32)

    def __init__(self, max_size: int = 300):
        self.candidates: List[Candidate] = []
        self.max_size   = max_size
        self._seen_repr: set = set()    # structural dedup (repr after simplification)
        self._seen_sigs: set = set()    # numerical signature dedup (5 probe values)

    @staticmethod
    def _num_sig(node: "Node", constants: List[float]) -> Optional[tuple]:
        """5-point numerical signature for fast dedup. None if all NaN."""
        from evaluator import eval_node as _ev
        vals = _ev(node, CandidatePool._PROBE_X, constants)
        if not np.any(np.isfinite(vals)):
            return None
        rounded = tuple(round(float(v), 4) if np.isfinite(v) else None for v in vals)
        return rounded

    def add(
        self,
        tokens: List[str],
        x_vals: np.ndarray,
        y_vals: np.ndarray,
    ) -> Optional[Candidate]:
        """
        Parse, simplify, fit constants, dedup, add if novel.

        Deduplication strategy (fast, no SymPy in hot path):
          1. Structural: repr(simplified_node) — catches identical trees
          2. Numerical:  5-point probe signature — catches algebraic equivalents
             (e.g. mul(x,x) vs pow2(x) both give the same 5 values)

        SymPy is reserved for the final falsification check only.
        Returns the Candidate if added, None if rejected.
        """
        if not is_valid_prefix(tokens):
            return None

        try:
            node = parse(tokens)
        except ParseError:
            return None

        # Skip expressions with 2+ constants — fitting is too slow
        if node.n_constants() >= 2:
            return None

        # From-scratch simplification (not SymPy)
        node = _simplify_node(node)

        # Structural dedup
        repr_key = repr(node)
        if repr_key in self._seen_repr:
            return None

        # Fit constants
        try:
            constants, fit = fit_constants(node, x_vals, y_vals)
        except Exception:
            return None

        if not np.isfinite(fit) or fit > 1e6:
            return None

        # Numerical signature dedup
        sig = self._num_sig(node, constants)
        if sig is None or sig in self._seen_sigs:
            return None

        # Complexity via our own node count (not SymPy in hot path)
        complexity = node.size()

        self._seen_repr.add(repr_key)
        self._seen_sigs.add(sig)

        cand = Candidate(
            tokens=tokens, node=node, constants=constants,
            fit_mse=fit, complexity=complexity,
        )
        self.candidates.append(cand)

        # Trim pool to max_size (keep best by MSE)
        if len(self.candidates) > self.max_size:
            self.candidates.sort(key=lambda c: c.fit_mse)
            self.candidates = self.candidates[:self.max_size]
            self._seen_repr = {repr(c.node) for c in self.candidates}
            self._seen_sigs = {self._num_sig(c.node, c.constants)
                               for c in self.candidates} - {None}

        return cand

    def compute_pareto(self) -> List[Candidate]:
        """
        Non-dominated sort on (MSE, complexity).
        Assigns .pareto_rank and .advantage to each candidate.

        SYMBOLIC → NEURAL interlock:
          advantage = 1 / (rank + 1) — Pareto-front candidates get
          the highest training weight fed back to the transformer.
        """
        n = len(self.candidates)
        if n == 0:
            return []

        mses  = np.array([c.fit_mse    for c in self.candidates])
        cmplx = np.array([c.complexity for c in self.candidates])

        # Normalise to [0, 1]
        mse_norm   = (mses  - mses.min())  / (mses.max()  - mses.min()  + 1e-8)
        cmplx_norm = (cmplx - cmplx.min()) / (cmplx.max() - cmplx.min() + 1e-8)

        # Non-dominated sort (O(n²) — fine for n ≤ 500)
        ranks = np.zeros(n, dtype=int)
        for i in range(n):
            for j in range(n):
                if i == j: continue
                # j dominates i if j is strictly better on at least one axis
                # and not worse on the other
                if (mse_norm[j] <= mse_norm[i] and cmplx_norm[j] <= cmplx_norm[i]
                        and (mse_norm[j] < mse_norm[i] or cmplx_norm[j] < cmplx_norm[i])):
                    ranks[i] += 1   # i is dominated by j

        for i, cand in enumerate(self.candidates):
            cand.pareto_rank = int(ranks[i])
            cand.advantage   = 1.0 / (ranks[i] + 1.0)

        return sorted(self.candidates, key=lambda c: c.pareto_rank)

    def best(self) -> Optional[Candidate]:
        """Return candidate with lowest MSE."""
        if not self.candidates:
            return None
        return min(self.candidates, key=lambda c: c.fit_mse)

    def pareto_front(self) -> List[Candidate]:
        """Return Pareto-rank-0 (non-dominated) candidates."""
        self.compute_pareto()
        return [c for c in self.candidates if c.pareto_rank == 0]
