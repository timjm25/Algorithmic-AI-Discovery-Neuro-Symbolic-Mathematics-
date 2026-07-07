"""
search.py — Best-first proof search using the kernel and policy.

INTERLOCK SUMMARY
-----------------
  Neural → Symbolic:
    policy.score_moves() supplies log-probabilities that define the heap
    priority.  The search explores high-probability moves first.

  Symbolic → Neural:
    KERNEL.legal_moves()  provides the action space — the policy scores
    only moves the kernel permits.
    KERNEL.apply()        certifies each step; a KernelRejectError (which
    never fires in normal search because we only propose legal moves) is
    the hard gate — no uncertified state ever enters the queue.
    KERNEL.is_zero()      is the sole goal check; the policy cannot declare
    success.

Search algorithm: best-first (min-heap on accumulated negative log-prob).
  Visited set deduplicates expressions by repr() to avoid cycles.
  Beam width k: at each node, only the top-k moves by policy score are expanded.
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from expressions import Expr, Const
from kernel import KERNEL, RULE_NAMES
from policy import MLPPolicy


# ---------------------------------------------------------------------------
# Result structures
# ---------------------------------------------------------------------------

@dataclass
class ProofStep:
    rule_name: str
    rule_idx: int
    path: List[int]
    expr_before: Expr
    expr_after: Expr


@dataclass
class ProofResult:
    found: bool
    steps: List[ProofStep]
    nodes_expanded: int
    final_expr: Expr

    @property
    def proof_length(self) -> int:
        return len(self.steps)


# ---------------------------------------------------------------------------
# Best-first search
# ---------------------------------------------------------------------------

def best_first_search(
    start: Expr,
    policy: MLPPolicy,
    max_nodes: int = 300,
    beam_width: int = 8,
) -> ProofResult:
    """
    Best-first proof search.

    Parameters
    ----------
    start      : initial expression (must be provably equal to 0)
    policy     : MLPPolicy — scores moves via log-probabilities
    max_nodes  : expansion budget
    beam_width : at each node, expand only the top-k moves by policy score

    Returns ProofResult (found=True with certified trace on success).
    """
    start = KERNEL.constant_fold(start)
    if KERNEL.is_zero(start):
        return ProofResult(found=True, steps=[], nodes_expanded=0,
                           final_expr=start)

    # Heap entries: (neg_accumulated_logprob, tiebreak_id, expr, steps_so_far)
    counter = 0
    heap = [(0.0, counter, start, [])]
    visited: set = set()
    nodes_expanded = 0

    while heap and nodes_expanded < max_nodes:
        neg_lp, _, curr, steps = heapq.heappop(heap)

        key = repr(curr)
        if key in visited:
            continue
        visited.add(key)
        nodes_expanded += 1

        # SYMBOLIC → NEURAL: kernel supplies legal action space
        moves = KERNEL.legal_moves(curr)
        if not moves:
            continue

        # NEURAL → SYMBOLIC: policy scores the moves for priority ordering
        log_probs = policy.score_moves(curr, moves)

        # Take top-k by policy score
        ranked = sorted(
            zip(log_probs, moves),
            key=lambda x: -x[0],
        )[:beam_width]

        for lp, (rule_idx, path) in ranked:
            # SYMBOLIC gate: kernel certifies the step
            try:
                new_expr = KERNEL.apply(rule_idx, path, curr)
            except Exception:
                continue                              # should never fire

            new_expr = KERNEL.constant_fold(new_expr)

            step = ProofStep(
                rule_name=RULE_NAMES[rule_idx],
                rule_idx=rule_idx,
                path=path,
                expr_before=curr,
                expr_after=new_expr,
            )

            # SYMBOLIC goal check — policy cannot declare victory
            if KERNEL.is_zero(new_expr):
                return ProofResult(
                    found=True,
                    steps=steps + [step],
                    nodes_expanded=nodes_expanded,
                    final_expr=new_expr,
                )

            counter += 1
            heapq.heappush(
                heap,
                (neg_lp - lp, counter, new_expr, steps + [step]),
            )

    return ProofResult(
        found=False, steps=[], nodes_expanded=nodes_expanded, final_expr=curr
    )


# ---------------------------------------------------------------------------
# Uniform baseline (same search, random/uniform priority)
# ---------------------------------------------------------------------------

class _UniformPolicy:
    """Scores all moves equally — the no-learning baseline."""
    import numpy as np

    def score_moves(self, expr, legal_moves):
        import numpy as np
        n = len(legal_moves)
        if n == 0:
            return np.array([], dtype=float)
        return -self.np.log(n) * self.np.ones(n, dtype=float)

    np = __import__("numpy")


def uniform_search(
    start: Expr,
    max_nodes: int = 300,
    beam_width: int = 8,
) -> ProofResult:
    """Best-first search with a uniform (non-learned) policy — ablation baseline."""
    return best_first_search(start, _UniformPolicy(), max_nodes, beam_width)
