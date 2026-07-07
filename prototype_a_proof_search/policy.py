"""
policy.py — 2-layer MLP policy with hand-rolled NumPy backprop.

Architecture
------------
  feature_vector (FEAT_DIM) → W1 → ReLU → W2 → scalar logit
  Softmax over all legal moves → categorical distribution over actions.

Feature vector for move (rule_idx, path):
  [0:N_RULES]           rule one-hot            (17 dims)
  [N_RULES:N_RULES+7]   node-type one-hot       ( 7 dims)
  [N_RULES+7:]          scalar context features ( 6 dims)
                          ↳ node depth (norm), is_root, expr size (norm),
                            subexpr size (norm), expr depth (norm), n_moves (norm)
Total: 30 dims

Training: behavioural cloning (imitation) on expert proof traces.
  Loss   = cross-entropy(softmax(logits), expert_action_index)
  Gradients computed analytically through ReLU and softmax.

NEURAL → SYMBOLIC interlock:
  score_moves() returns log-probs that the search uses as priorities.
  The policy never touches the proof state directly.
"""

from __future__ import annotations
import numpy as np
from typing import List, Tuple

from expressions import Expr, N_NODE_TYPES
from kernel import N_RULES

# Feature vector dimensions
FEAT_DIM  = N_RULES + N_NODE_TYPES + 6   # = 17 + 7 + 6 = 30
HIDDEN    = 64


def extract_features(
    expr: Expr,
    rule_idx: int,
    path: List[int],
    n_legal_moves: int,
) -> np.ndarray:
    """Build a fixed-size feature vector for one (rule, path) candidate move."""
    subexpr = expr.get_at(path)

    # Rule one-hot
    r = np.zeros(N_RULES, dtype=np.float32)
    r[rule_idx] = 1.0

    # Node-type one-hot
    t = np.zeros(N_NODE_TYPES, dtype=np.float32)
    t[subexpr.node_type_id()] = 1.0

    # Scalar context
    max_depth = max(expr.depth(), 1)
    scalars = np.array([
        len(path) / max(max_depth, 1),          # node depth (normalised)
        1.0 if not path else 0.0,               # is root
        min(expr.size()    / 20.0, 1.0),        # expression size
        min(subexpr.size() / 10.0, 1.0),        # subexpr size
        min(expr.depth()   /  6.0, 1.0),        # expression depth
        min(n_legal_moves  / 50.0, 1.0),        # action-space size
    ], dtype=np.float32)

    return np.concatenate([r, t, scalars])


class MLPPolicy:
    """
    Hand-rolled 2-layer MLP.

    Weights:
      W1: (HIDDEN, FEAT_DIM)    b1: (HIDDEN,)
      W2: (1, HIDDEN)           b2: (1,)
    """

    def __init__(self, hidden: int = HIDDEN, lr: float = 0.01, seed: int = 0):
        rng = np.random.default_rng(seed)
        scale = 0.1
        self.W1 = rng.standard_normal((hidden, FEAT_DIM)).astype(np.float32) * scale
        self.b1 = np.zeros(hidden, dtype=np.float32)
        self.W2 = rng.standard_normal((1, hidden)).astype(np.float32) * scale
        self.b2 = np.zeros(1, dtype=np.float32)
        self.lr = lr

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def _forward(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        X: (n_moves, FEAT_DIM)
        Returns: logits (n_moves,), hidden (n_moves, HIDDEN)
        """
        h = np.maximum(0.0, X @ self.W1.T + self.b1)     # ReLU  (n, H)
        logits = (h @ self.W2.T + self.b2).ravel()        # (n,)
        return logits, h

    def score_moves(
        self,
        expr: Expr,
        legal_moves: List[Tuple[int, List[int]]],
    ) -> np.ndarray:
        """
        Return log-probabilities over legal moves.

        NEURAL → SYMBOLIC interlock:
          These log-probs feed directly into the search priority queue.
          No proof state is modified here.
        """
        if not legal_moves:
            return np.array([], dtype=np.float32)

        n = len(legal_moves)
        X = np.stack([
            extract_features(expr, r, p, n)
            for r, p in legal_moves
        ])                                                  # (n, FEAT_DIM)

        logits, _ = self._forward(X)
        logits = logits - logits.max()                      # numerical stability
        probs  = np.exp(logits)
        probs /= probs.sum()
        return np.log(probs + 1e-8)

    # ------------------------------------------------------------------
    # Training step (behavioural cloning)
    # ------------------------------------------------------------------

    def train_step(
        self,
        examples: List[Tuple[Expr, List[Tuple[int, List[int]]], int]],
    ) -> float:
        """
        Gradient step on a batch of (expr, legal_moves, expert_idx) tuples.

        Loss = mean cross-entropy(softmax(logits), expert_idx).
        Gradient computed analytically:
          dL/d_logit_i  = p_i − 1[i == expert_idx]   (softmax-CE identity)
          dL/d_W2       = (dL/d_logit)ᵀ @ h
          dL/d_h        = (dL/d_logit) ⊗ W2
          dL/d_pre      = dL/d_h ⊙ 1[h > 0]           (ReLU gate)
          dL/d_W1       = (dL/d_pre)ᵀ @ X
        Returns mean loss.
        """
        if not examples:
            return 0.0

        total_loss = 0.0
        dW1 = np.zeros_like(self.W1)
        db1 = np.zeros_like(self.b1)
        dW2 = np.zeros_like(self.W2)
        db2 = np.zeros_like(self.b2)

        for expr, legal_moves, expert_idx in examples:
            if not legal_moves:
                continue
            n = len(legal_moves)
            X = np.stack([
                extract_features(expr, r, p, n)
                for r, p in legal_moves
            ])                                              # (n, FEAT_DIM)

            logits, h = self._forward(X)

            # Stable softmax
            shifted  = logits - logits.max()
            exp_l    = np.exp(shifted)
            probs    = exp_l / exp_l.sum()                 # (n,)

            # Cross-entropy loss
            total_loss += -np.log(probs[expert_idx] + 1e-8)

            # Gradient of CE w.r.t. logits
            d_logits = probs.copy()
            d_logits[expert_idx] -= 1.0                    # (n,)

            # Layer 2 gradients
            # logits = h @ W2ᵀ + b2  →  d/dW2 = d_logitsᵀ @ h
            dW2 += d_logits[np.newaxis, :] @ h             # (1, n) @ (n, H) = (1, H)
            db2 += d_logits.sum(keepdims=True)             # (1,)

            # Backprop into h
            # d/dh = d_logits[:, None] * W2  →  broadcast (n,1) * (1,H) = (n,H)
            d_h = d_logits[:, np.newaxis] * self.W2        # (n, H)

            # ReLU gate
            d_pre = d_h * (h > 0).astype(np.float32)      # (n, H)

            # Layer 1 gradients
            # pre = X @ W1ᵀ + b1  →  d/dW1 = d_preᵀ @ X
            dW1 += d_pre.T @ X                             # (H, n) @ (n, F) = (H, F)
            db1 += d_pre.sum(axis=0)                       # (H,)

        n_ex = len(examples)
        self.W1 -= self.lr * dW1 / n_ex
        self.b1 -= self.lr * db1 / n_ex
        self.W2 -= self.lr * dW2 / n_ex
        self.b2 -= self.lr * db2 / n_ex

        return total_loss / n_ex

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        np.savez(path, W1=self.W1, b1=self.b1, W2=self.W2, b2=self.b2)

    def load(self, path: str) -> None:
        d = np.load(path)
        self.W1, self.b1, self.W2, self.b2 = d["W1"], d["b1"], d["W2"], d["b2"]
