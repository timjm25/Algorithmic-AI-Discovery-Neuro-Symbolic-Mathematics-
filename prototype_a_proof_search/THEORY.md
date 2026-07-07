# Theory — Prototype A

## The non-obvious piece: why behavioural cloning on proof traces works as a training signal

### Setup

At training step t, the policy θ defines a distribution over actions at each
proof state s:

```
π_θ(a | s)  =  softmax( f_θ(s, a) )_{a ∈ A(s)}
```

where A(s) = KERNEL.legal_moves(s) is the kernel-supplied action space.

The expert iteration objective is:

```
θ*  =  argmin_θ  E_{(s, a*) ~ D*}[ − log π_θ(a* | s) ]
```

where D* is the distribution over (state, action) pairs from *successful*
proof traces.  This is standard behavioural cloning (imitation learning).

### Why kernel certification makes D* self-improving

D* is not a fixed dataset.  At iteration k, D* = {traces from policy θ_k}.
Because the kernel only accepts certified steps:

1. A proof found at iteration k is a *sound* proof in ring theory.
2. Every (s, a*) pair in D* corresponds to a step that the kernel accepted.
3. Training on D* pushes θ toward rules that lead to *kernel-verified* proofs.

This creates the self-reinforcing loop:
θ_k → search → D*_k (certified traces) → θ_{k+1} → ...

### Gradient derivation

The MLP maps feature vector x(s, a) ∈ ℝ^30 → logit ∈ ℝ via:

```
h(s,a)  =  ReLU( W₁ x(s,a) + b₁ )       h ∈ ℝ^64
f(s,a)  =  W₂ h(s,a) + b₂                f ∈ ℝ
```

Softmax over n moves: p_i = exp(f_i) / Σ_j exp(f_j)

Cross-entropy loss for expert action index k*:

```
L  =  − log p_{k*}  =  − f_{k*} + log Σ_j exp(f_j)
```

Key identity (softmax-CE gradient):

```
∂L / ∂f_i  =  p_i − 𝟙[i = k*]
```

This is the "prediction minus label" form.  Backprop then gives:

```
∂L / ∂W₂  =  (∂L/∂f)^⊤ h   ∈ ℝ^{1×64}

∂L / ∂h   =  (∂L/∂f) ⊗ W₂  ∈ ℝ^{n×64}    (outer product, broadcast)

∂L / ∂(W₁ x + b₁)  =  ∂L/∂h ⊙ 𝟙[h > 0]   (ReLU gate)

∂L / ∂W₁  =  (∂L/∂(pre))^⊤ X   ∈ ℝ^{64×30}
```

All of this is implemented analytically in `policy.py:train_step()` with no
automatic differentiation.

### Why the node count didn't improve (honest analysis)

The uniform baseline with beam width 8 is equivalent to a BFS over at most 8
branches at each level. For proofs of depth 2–4 with branching factor ~20
legal moves, the beam covers a large fraction of short proofs without guidance.

For the policy to beat this baseline on node count, the features must capture
*global* expression structure — which rule to apply and *where in the tree* to
apply it. The current MLP features encode: rule one-hot, node-type one-hot,
and 6 scalars (size, depth, etc.). These are local and cannot distinguish
"apply dist_l at the left subtree of an add_inv target" from "apply dist_l
elsewhere."

A tree-structured GNN (message passing over the AST) would encode this
context and is expected to show meaningful node reduction. The interlock
mechanism is correct; the policy capacity is the bottleneck.
