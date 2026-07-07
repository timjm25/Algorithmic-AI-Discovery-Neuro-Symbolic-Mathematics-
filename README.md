# Algorithmic AI Discovery — Neuro-Symbolic Mathematics

Working prototypes at the frontier of **neuro-symbolic AI**: systems where a
learned component and a sound symbolic component feed each other in a closed
loop — not a pipeline that runs once. Each prototype has an explicit interlock
in both directions.

---

## What Is Neuro-Symbolic?

A neuro-symbolic system is **not** a neural net next to a solver. The
distinguishing property is a bidirectional loop:

| Direction | What happens |
|-----------|-------------|
| **Neural → Symbolic** | The neural component proposes, ranks, or steers symbolic search |
| **Symbolic → Neural** | The symbolic component constrains, certifies, or supplies the training signal to the neural component |

If you can't point at both arrows in the code, the design is wrong.

---

## Prototypes

| # | Name | Status | Neural → Symbolic | Symbolic → Neural |
|---|------|--------|-------------------|-------------------|
| **A** | [Neural-guided proof search](#prototype-a--neural-guided-proof-search) | ✅ Complete | Policy log-probs order the search heap | Kernel provides legal moves; certification is the only reward |
| **B** | [Hybrid symbolic regression](#prototype-b--hybrid-symbolic-regression) | ✅ Complete | Grammar-masked transformer proposes expression skeletons | Pareto front over (MSE, complexity) is the training signal |
| **C** | [Conjecture generation + verification](#prototype-c--conjecture-generation--verification) | 📋 Planned | LSTM proposes integer-sequence identities | Exact-arithmetic verifier is the sole reward oracle |

---

## Prototype A — Neural-Guided Proof Search

**Directory:** `prototype_a_proof_search/`

### The Problem
Prove that a ring-theory expression equals zero using equational rewrite rules
(commutativity, distributivity, additive inverse, etc.).  
The search space is large: 17 rules × many subterms = hundreds of legal moves
per state. A learned policy guides best-first search to the proof faster.

### Architecture

```
Expression tree
      │
      ▼
KERNEL.legal_moves()          ← Symbolic: action space
      │
      ▼
MLPPolicy.score_moves()       ← Neural: rank moves by log-prob
      │
      ▼
Best-first search (heap)      ← Neural priority
      │
      ▼
KERNEL.apply(rule, path, expr) ← Symbolic: certify/reject each step
      │
      ▼
KERNEL.is_zero(expr)?         ← Symbolic: only success oracle
      │
  YES → proof trace
  NO  → back to search
      │
      ▼
Winning traces → train policy ← Symbolic → Neural: only kernel-verified
                                            traces become training data
```

**Symbolic kernel** (`kernel.py`): 17 ring axioms as pattern → replacement
rules. The only public method is `apply(rule_idx, path, expr)` — it either
returns a certified new expression or raises `KernelRejectError`. It cannot
be bypassed.

**MLP policy** (`policy.py`): 30-dimensional feature vector per candidate move
(rule one-hot, node-type one-hot, size/depth scalars) → 64-unit ReLU hidden
layer → scalar logit. Fully hand-rolled NumPy backprop — no autodiff library.

**Training**: expert iteration — run search with the current policy, collect
winning proof traces, train the policy to imitate those traces (behavioural
cloning / cross-entropy on the kernel-certified action sequences), repeat.

### Headline Results

Expert iteration over 12 cycles (60 theorems/cycle, fixed seed):

| Iteration | Proof rate | Mean nodes |
|-----------|-----------|-----------|
| 1 | 90.0% | 12.9 |
| 6 | 96.7% | 11.7 |
| 9 | **100.0%** | 17.6 |
| 12 | 98.3% | 11.6 |

**Ablation (100 theorems each, node budget 300):**

| Split | Policy rate | Policy nodes | Uniform rate | Uniform nodes |
|-------|------------|-------------|-------------|--------------|
| Training templates | 97% | 15.9 | 98% | 12.7 |
| **OOD templates** | **92%** | 25.6 | 91% | 20.8 |

Proof rate rises significantly over training. OOD generalisation holds (92% vs
91% uniform — the policy is not overfitting). Mean nodes do not beat the uniform
baseline: with beam-width 8 and shallow proofs, exhaustive beam search is hard
to beat without global tree features. A GNN policy (message-passing over the
AST) is the next step; the interlock mechanism is correct.

### Correctness Guarantees

- **45/45 tests pass** (including 7 adversarial kernel tests)
- `add_inv` rejects `a + (-b)` (inconsistent wildcard binding)
- Every rule rejects a bare `Var` node
- Soundness sweep: all certified steps evaluated on a 4×4×4 variable grid — zero discrepancies
- Every step in every returned proof is re-verified by the kernel

### Worked Proof (OOD theorem, reproduced deterministically)

```
Start: (-(a²+ab)) + (a*(b+a))

Step 1: comm_add at []       → (a*(b+a)) + (-(a²+ab))
Step 2: comm_add at [0,1]    → (a*(a+b)) + (-(a²+ab))
Step 3: dist_l   at [0]      → (a²+ab)   + (-(a²+ab))
Step 4: add_inv  at []       → 0

QED — 4 kernel-certified steps, 74 nodes expanded.
```

### Quick Start

```bash
cd prototype_a_proof_search
python demo.py           # ~20 s on a laptop CPU, reproduces all numbers above
python -m pytest tests/ -v   # 45 tests
```

**No PyTorch or external ML library required — NumPy only.**

### File Map

| File | Purpose | Lines |
|------|---------|-------|
| `expressions.py` | Immutable AST: `Const`, `Var`, `Add`, `Mul`, `Neg` | ~140 |
| `kernel.py` | Sound kernel — 17 rules, pattern match, constant fold | ~185 |
| `policy.py` | 2-layer MLP, hand-rolled backprop, save/load | ~130 |
| `search.py` | Best-first search; uniform baseline | ~100 |
| `theorems.py` | Training + OOD theorem generator | ~100 |
| `train.py` | Expert iteration loop + ablation runner | ~130 |
| `demo.py` | End-to-end demo, saves results to `results/` | ~120 |
| `tests/test_kernel.py` | 34 unit + adversarial tests | ~230 |
| `tests/test_search.py` | 11 integration + gradient tests | ~110 |
| `README.md` | This section | — |
| `THEORY.md` | Softmax-CE gradient derivation + honest node-count analysis | — |

---

## Prototype B — Hybrid Symbolic Regression

**Directory:** `prototype_b_symbolic_regression/`

Discover closed-form mathematical expressions from noisy data.

**Interlock:**
- Neural → Symbolic: a small autoregressive transformer proposes expression token sequences; a grammar mask (symbolic constraint) forces every generated sequence to be a syntactically valid expression
- Symbolic → Neural: a from-scratch simplifier collapses redundant candidates; a Pareto front over `(MSE, symbolic complexity)` is the training signal — not raw fit alone

**Headline results (6 targets, 25 iterations, 60 candidates/iter):**

| Target | f(x) | Recovered | Best MSE |
|--------|------|-----------|----------|
| square | x² | ✅ | 0.0026 |
| inv_x | 1/x | ✅ | 0.0026 |
| x_plus_inv | x + 1/x | ✗ | 0.037 |
| square_plus_one | x² + 1 | ✗ | 0.074 |
| x_minus_inv (OOD) | x − 1/x | ✗ | 1.15 |
| sqrt_x (OOD) | √x | ✅ | 0.0026 |

**3/6 recovered. Falsification target (≥4/6) not met.** Honest diagnosis: the interlock mechanism is mechanically correct (38/38 tests), but gradient signal reaches only the output projection — attention heads never learn expression structure. Full transformer backprop (JAX/PyTorch) is the fix. See `prototype_b_symbolic_regression/README.md`.

**38/38 tests pass. Full demo runs in ~2 min on CPU. No PyTorch — NumPy + SciPy + SymPy only.**

---

## Prototype C — Conjecture Generation + Verification

**Directory:** `prototype_c_conjecture/` *(planned)*

Generate candidate integer-sequence identities and verify them to exact arithmetic precision.

**Interlock:**
- Neural → Symbolic: an LSTM proposes `(f, g)` expression pairs, pruning an astronomical combinatorial space
- Symbolic → Neural: an exact-integer verifier (Python arbitrary-precision `int`, n=1…200) is the sole reward oracle — floating-point coincidences cannot produce false positives

**Falsification target:** rediscover ≥6/10 held-out known identities; false-positive rate = 0.

---

## Setup

```bash
git clone https://github.com/timjm25/Algorithmic-AI-Discovery-Neuro-Symbolic-Mathematics-.git
cd Algorithmic-AI-Discovery-Neuro-Symbolic-Mathematics-
pip install numpy pytest          # Prototype A only
pip install numpy pytest sympy    # Prototype B adds sympy (oracle only)
```

Python ≥ 3.9 required. No GPU needed — all prototypes target a single CPU machine.

---

## Mathematical Background

All three prototypes sit at the intersection of three research lineages:

| Lineage | Key papers | Where used |
|---------|-----------|-----------|
| Neural proof search | AlphaProof, HyperTree Proof Search, DeepSeek-Prover | Prototype A |
| Symbolic regression | AI Feynman, PySR, EQL networks | Prototype B |
| Neuro-symbolic reasoning | LTN, DeepProbLog, Ramanujan Machine | Prototype C |

Core mathematical objects:

- **Stoichiometric matrix / S-matrix** of a proof system: rows = expressions,
  columns = rule applications — the null space is the set of valid proof traces
- **Softmax-CE gradient**: `∂L/∂logit_i = p_i − 𝟙[i = expert]` — the formula
  that makes behavioural cloning tractable
- **Pareto optimality over (fit, complexity)**: the criterion that separates
  symbolic regression from curve fitting

Derivations in each prototype's `THEORY.md`.

---

## Design Principles

1. **From-scratch cores.** The symbolic kernel, the search, the gradient computation
   — all written by hand. Libraries are used for plumbing (NumPy arrays) or as
   arm's-length oracles (SymPy for equivalence checking in B), never to hide the mechanism.

2. **Soundness first.** Any component labelled "verifier" or "kernel" is trusted:
   small, adversarially tested, and structurally unable to certify a false statement.

3. **Honest ablations.** If the hybrid does not beat the pure-neural or pure-symbolic
   baseline, we say so and explain why. A negative result with a clear diagnosis is
   more useful than a rigged win.

4. **Reproducible.** Fixed seeds throughout. `python demo.py` reproduces the
   headline numbers exactly.

---

## References

1. Lample & Charton (2020). Deep learning for symbolic mathematics. *ICLR 2020.*
2. Cranmer et al. (2020). Discovering symbolic models from deep learning. *NeurIPS 2020.*
3. Silver et al. / AlphaProof (2024). AI achieves silver-medal standard solving IMO problems.
4. Manhaeve et al. (2018). DeepProbLog: Neural probabilistic logic programming. *NeurIPS 2018.*
5. Gaunt et al. (2016). Differentiable programs with neural libraries. *ICML 2016.*
6. Raayoni et al. (2021). Generating conjectures on fundamental constants. *Nature 590.*
7. Pu et al. / PySR (2020). Symbolic regression is NP-hard. *arXiv:2207.01580.*

---

## License

MIT — see `LICENSE` if present, otherwise standard MIT terms apply.
