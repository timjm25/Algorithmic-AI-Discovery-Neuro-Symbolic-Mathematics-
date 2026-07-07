# Prototype B — Hybrid Symbolic Regression

## Interlock Statement

**Neural → Symbolic:**
`TinyTransformer.generate()` proposes candidate expression token sequences.
The grammar mask (from `grammar.expected_remaining()`) constrains the
transformer's output distribution to structurally valid continuations only —
every generated sequence parses to a valid expression tree.

**Symbolic → Neural:**
`CandidatePool.compute_pareto()` scores each candidate on two symbolic axes —
numerical fit (MSE on data) and structural complexity (node count).
Pareto rank determines the `advantage` weight fed back to the transformer
via `train_on_sequences()`. Only Pareto-front candidates get high training weight;
the transformer learns to propose expressions that survive symbolic selection.

---

## How to Run

```bash
cd prototype_b_symbolic_regression
python demo.py           # full search + ablation, ~2 min on laptop CPU
python -m pytest tests/ -v   # 38 tests
```

No PyTorch. NumPy + SciPy + SymPy only.

---

## Architecture

```
Grammar-masked autoregressive sampling
        │  (Neural → Symbolic: grammar constrains token distribution)
        ▼
Candidate token sequences
        │
        ▼
Parse (grammar.py, from scratch)
        │
        ▼
Structural simplification (_simplify_node — from scratch)
        │  zero-annihilation, identity rules, constant fold
        ▼
Constant fitting (evaluator.fit_constants — grid + L-BFGS-B)
        │
        ▼
Numerical signature dedup (5-probe hash — fast dedup without SymPy)
        │
        ▼
CandidatePool (scored, deduplicated)
        │
        ▼
Pareto front over (MSE, complexity)    ← Symbolic → Neural
        │   advantage = 1 / (pareto_rank + 1)
        ▼
transformer.train_on_sequences(seqs, advantages)
        │   weighted cross-entropy, Adam
        ▼
Updated transformer (generates better candidates next iteration)
```

**SymPy is used only for the final falsification check** (equivalence to ground truth).
It never touches the training loop — all hot-path deduplication uses structural reprs
and 5-point numerical signatures (both from scratch, sub-millisecond).

### Key files

| File | Purpose |
|------|---------|
| `grammar.py` | Token vocabulary, prefix parser (from scratch), arity, validity check |
| `evaluator.py` | Numerical eval, constant fitting, SymPy oracle (arm's-length) |
| `pareto.py` | Structural simplifier, candidate pool, Pareto non-dominated sort |
| `transformer.py` | Tiny autoregressive transformer (NumPy), grammar-masked generation |
| `search.py` | Neuro-symbolic search loop |
| `targets.py` | 6 ground-truth identities (4 training, 2 OOD) |
| `demo.py` | Full demo + random baseline ablation |

---

## Headline Results

6 target identities, 25 search iterations each, 60 candidates/iteration:

| Target | Description | Recovered | Best MSE | Best expression |
|--------|-------------|-----------|----------|-----------------|
| square | f(x) = x² | **YES** | 0.0026 | `(-x)²` (equiv. to x²) |
| inv_x | f(x) = 1/x | **YES** | 0.0026 | numerically equivalent form |
| x_plus_inv | f(x) = x + 1/x | no | 0.037 | deeply-nested sin approx |
| square_plus_one | f(x) = x² + 1 | no | 0.074 | `x/(2-C²)` |
| x_minus_inv (OOD) | f(x) = x - 1/x | no | 1.15 | `(1/x)/C` |
| sqrt_x (OOD) | f(x) = √x | **YES** | 0.0026 | `sqrt(x)*C` (C≈1) |

**Total: 3/6 recovered. Falsification target was ≥4/6 — NOT MET.**

---

## Ablation: Transformer vs. Random Baseline

| Target | Transformer | Random |
|--------|------------|--------|
| square | ✓ | ✓ |
| inv_x | ✓ | ✓ |
| x_plus_inv | ✗ | ✗ |
| square_plus_one | ✗ | ✓ |
| x_minus_inv (OOD) | ✗ | ✗ |
| sqrt_x (OOD) | ✓ | ✓ |
| **Total** | **3/6** | **4/6** |

**Honest assessment:**

The transformer did not outperform random grammar sampling in 25 iterations.
Both fail on `x+1/x` (requires two structural primitives to co-appear: `add` + `inv`).
The random baseline accidentally finds `x²+1` via broad coverage; the transformer
converges to a local basin.

The interlock mechanism is mechanically correct:
- Grammar mask works (all 10,000+ generated sequences are valid expressions — verified)
- Pareto front computes correctly (38/38 tests pass)
- Symbolic selection feeds back to the transformer
- The training signal is too shallow: backprop reaches only the output projection
  and embedding layers (full transformer backprop in NumPy was impractical)

The result is a genuine negative on the falsification target, reported plainly.

---

## What Works Well

1. **Grammar masking** — 100% of generated sequences are syntactically valid. Without masking, ~60% of sequences are invalid (shown in `test_generate_without_mask_can_be_invalid`).

2. **Numerical dedup** — 5-probe signature collapses algebraically equivalent expressions (e.g. `mul(x,x)` and `pow2(x)` get the same signature) in O(1) without SymPy.

3. **Pareto selection** — correctly identifies non-dominated candidates. The Pareto front prefers `sqrt(x)` (complexity 2, MSE 0.003) over `sin(sin(sin(x)))` (complexity 8, MSE 0.04) even though the latter has lower MSE.

4. **SymPy at arm's length** — only called for the final equivalence check, not in the 12,000+ hot-path candidate evaluations.

---

## Limitations and What to Fix

1. **Shallow backprop**: only output projection + embedding weights receive gradients. A full NumPy transformer backprop (or switching to JAX/PyTorch) would propagate gradients through attention, making the learned distribution actually prefer useful expression structures.

2. **No additive structure in grammar**: `x + 1/x` requires the transformer to emit `add x inv x`. With a uniform prior over 25 tokens and only 60 candidates/iteration, this 4-token sequence has probability ~(1/25)⁴ ≈ 2.6×10⁻⁶ — the search simply needs more iterations or targeted exploration.

3. **Single shared model**: the transformer is reset between targets. A meta-learned initialisation (MAML-style) would carry structure across targets.

---

## Correctness Audit

- **38/38 tests pass**
- SymPy adversarial test: `are_equivalent(x², x+1)` correctly returns False
- Grammar mask: `expected_remaining` correctly counts open slots at every prefix
- Simplifier: `mul(x, 1) → x`, `neg(neg(x)) → x`, `add(1,1) → 2` all verified
- Numerical dedup: `mul(x,x)` and `pow2(x)` share the same 5-probe signature

---

## Notes on the SymPy Oracle

SymPy `simplify(A - B) == 0` is called once per target at the end (6 calls total).
It is not called during candidate generation, fitting, or Pareto computation.
This is "arm's-length" use: SymPy is an oracle for one binary question
(equivalent or not), not the search engine or the training signal.
