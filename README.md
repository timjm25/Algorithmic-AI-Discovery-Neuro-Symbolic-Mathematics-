# Algorithmic AI Discovery — Neuro-Symbolic Mathematics

Three working prototypes at the frontier of neuro-symbolic mathematics.
Each prototype has a **genuine interlock**: the neural and symbolic components
feed each other in a loop, not a one-shot pipeline.

## Prototypes

| # | Idea | Status | Interlock mechanism |
|---|------|--------|---------------------|
| A | Neural-guided proof search | **Complete** | Policy prioritises search / kernel certifies steps |
| B | Hybrid symbolic regression | Planned | Transformer proposes / CAS collapses & Pareto-selects |
| C | Conjecture generation + verification | Planned | LSTM proposes / exact-integer verifier rewards |

## Quick start (Prototype A)

```bash
git clone https://github.com/timjm25/Algorithmic-AI-Discovery-Neuro-Symbolic-Mathematics-.git
cd Algorithmic-AI-Discovery-Neuro-Symbolic-Mathematics-
pip install numpy pytest

cd prototype_a_proof_search
python demo.py            # ~20 s on laptop CPU
python -m pytest tests/ -v
```

## Repository layout

```
prototype_a_proof_search/
├── expressions.py    — immutable AST nodes
├── kernel.py         — sound rewriting kernel (17 ring axioms)
├── policy.py         — 2-layer MLP, hand-rolled NumPy backprop
├── search.py         — best-first proof search
├── theorems.py       — theorem generator (training + OOD families)
├── train.py          — expert iteration loop
├── demo.py           — runnable demo (reproduces all headline numbers)
├── tests/            — 45 unit + adversarial tests
├── results/          — JSON metrics, policy weights (generated on run)
├── README.md
└── THEORY.md
```

## Headline result (Prototype A)

Expert iteration over 12 cycles on ring-theory proof tasks:

- Proof rate: **90% → ~98-100%** (training theorems)
- OOD proof rate: **92%** vs 91% uniform baseline (generalises)
- Mean nodes: **comparable to uniform** — see README.md for honest analysis
  (policy capacity, not interlock, is the bottleneck; a GNN would improve this)
- All 45 tests pass including adversarial kernel soundness sweeps
