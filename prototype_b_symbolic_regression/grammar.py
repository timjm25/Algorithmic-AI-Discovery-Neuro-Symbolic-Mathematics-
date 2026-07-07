"""
grammar.py — Token vocabulary and expression grammar for symbolic regression.

Expressions are sequences of tokens in prefix (Polish) notation so that
the tree structure is unambiguous without parentheses.

Token set (25 tokens + 3 control tokens):
  Operators : add, mul, sub, div, pow2  (binary)
  Functions : sin, exp, sqrt, inv, neg  (unary)
  Terminals : x, C                      (C = learnable constant placeholder)
  Constants : 0, 1, 2, -1               (integer literals)
  Control   : <BOS>, <EOS>, <PAD>

An expression token sequence is valid if it parses to a complete binary/unary
tree.  Invalid sequences are those that end mid-tree or use more than
MAX_TOKENS tokens.

The parser here is a recursive descent parser written from scratch —
no eval(), no exec(), no ast.parse().
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

# ---------------------------------------------------------------------------
# Vocabulary
# ---------------------------------------------------------------------------

BINARY_OPS  = ["add", "mul", "sub", "div"]
UNARY_OPS   = ["pow2", "sin", "exp", "sqrt", "inv", "neg"]
TERMINALS   = ["x", "C", "0", "1", "2", "-1"]
CONTROL     = ["<BOS>", "<EOS>", "<PAD>"]

ALL_TOKENS  = CONTROL + BINARY_OPS + UNARY_OPS + TERMINALS

TOKEN_TO_ID = {t: i for i, t in enumerate(ALL_TOKENS)}
ID_TO_TOKEN = {i: t for t, i in TOKEN_TO_ID.items()}

VOCAB_SIZE  = len(ALL_TOKENS)
BOS_ID      = TOKEN_TO_ID["<BOS>"]
EOS_ID      = TOKEN_TO_ID["<EOS>"]
PAD_ID      = TOKEN_TO_ID["<PAD>"]

MAX_TOKENS  = 15   # max prefix sequence length (excl. BOS/EOS)


def arity(token: str) -> int:
    if token in BINARY_OPS: return 2
    if token in UNARY_OPS:  return 1
    return 0  # terminal


# ---------------------------------------------------------------------------
# Expression tree node
# ---------------------------------------------------------------------------

@dataclass
class Node:
    token: str
    children: List[Node] = field(default_factory=list)

    def to_prefix(self) -> List[str]:
        return [self.token] + [t for c in self.children for t in c.to_prefix()]

    def depth(self) -> int:
        return 0 if not self.children else 1 + max(c.depth() for c in self.children)

    def size(self) -> int:
        return 1 + sum(c.size() for c in self.children)

    def n_constants(self) -> int:
        return (1 if self.token == "C" else 0) + sum(c.n_constants() for c in self.children)

    def __repr__(self) -> str:
        return _node_to_infix(self)


def _node_to_infix(node: Node) -> str:
    t = node.token
    ch = node.children
    if t == "add":  return f"({_node_to_infix(ch[0])}+{_node_to_infix(ch[1])})"
    if t == "mul":  return f"({_node_to_infix(ch[0])}*{_node_to_infix(ch[1])})"
    if t == "sub":  return f"({_node_to_infix(ch[0])}-{_node_to_infix(ch[1])})"
    if t == "div":  return f"({_node_to_infix(ch[0])}/{_node_to_infix(ch[1])})"
    if t == "pow2": return f"({_node_to_infix(ch[0])}**2)"
    if t == "sin":  return f"sin({_node_to_infix(ch[0])})"
    if t == "exp":  return f"exp({_node_to_infix(ch[0])})"
    if t == "sqrt": return f"sqrt({_node_to_infix(ch[0])})"
    if t == "inv":  return f"(1/{_node_to_infix(ch[0])})"
    if t == "neg":  return f"(-{_node_to_infix(ch[0])})"
    return t   # terminal: x, C, 0, 1, 2, -1


# ---------------------------------------------------------------------------
# Parser: token list → Node tree
# ---------------------------------------------------------------------------

class ParseError(Exception):
    pass


def parse(tokens: List[str]) -> Node:
    """
    Parse a prefix token sequence into a Node tree.
    Raises ParseError if the sequence is incomplete or has leftover tokens.
    """
    idx = [0]

    def _parse_node() -> Node:
        if idx[0] >= len(tokens):
            raise ParseError("Unexpected end of token sequence")
        tok = tokens[idx[0]]
        idx[0] += 1
        node = Node(tok)
        for _ in range(arity(tok)):
            node.children.append(_parse_node())
        return node

    root = _parse_node()
    if idx[0] != len(tokens):
        raise ParseError(
            f"Leftover tokens after complete expression: {tokens[idx[0]:]}"
        )
    return root


def is_valid_prefix(tokens: List[str]) -> bool:
    """Check whether a token sequence forms a complete, valid expression."""
    try:
        parse(tokens)
        return True
    except ParseError:
        return False


def expected_remaining(tokens: List[str]) -> int:
    """
    Number of additional tokens needed to complete a valid expression,
    given a partial prefix sequence.  Returns -1 if the sequence is
    already complete, -2 if the sequence is invalid.
    """
    need = 1  # need one more root
    for tok in tokens:
        if tok not in TOKEN_TO_ID or tok in ("<BOS>", "<EOS>", "<PAD>"):
            return -2
        need -= 1        # consumed one slot
        need += arity(tok)  # opened new slots
        if need < 0:
            return -2    # closed too many slots
    return need          # 0 = complete, >0 = still need more
