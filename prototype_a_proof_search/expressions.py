"""
expressions.py — Immutable expression trees for ring expressions over Z[a, b, c].

Node types
----------
  Const(n: int)         — integer literal
  Var(name: str)        — variable in {"a", "b", "c"}
  Add(left, right)      — addition
  Mul(left, right)      — multiplication
  Neg(child)            — unary negation

All nodes are immutable and structurally compared via __eq__ / __hash__.
"""

from __future__ import annotations
from typing import Dict, Iterator, List, Tuple

# Node type integers used by the policy as input features
NODE_TYPES: Dict[str, int] = {
    "Const": 0,
    "Var_a": 1,
    "Var_b": 2,
    "Var_c": 3,
    "Add":   4,
    "Mul":   5,
    "Neg":   6,
}
N_NODE_TYPES = len(NODE_TYPES)


class Expr:
    """Base class — do not instantiate directly."""

    # Arithmetic sugar so we can write `a + b`, `a * b`, `-a`, `a - b`
    def __add__(self, other: Expr) -> Add:  return Add(self, other)
    def __mul__(self, other: Expr) -> Mul:  return Mul(self, other)
    def __neg__(self)              -> Neg:  return Neg(self)
    def __sub__(self, other: Expr) -> Add:  return Add(self, Neg(other))

    def __eq__(self, other: object) -> bool:
        return type(self) is type(other) and self._eq_fields(other)

    def __hash__(self) -> int:
        return hash(repr(self))

    # --- subclass contract ---------------------------------------------------

    def _eq_fields(self, other: Expr) -> bool:
        raise NotImplementedError

    def children(self) -> List[Expr]:
        """Direct child nodes."""
        raise NotImplementedError

    def node_type_id(self) -> int:
        """Integer encoding for the GNN embedding table."""
        raise NotImplementedError

    def eval(self, env: Dict[str, int]) -> int:
        """Evaluate to an integer under variable assignment."""
        raise NotImplementedError

    def _rebuild(self, new_children: List[Expr]) -> Expr:
        """Return a copy of this node with new children (same type)."""
        raise NotImplementedError

    # --- derived properties --------------------------------------------------

    def depth(self) -> int:
        ch = self.children()
        return 0 if not ch else 1 + max(c.depth() for c in ch)

    def size(self) -> int:
        return 1 + sum(c.size() for c in self.children())

    def all_nodes(self) -> List[Tuple[List[int], Expr]]:
        """
        BFS/DFS over every node in the tree.
        Returns [(path, node), ...] where path is the child-index route from root.
        """
        result: List[Tuple[List[int], Expr]] = [([], self)]
        for i, child in enumerate(self.children()):
            for sub_path, sub_node in child.all_nodes():
                result.append(([i] + sub_path, sub_node))
        return result

    def get_at(self, path: List[int]) -> Expr:
        """Retrieve the subexpression reachable by following path."""
        node: Expr = self
        for idx in path:
            node = node.children()[idx]
        return node

    def replace_at(self, path: List[int], new_expr: Expr) -> Expr:
        """Return a new tree identical to self except path points to new_expr."""
        if not path:
            return new_expr
        ch = list(self.children())
        ch[path[0]] = ch[path[0]].replace_at(path[1:], new_expr)
        return self._rebuild(ch)

    def is_zero(self) -> bool:
        return isinstance(self, Const) and self.n == 0

    def is_one(self) -> bool:
        return isinstance(self, Const) and self.n == 1


# ---------------------------------------------------------------------------
# Concrete node types
# ---------------------------------------------------------------------------

class Const(Expr):
    __slots__ = ("n",)
    def __init__(self, n: int):            self.n = int(n)
    def _eq_fields(self, o: Const) -> bool: return self.n == o.n
    def children(self) -> List[Expr]:      return []
    def node_type_id(self) -> int:         return NODE_TYPES["Const"]
    def eval(self, env) -> int:            return self.n
    def _rebuild(self, _) -> Const:        return self
    def __repr__(self) -> str:             return str(self.n)


class Var(Expr):
    __slots__ = ("name",)
    def __init__(self, name: str):
        assert name in ("a", "b", "c"), f"Unknown variable '{name}'"
        self.name = name
    def _eq_fields(self, o: Var) -> bool:  return self.name == o.name
    def children(self) -> List[Expr]:      return []
    def node_type_id(self) -> int:         return NODE_TYPES[f"Var_{self.name}"]
    def eval(self, env) -> int:            return env[self.name]
    def _rebuild(self, _) -> Var:          return self
    def __repr__(self) -> str:             return self.name


class Add(Expr):
    __slots__ = ("left", "right")
    def __init__(self, left: Expr, right: Expr):
        self.left, self.right = left, right
    def _eq_fields(self, o: Add) -> bool:
        return self.left == o.left and self.right == o.right
    def children(self) -> List[Expr]:      return [self.left, self.right]
    def node_type_id(self) -> int:         return NODE_TYPES["Add"]
    def eval(self, env) -> int:            return self.left.eval(env) + self.right.eval(env)
    def _rebuild(self, ch) -> Add:         return Add(ch[0], ch[1])
    def __repr__(self) -> str:             return f"({self.left} + {self.right})"


class Mul(Expr):
    __slots__ = ("left", "right")
    def __init__(self, left: Expr, right: Expr):
        self.left, self.right = left, right
    def _eq_fields(self, o: Mul) -> bool:
        return self.left == o.left and self.right == o.right
    def children(self) -> List[Expr]:      return [self.left, self.right]
    def node_type_id(self) -> int:         return NODE_TYPES["Mul"]
    def eval(self, env) -> int:            return self.left.eval(env) * self.right.eval(env)
    def _rebuild(self, ch) -> Mul:         return Mul(ch[0], ch[1])
    def __repr__(self) -> str:             return f"({self.left} * {self.right})"


class Neg(Expr):
    __slots__ = ("child",)
    def __init__(self, child: Expr):       self.child = child
    def _eq_fields(self, o: Neg) -> bool:  return self.child == o.child
    def children(self) -> List[Expr]:      return [self.child]
    def node_type_id(self) -> int:         return NODE_TYPES["Neg"]
    def eval(self, env) -> int:            return -self.child.eval(env)
    def _rebuild(self, ch) -> Neg:         return Neg(ch[0])
    def __repr__(self) -> str:             return f"(-{self.child})"


# ---------------------------------------------------------------------------
# Convenience constructors
# ---------------------------------------------------------------------------

def C(n: int) -> Const: return Const(n)
def V(name: str) -> Var: return Var(name)

# Module-level variable singletons
a = V("a")
b = V("b")
c = V("c")
