"""
Value mutation operators.

This module contains operators that mutate constant values like booleans, 
boundary numbers, and strings.
"""

import ast
from typing import Iterable, Tuple

from ..core import Operator, Example


class BooleanReplacer(Operator):
    """Base class for boolean constant replacement."""
    
    def __init__(self, from_value: bool, to_value: bool):
        self.from_value = from_value
        self.to_value = to_value
    
    def mutation_positions(self, node: ast.AST) -> Iterable[Tuple[int, int]]:
        """Find boolean constants that match our target."""
        if isinstance(node, ast.Constant) and node.value is self.from_value:
            yield (getattr(node, 'lineno', 0), getattr(node, 'col_offset', 0))
        elif isinstance(node, ast.NameConstant) and node.value is self.from_value:  # Python < 3.8
            yield (getattr(node, 'lineno', 0), getattr(node, 'col_offset', 0))
    
    def mutate(self, node: ast.AST, index: int) -> ast.AST:
        """Replace the boolean value."""
        assert index == 0
        
        if isinstance(node, ast.Constant):
            node.value = self.to_value
        elif isinstance(node, ast.NameConstant):
            node.value = self.to_value
            
        return node


class ReplaceTrueWithFalse(BooleanReplacer):
    """Replace True with False."""
    
    def __init__(self):
        super().__init__(True, False)
    
    @classmethod
    def examples(cls):
        return (
            Example("if True: pass", "if False: pass"),
            Example("x = True", "x = False"),
        )


class ReplaceFalseWithTrue(BooleanReplacer):
    """Replace False with True."""
    
    def __init__(self):
        super().__init__(False, True)
    
    @classmethod
    def examples(cls):
        return (
            Example("if False: pass", "if True: pass"),
            Example("x = False", "x = True"),
        )


class BoundaryValueMutator(Operator):
    """Mutate common boundary values (0, 1, -1)."""
    
    def __init__(self, from_value: int, to_value: int):
        self.from_value = from_value
        self.to_value = to_value
    
    def mutation_positions(self, node: ast.AST) -> Iterable[Tuple[int, int]]:
        """Find numeric constants that match our boundary value."""
        if isinstance(node, (ast.Constant, ast.Num)):
            value = self._get_value(node)
            if value == self.from_value:
                yield (getattr(node, 'lineno', 0), getattr(node, 'col_offset', 0))
    
    def mutate(self, node: ast.AST, index: int) -> ast.AST:
        """Replace the boundary value."""
        assert index == 0
        
        if isinstance(node, ast.Constant):
            node.value = self.to_value
        elif isinstance(node, ast.Num):
            node.n = self.to_value
            
        return node
    
    def _get_value(self, node):
        """Get the numeric value from a node."""
        if isinstance(node, ast.Constant):
            return node.value
        elif isinstance(node, ast.Num):
            return node.n
        return None


class ReplaceZeroWithOne(BoundaryValueMutator):
    """Replace 0 with 1 (common boundary)."""
    
    def __init__(self):
        super().__init__(0, 1)
    
    @classmethod
    def examples(cls):
        return (
            Example("range(0, 10)", "range(1, 10)"),
            Example("if x == 0: pass", "if x == 1: pass"),
            Example("list[0]", "list[1]"),
        )


class ReplaceOneWithZero(BoundaryValueMutator):
    """Replace 1 with 0 (common boundary)."""
    
    def __init__(self):
        super().__init__(1, 0)
    
    @classmethod
    def examples(cls):
        return (
            Example("range(1, 10)", "range(0, 10)"),
            Example("if x == 1: pass", "if x == 0: pass"),
            Example("x += 1", "x += 0"),
        )


class ReplaceMinusOneWithZero(BoundaryValueMutator):
    """Replace -1 with 0 (common boundary)."""
    
    def __init__(self):
        super().__init__(-1, 0)
    
    @classmethod
    def examples(cls):
        return (
            Example("list[-1]", "list[0]"),
            Example("if x == -1: pass", "if x == 0: pass"),
            Example("return -1", "return 0"),
        )


class StringMutator(Operator):
    """Base class for string constant mutations."""
    
    def __init__(self, from_value: str = None, to_value: str = ""):
        self.from_value = from_value
        self.to_value = to_value
    
    def mutation_positions(self, node: ast.AST) -> Iterable[Tuple[int, int]]:
        """Find string constants."""
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            if self.from_value is None or node.value == self.from_value:
                yield (getattr(node, 'lineno', 0), getattr(node, 'col_offset', 0))
        elif isinstance(node, ast.Str):  # Python < 3.8
            if self.from_value is None or node.s == self.from_value:
                yield (getattr(node, 'lineno', 0), getattr(node, 'col_offset', 0))
    
    def mutate(self, node: ast.AST, index: int) -> ast.AST:
        """Replace the string value."""
        assert index == 0
        
        if isinstance(node, ast.Constant):
            node.value = self.to_value
        elif isinstance(node, ast.Str):
            node.s = self.to_value
            
        return node


class StringToEmpty(StringMutator):
    """Replace non-empty strings with empty string."""
    
    def __init__(self):
        super().__init__(to_value="")
    
    def mutation_positions(self, node: ast.AST) -> Iterable[Tuple[int, int]]:
        """Find non-empty string constants."""
        if isinstance(node, ast.Constant) and isinstance(node.value, str) and node.value != "":
            yield (getattr(node, 'lineno', 0), getattr(node, 'col_offset', 0))
        elif isinstance(node, ast.Str) and node.s != "":
            yield (getattr(node, 'lineno', 0), getattr(node, 'col_offset', 0))
    
    @classmethod
    def examples(cls):
        return (
            Example('"hello"', '""'),
            Example("'test'", "''"),
            Example('name = "John"', 'name = ""'),
        )


class StringToX(StringMutator):
    """Replace strings with 'X'."""
    
    def __init__(self):
        super().__init__(to_value="X")
    
    def mutation_positions(self, node: ast.AST) -> Iterable[Tuple[int, int]]:
        """Find string constants that aren't already 'X'."""
        if isinstance(node, ast.Constant) and isinstance(node.value, str) and node.value != "X":
            yield (getattr(node, 'lineno', 0), getattr(node, 'col_offset', 0))
        elif isinstance(node, ast.Str) and node.s != "X":
            yield (getattr(node, 'lineno', 0), getattr(node, 'col_offset', 0))
    
    @classmethod
    def examples(cls):
        return (
            Example('"hello"', '"X"'),
            Example("'message'", "'X'"),
            Example('text = "data"', 'text = "X"'),
        )
