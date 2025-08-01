"""
Comparison operation mutation operators.

This module contains operators that mutate comparison operations like ==, !=, <, >, etc.
"""

import ast
from typing import Iterable, Tuple

from ..core import Operator, Example


class ComparisonOperatorReplacer(Operator):
    """Replace comparison operators."""
    
    def __init__(self, from_op: type, to_op: type):
        self.from_op = from_op
        self.to_op = to_op
    
    def mutation_positions(self, node: ast.AST) -> Iterable[Tuple[int, int]]:
        """Find comparison operations."""
        if isinstance(node, ast.Compare):
            for op in node.ops:
                if isinstance(op, self.from_op):
                    yield (getattr(node, 'lineno', 0), getattr(node, 'col_offset', 0))
    
    def mutate(self, node: ast.AST, index: int) -> ast.AST:
        """Replace the comparison operator."""
        assert isinstance(node, ast.Compare)
        
        # Find and replace the first matching operator
        for i, op in enumerate(node.ops):
            if isinstance(op, self.from_op):
                if index == 0:
                    node.ops[i] = self.to_op()
                    break
                index -= 1
                
        return node


class ReplaceEqWithNotEq(ComparisonOperatorReplacer):
    """Replace == with !=."""
    
    def __init__(self):
        super().__init__(ast.Eq, ast.NotEq)
    
    @classmethod
    def examples(cls):
        return (
            Example("x == y", "x != y"),
            Example("if a == b: pass", "if a != b: pass"),
        )


class ReplaceNotEqWithEq(ComparisonOperatorReplacer):
    """Replace != with ==."""
    
    def __init__(self):
        super().__init__(ast.NotEq, ast.Eq)
    
    @classmethod
    def examples(cls):
        return (
            Example("x != y", "x == y"),
            Example("if a != b: pass", "if a == b: pass"),
        )


class ReplaceLtWithGt(ComparisonOperatorReplacer):
    """Replace < with >."""
    
    def __init__(self):
        super().__init__(ast.Lt, ast.Gt)
    
    @classmethod
    def examples(cls):
        return (
            Example("x < y", "x > y"),
            Example("if a < 10: pass", "if a > 10: pass"),
        )


class ReplaceGtWithLt(ComparisonOperatorReplacer):
    """Replace > with <."""
    
    def __init__(self):
        super().__init__(ast.Gt, ast.Lt)
    
    @classmethod
    def examples(cls):
        return (
            Example("x > y", "x < y"),
            Example("if a > 10: pass", "if a < 10: pass"),
        )


class ReplaceLtEqWithGtEq(ComparisonOperatorReplacer):
    """Replace <= with >=."""
    
    def __init__(self):
        super().__init__(ast.LtE, ast.GtE)
    
    @classmethod
    def examples(cls):
        return (
            Example("x <= y", "x >= y"),
            Example("if a <= 10: pass", "if a >= 10: pass"),
        )


class ReplaceGtEqWithLtEq(ComparisonOperatorReplacer):
    """Replace >= with <=."""
    
    def __init__(self):
        super().__init__(ast.GtE, ast.LtE)
    
    @classmethod
    def examples(cls):
        return (
            Example("x >= y", "x <= y"),
            Example("if a >= 10: pass", "if a <= 10: pass"),
        )
