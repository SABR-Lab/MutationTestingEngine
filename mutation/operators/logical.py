"""
Logical operation mutation operators.

This module contains operators that mutate logical operations like and, or, not, unary operators.
"""

import ast
from typing import Iterable, Tuple

from ..core import Operator, Example


class LogicalOperatorReplacer(Operator):
    """Base class for logical operator replacement."""
    
    def __init__(self, from_op: type, to_op: type):
        self.from_op = from_op
        self.to_op = to_op
    
    def mutation_positions(self, node: ast.AST) -> Iterable[Tuple[int, int]]:
        """Find logical operations that match our target operator."""
        if isinstance(node, ast.BoolOp) and isinstance(node.op, self.from_op):
            yield (getattr(node, 'lineno', 0), getattr(node, 'col_offset', 0))
    
    def mutate(self, node: ast.AST, index: int) -> ast.AST:
        """Replace the logical operator."""
        assert isinstance(node, ast.BoolOp)
        assert isinstance(node.op, self.from_op)
        assert index == 0
        
        node.op = self.to_op()
        return node


class ReplaceAndWithOr(LogicalOperatorReplacer):
    """Replace 'and' with 'or'."""
    
    def __init__(self):
        super().__init__(ast.And, ast.Or)
    
    @classmethod
    def examples(cls):
        return (
            Example("x and y", "x or y"),
            Example("if a and b: pass", "if a or b: pass"),
        )


class ReplaceOrWithAnd(LogicalOperatorReplacer):
    """Replace 'or' with 'and'."""
    
    def __init__(self):
        super().__init__(ast.Or, ast.And)
    
    @classmethod
    def examples(cls):
        return (
            Example("x or y", "x and y"),
            Example("if a or b: pass", "if a and b: pass"),
        )


class UnaryOperatorRemover(Operator):
    """Base class for removing unary operators."""
    
    def __init__(self, target_op: type):
        self.target_op = target_op
    
    def mutation_positions(self, node: ast.AST) -> Iterable[Tuple[int, int]]:
        """Find unary operations that match our target operator."""
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, self.target_op):
            yield (getattr(node, 'lineno', 0), getattr(node, 'col_offset', 0))
    
    def mutate(self, node: ast.AST, index: int) -> ast.AST:
        """Remove the unary operator, returning just the operand."""
        assert isinstance(node, ast.UnaryOp)
        assert isinstance(node.op, self.target_op)
        assert index == 0
        
        # Return the operand without the unary operator
        return node.operand


class RemoveNotOperator(UnaryOperatorRemover):
    """Remove 'not' operator."""
    
    def __init__(self):
        super().__init__(ast.Not)
    
    @classmethod
    def examples(cls):
        return (
            Example("not x", "x"),
            Example("if not condition: pass", "if condition: pass"),
            Example("not (a and b)", "(a and b)"),
        )


class RemoveUnaryMinus(UnaryOperatorRemover):
    """Remove unary minus operator."""
    
    def __init__(self):
        super().__init__(ast.USub)
    
    @classmethod
    def examples(cls):
        return (
            Example("-x", "x"),
            Example("y = -5", "y = 5"),
            Example("return -result", "return result"),
        )


class RemoveUnaryPlus(UnaryOperatorRemover):
    """Remove unary plus operator."""
    
    def __init__(self):
        super().__init__(ast.UAdd)
    
    @classmethod
    def examples(cls):
        return (
            Example("+x", "x"),
            Example("y = +5", "y = 5"),
            Example("return +result", "return result"),
        )
