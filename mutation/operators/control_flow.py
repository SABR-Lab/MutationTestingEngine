"""
Control flow mutation operators.

This module contains operators that mutate control flow statements like break, continue, return.
"""

import ast
from typing import Iterable, Tuple

from ..core import Operator, Example


class ControlFlowMutator(Operator):
    """Base class for control flow statement mutations."""
    
    def __init__(self, from_stmt: type, to_stmt: type):
        self.from_stmt = from_stmt
        self.to_stmt = to_stmt
    
    def mutation_positions(self, node: ast.AST) -> Iterable[Tuple[int, int]]:
        """Find control flow statements that match our target."""
        if isinstance(node, self.from_stmt):
            yield (getattr(node, 'lineno', 0), getattr(node, 'col_offset', 0))
    
    def mutate(self, node: ast.AST, index: int) -> ast.AST:
        """Replace the control flow statement."""
        assert isinstance(node, self.from_stmt)
        assert index == 0
        
        # Create new statement of target type
        new_stmt = self.to_stmt()
        
        # Copy line information if available
        if hasattr(node, 'lineno'):
            new_stmt.lineno = node.lineno
        if hasattr(node, 'col_offset'):
            new_stmt.col_offset = node.col_offset
            
        return new_stmt


class ReplaceBreakWithContinue(ControlFlowMutator):
    """Replace 'break' with 'continue'."""
    
    def __init__(self):
        super().__init__(ast.Break, ast.Continue)
    
    @classmethod
    def examples(cls):
        return (
            Example("for x in items:\n    if condition:\n        break", 
                   "for x in items:\n    if condition:\n        continue"),
            Example("while True:\n    if done:\n        break", 
                   "while True:\n    if done:\n        continue"),
        )


class ReplaceContinueWithBreak(ControlFlowMutator):
    """Replace 'continue' with 'break'."""
    
    def __init__(self):
        super().__init__(ast.Continue, ast.Break)
    
    @classmethod
    def examples(cls):
        return (
            Example("for x in items:\n    if skip:\n        continue", 
                   "for x in items:\n    if skip:\n        break"),
            Example("while True:\n    if not ready:\n        continue", 
                   "while True:\n    if not ready:\n        break"),
        )


class ReturnStatementMutator(Operator):
    """Base class for return statement mutations."""
    
    def mutation_positions(self, node: ast.AST) -> Iterable[Tuple[int, int]]:
        """Find return statements with values."""
        if isinstance(node, ast.Return) and node.value is not None:
            yield (getattr(node, 'lineno', 0), getattr(node, 'col_offset', 0))
    
    def mutate(self, node: ast.AST, index: int) -> ast.AST:
        """Remove the return value."""
        assert isinstance(node, ast.Return)
        assert index == 0
        
        # Create return statement without value
        new_return = ast.Return(value=None)
        
        # Copy line information if available
        if hasattr(node, 'lineno'):
            new_return.lineno = node.lineno
        if hasattr(node, 'col_offset'):
            new_return.col_offset = node.col_offset
            
        return new_return


class RemoveReturnValue(ReturnStatementMutator):
    """Remove return values from functions."""
    
    @classmethod
    def examples(cls):
        return (
            Example("return result", "return"),
            Example("return x + y", "return"),
            Example("return True", "return"),
        )
