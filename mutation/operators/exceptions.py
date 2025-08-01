"""
Exception handling mutation operators.

This module contains operators that mutate exception handling statements 
like raise statements.
"""

import ast
from typing import Iterable, Tuple

from ..core import Operator, Example


class ExceptionMutator(Operator):
    """Base class for exception handling mutations."""
    
    def mutation_positions(self, node: ast.AST) -> Iterable[Tuple[int, int]]:
        """Find raise statements."""
        if isinstance(node, ast.Raise):
            yield (getattr(node, 'lineno', 0), getattr(node, 'col_offset', 0))
    
    def mutate(self, node: ast.AST, index: int) -> ast.AST:
        """Replace raise with pass."""
        assert isinstance(node, ast.Raise)
        assert index == 0
        
        # Create pass statement
        pass_stmt = ast.Pass()
        
        # Copy line information if available
        if hasattr(node, 'lineno'):
            pass_stmt.lineno = node.lineno
        if hasattr(node, 'col_offset'):
            pass_stmt.col_offset = node.col_offset
            
        return pass_stmt


class ReplaceRaiseWithPass(ExceptionMutator):
    """Replace raise statements with pass."""
    
    @classmethod
    def examples(cls):
        return (
            Example("raise ValueError('error')", "pass"),
            Example("raise Exception()", "pass"),
            Example("if error:\n    raise RuntimeError()", "if error:\n    pass"),
        )
