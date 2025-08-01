"""
Collection and conditional mutation operators.

This module contains operators that mutate collections (lists, dicts) and 
conditional expressions (ternary operators).
"""

import ast
from typing import Iterable, Tuple

from ..core import Operator, Example


class CollectionMutator(Operator):
    """Base class for collection mutations."""
    
    def mutation_positions(self, node: ast.AST) -> Iterable[Tuple[int, int]]:
        """Find non-empty lists."""
        if isinstance(node, ast.List) and len(node.elts) > 0:
            yield (getattr(node, 'lineno', 0), getattr(node, 'col_offset', 0))
    
    def mutate(self, node: ast.AST, index: int) -> ast.AST:
        """Replace list with empty list."""
        assert isinstance(node, ast.List)
        assert index == 0
        
        # Create empty list
        empty_list = ast.List(elts=[], ctx=node.ctx)
        
        # Copy line information if available
        if hasattr(node, 'lineno'):
            empty_list.lineno = node.lineno
        if hasattr(node, 'col_offset'):
            empty_list.col_offset = node.col_offset
            
        return empty_list


class ListToEmpty(CollectionMutator):
    """Replace non-empty lists with empty list."""
    
    @classmethod
    def examples(cls):
        return (
            Example("[1, 2, 3]", "[]"),
            Example("items = ['a', 'b']", "items = []"),
            Example("for x in [1, 2]: pass", "for x in []: pass"),
        )


class DictToEmpty(Operator):
    """Replace non-empty dictionaries with empty dict."""
    
    def mutation_positions(self, node: ast.AST) -> Iterable[Tuple[int, int]]:
        """Find non-empty dictionaries."""
        if isinstance(node, ast.Dict) and len(node.keys) > 0:
            yield (getattr(node, 'lineno', 0), getattr(node, 'col_offset', 0))
    
    def mutate(self, node: ast.AST, index: int) -> ast.AST:
        """Replace dict with empty dict."""
        assert isinstance(node, ast.Dict)
        assert index == 0
        
        # Create empty dict
        empty_dict = ast.Dict(keys=[], values=[])
        
        # Copy line information if available
        if hasattr(node, 'lineno'):
            empty_dict.lineno = node.lineno
        if hasattr(node, 'col_offset'):
            empty_dict.col_offset = node.col_offset
            
        return empty_dict
    
    @classmethod
    def examples(cls):
        return (
            Example("{'a': 1, 'b': 2}", "{}"),
            Example("config = {'debug': True}", "config = {}"),
            Example("data = {1: 'one', 2: 'two'}", "data = {}"),
        )


class ConditionalBoundaryMutator(Operator):
    """Mutate conditional expressions (ternary operator)."""
    
    def mutation_positions(self, node: ast.AST) -> Iterable[Tuple[int, int]]:
        """Find conditional expressions."""
        if isinstance(node, ast.IfExp):
            # Two mutations: condition to True, condition to False
            yield (getattr(node, 'lineno', 0), getattr(node, 'col_offset', 0))
            yield (getattr(node, 'lineno', 0), getattr(node, 'col_offset', 0))
    
    def mutate(self, node: ast.AST, index: int) -> ast.AST:
        """Replace conditional test with True or False."""
        assert isinstance(node, ast.IfExp)
        assert index in [0, 1]
        
        # Create new conditional with constant test
        if index == 0:
            # Replace condition with True
            new_test = ast.Constant(value=True)
        else:
            # Replace condition with False  
            new_test = ast.Constant(value=False)
        
        # Copy line information if available
        if hasattr(node, 'lineno'):
            new_test.lineno = node.lineno
        if hasattr(node, 'col_offset'):
            new_test.col_offset = node.col_offset
        
        # Create new IfExp with modified test
        new_ifexp = ast.IfExp(test=new_test, body=node.body, orelse=node.orelse)
        
        # Copy line information
        if hasattr(node, 'lineno'):
            new_ifexp.lineno = node.lineno
        if hasattr(node, 'col_offset'):
            new_ifexp.col_offset = node.col_offset
            
        return new_ifexp


class ReplaceTernaryCondition(ConditionalBoundaryMutator):
    """Replace ternary operator conditions with True/False."""
    
    @classmethod
    def examples(cls):
        return (
            Example("x if condition else y", "x if True else y"),
            Example("x if condition else y", "x if False else y", occurrence=1),
            Example("result = a if test else b", "result = a if True else b"),
        )
