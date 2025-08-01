"""
Core mutation framework classes and infrastructure.

This module contains the base classes and core infrastructure for the mutation
testing framework, including abstract base classes for visitors and operators.
"""

import ast
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Iterable, Tuple


@dataclass(frozen=True)
class Example:
    """A structure to store pre and post mutation operator code snippets."""
    pre_mutation_code: str
    post_mutation_code: str
    occurrence: int = 0
    operator_args: dict = None
    
    def __post_init__(self):
        if self.operator_args is None:
            object.__setattr__(self, 'operator_args', {})


class Visitor(ABC):
    """AST visitor for Python AST nodes."""
    
    def walk(self, node):
        """Walk an AST, calling visit for each node."""
        node = self.visit(node)
        
        if node is None:
            return None
            
        # Visit all child nodes
        for field, value in ast.iter_fields(node):
            if isinstance(value, list):
                new_list = []
                for item in value:
                    if isinstance(item, ast.AST):
                        new_item = self.walk(item)
                        if new_item is not None:
                            new_list.append(new_item)
                    else:
                        new_list.append(item)
                setattr(node, field, new_list)
            elif isinstance(value, ast.AST):
                new_node = self.walk(value)
                setattr(node, field, new_node)
                
        return node
    
    @abstractmethod
    def visit(self, node):
        """Called for each node in the walk.
        
        Should return a node that will replace the node argument in the AST.
        Can be the node itself, a new node, or None to remove the node.
        """
        pass


class Operator(ABC):
    """Base class for mutation operators."""
    
    @abstractmethod
    def mutation_positions(self, node: ast.AST) -> Iterable[Tuple[int, int]]:
        """Find all positions where this operator can mutate the node.
        
        Args:
            node: The AST node being examined
            
        Yields:
            Tuples of (start_pos, end_pos) where mutations can be applied
        """
        pass
    
    @abstractmethod
    def mutate(self, node: ast.AST, index: int) -> ast.AST:
        """Mutate a node in an operator-specific manner.
        
        Args:
            node: The AST node to mutate
            index: The index of the mutation to apply
            
        Returns:
            The mutated node, or None to remove the node
        """
        pass
    
    @classmethod
    def examples(cls):
        """Examples of mutations this operator can make."""
        return ()


class MutationPositionFinder(Visitor):
    """Visitor that counts all possible mutation positions for an operator."""
    
    def __init__(self, operator: Operator):
        self.operator = operator
        self.total_positions = 0
        
    def visit(self, node):
        # Count all positions where this operator can mutate
        for _ in self.operator.mutation_positions(node):
            self.total_positions += 1
        return node


class StandaloneMutationVisitor(Visitor):
    """Visitor that applies a specific mutation occurrence to an AST."""
    
    def __init__(self, occurrence: int, operator: Operator):
        self.operator = operator
        self._occurrence = occurrence
        self._count = 0
        self._mutation_applied = False

    @property
    def mutation_applied(self) -> bool:
        """Whether this visitor has applied a mutation."""
        return self._mutation_applied

    def visit(self, node):
        """Visit a node and potentially apply a mutation."""
        for index, _ in enumerate(self.operator.mutation_positions(node)):
            if self._count == self._occurrence:
                self._mutation_applied = True
                node = self.operator.mutate(node, index)
            self._count += 1

        return node
