"""
Main mutation engine for applying mutations to Python source code.

This module provides the core MutationEngine class that orchestrates the
mutation process, including finding mutation positions, applying mutations,
and generating mutated code variants.
"""

import logging
from pathlib import Path
from typing import Generator, Tuple, Optional, Union

from .core import Operator, MutationPositionFinder, StandaloneMutationVisitor
from .utils import get_ast, ast_to_code


class MutationEngine:
    """
    Main engine for generating and applying mutations to Python source code.
    
    This class provides a high-level interface for mutation testing operations,
    encapsulating the logic for finding mutation positions, applying mutations,
    and managing the mutation process.
    """
    
    def __init__(self):
        """Initialize the mutation engine."""
        pass
    
    def parse_and_yield_mutations(
        self, 
        source: Union[str, Path], 
        operator: Operator
    ) -> Generator[Tuple[str, Optional[str], int], None, None]:
        """Parse code and yield all possible mutations for a given operator.
        
        Args:
            source: Either a string containing Python source code or a Path to a Python file
            operator: An instance of a mutation operator (subclass of Operator)
            
        Yields:
            Tuples of (original_code, mutated_code, occurrence) where:
            - original_code: The original source code as a string
            - mutated_code: The mutated code as a string, or None if no mutation was applied
            - occurrence: The occurrence index (0-based) of the mutation
        """
        # Handle both string and Path inputs
        if isinstance(source, Path):
            original_code = source.read_text(encoding='utf-8')
        else:
            original_code = source
            
        # Find all possible mutation positions
        tree = get_ast(original_code)
        position_finder = MutationPositionFinder(operator)
        position_finder.walk(tree)
        
        total_positions = position_finder.total_positions
        log.info("Found %d possible mutation positions for operator %s", 
                 total_positions, operator.__class__.__name__)
        
        # Generate mutations for each occurrence
        for occurrence in range(total_positions):
            mutated_code = self._apply_mutation_to_code(original_code, operator, occurrence)
            yield original_code, mutated_code, occurrence
    
    def get_mutation_count(self, source: Union[str, Path], operator: Operator) -> int:
        """Get the total number of possible mutations for a given operator and source code.
        
        Args:
            source: Either a string containing Python source code or a Path to a Python file
            operator: An instance of a mutation operator (subclass of Operator)
            
        Returns:
            The total number of possible mutations
        """
        # Handle both string and Path inputs
        if isinstance(source, Path):
            original_code = source.read_text(encoding='utf-8')
        else:
            original_code = source
            
        # Find all possible mutation positions
        tree = get_ast(original_code)
        position_finder = MutationPositionFinder(operator)
        position_finder.walk(tree)
        
        return position_finder.total_positions
    
    def apply_single_mutation(
        self, 
        source: Union[str, Path], 
        operator: Operator, 
        occurrence: int
    ) -> Tuple[str, Optional[str]]:
        """Apply a single mutation to source code.
        
        Args:
            source: Either a string containing Python source code or a Path to a Python file
            operator: An instance of a mutation operator (subclass of Operator)
            occurrence: The occurrence index (0-based) of the mutation to apply
            
        Returns:
            A tuple of (original_code, mutated_code) where mutated_code is None if
            no mutation was applied
        """
        # Handle both string and Path inputs
        if isinstance(source, Path):
            original_code = source.read_text(encoding='utf-8')
        else:
            original_code = source
            
        mutated_code = self._apply_mutation_to_code(original_code, operator, occurrence)
        return original_code, mutated_code
    
    def _apply_mutation_to_code(self, source_code: str, operator: Operator, occurrence: int) -> Optional[str]:
        """Apply a specific mutation to source code.
        
        Args:
            source_code: The source code to mutate
            operator: The operator instance to use
            occurrence: The occurrence index of the mutation to apply
            
        Returns:
            The mutated code, or None if no mutation was applied
        """
        tree = get_ast(source_code)
        visitor = StandaloneMutationVisitor(occurrence, operator)
        mutated_tree = visitor.walk(tree)
        
        if not visitor.mutation_applied:
            return None
            
        return ast_to_code(mutated_tree)


# Convenience functions for backward compatibility
def parse_and_yield_mutations(
    source: Union[str, Path], 
    operator: Operator
) -> Generator[Tuple[str, Optional[str], int], None, None]:
    """Convenience function that creates a MutationEngine and calls parse_and_yield_mutations."""
    engine = MutationEngine()
    return engine.parse_and_yield_mutations(source, operator)


def get_mutation_count(source: Union[str, Path], operator: Operator) -> int:
    """Convenience function that creates a MutationEngine and calls get_mutation_count."""
    engine = MutationEngine()
    return engine.get_mutation_count(source, operator)


def apply_single_mutation(
    source: Union[str, Path], 
    operator: Operator, 
    occurrence: int
) -> Tuple[str, Optional[str]]:
    """Convenience function that creates a MutationEngine and calls apply_single_mutation."""
    engine = MutationEngine()
    return engine.apply_single_mutation(source, operator, occurrence)
