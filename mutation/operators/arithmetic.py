"""
Arithmetic and binary operation mutation operators.

This module contains operators that mutate arithmetic operations, binary operators,
and augmented assignment operations.
"""

import ast
from enum import Enum
from typing import Iterable, Tuple

from ..core import Operator, Example


class BinaryOperators(Enum):
    """Binary operators that can be mutated."""
    Add = ast.Add
    Sub = ast.Sub
    Mult = ast.Mult
    Div = ast.Div
    FloorDiv = ast.FloorDiv
    Mod = ast.Mod
    Pow = ast.Pow
    LShift = ast.LShift
    RShift = ast.RShift
    BitOr = ast.BitOr
    BitXor = ast.BitXor
    BitAnd = ast.BitAnd


class NumberReplacer(Operator):
    """An operator that modifies numeric constants."""
    
    # List of offsets to apply to numbers
    OFFSETS = [+1, -1]
    
    def mutation_positions(self, node: ast.AST) -> Iterable[Tuple[int, int]]:
        """Find numeric constants that can be mutated."""
        if isinstance(node, (ast.Constant, ast.Num)) and isinstance(self._get_value(node), (int, float)):
            for _ in self.OFFSETS:
                yield (getattr(node, 'lineno', 0), getattr(node, 'col_offset', 0))
    
    def mutate(self, node: ast.AST, index: int) -> ast.AST:
        """Modify the numeric value."""
        assert index < len(self.OFFSETS), f"Invalid index {index} for {len(self.OFFSETS)} offsets"
        
        if isinstance(node, ast.Constant):
            value = node.value
        elif isinstance(node, ast.Num):  # Python < 3.8 compatibility
            value = node.n
        else:
            return node
            
        if isinstance(value, (int, float)):
            new_value = value + self.OFFSETS[index]
            if isinstance(node, ast.Constant):
                node.value = new_value
            else:  # ast.Num
                node.n = new_value
                
        return node
    
    def _get_value(self, node):
        """Get the numeric value from a node."""
        if isinstance(node, ast.Constant):
            return node.value
        elif isinstance(node, ast.Num):
            return node.n
        return None
    
    @classmethod
    def examples(cls):
        return (
            Example("x = 1", "x = 2"),
            Example("x = 1", "x = 0", occurrence=1),
            Example("y = 3.14", "y = 4.14"),
        )


class BinaryOperatorReplacer(Operator):
    """Base class for binary operator replacement operators."""
    
    def __init__(self, from_op: type, to_op: type):
        self.from_op = from_op
        self.to_op = to_op
    
    def mutation_positions(self, node: ast.AST) -> Iterable[Tuple[int, int]]:
        """Find binary operations that match our target operator."""
        if isinstance(node, ast.BinOp) and isinstance(node.op, self.from_op):
            yield (getattr(node, 'lineno', 0), getattr(node, 'col_offset', 0))
    
    def mutate(self, node: ast.AST, index: int) -> ast.AST:
        """Replace the binary operator."""
        assert isinstance(node, ast.BinOp)
        assert isinstance(node.op, self.from_op)
        assert index == 0
        
        node.op = self.to_op()
        return node


# Basic arithmetic operators
class ReplaceBinaryOperator_Add_Sub(BinaryOperatorReplacer):
    """Replace + with -."""
    
    def __init__(self):
        super().__init__(ast.Add, ast.Sub)
    
    @classmethod
    def examples(cls):
        return (
            Example("x + y", "x - y"),
            Example("1 + 2 + 3", "1 - 2 + 3"),
        )


class ReplaceBinaryOperator_Sub_Add(BinaryOperatorReplacer):
    """Replace - with +."""
    
    def __init__(self):
        super().__init__(ast.Sub, ast.Add)
    
    @classmethod
    def examples(cls):
        return (
            Example("x - y", "x + y"),
            Example("10 - 5", "10 + 5"),
        )


class ReplaceBinaryOperator_Mult_Div(BinaryOperatorReplacer):
    """Replace * with /."""
    
    def __init__(self):
        super().__init__(ast.Mult, ast.Div)
    
    @classmethod
    def examples(cls):
        return (
            Example("x * y", "x / y"),
            Example("2 * 3", "2 / 3"),
        )
        

class ReplaceBinaryOperator_Div_Mult(BinaryOperatorReplacer):
    """Replace / with *."""
    
    def __init__(self):
        super().__init__(ast.Div, ast.Mult)
    
    @classmethod
    def examples(cls):
        return (
            Example("x / y", "x * y"),
            Example("10 / 2", "10 * 2"),
        )
        

class ReplaceBinaryOperator_FloorDiv_Mod(BinaryOperatorReplacer):
    """Replace // with %."""
    
    def __init__(self):
        super().__init__(ast.FloorDiv, ast.Mod)
    
    @classmethod
    def examples(cls):
        return (
            Example("x // y", "x % y"),
            Example("10 // 3", "10 % 3"),
        )
        

class ReplaceBinaryOperator_Mod_FloorDiv(BinaryOperatorReplacer):
    """Replace % with //."""
    
    def __init__(self):
        super().__init__(ast.Mod, ast.FloorDiv)
    
    @classmethod
    def examples(cls):
        return (
            Example("x % y", "x // y"),
            Example("10 % 3", "10 // 3"),
        )


class ReplaceBinaryOperator_Pow_Mult(BinaryOperatorReplacer):
    """Replace ** with *."""
    
    def __init__(self):
        super().__init__(ast.Pow, ast.Mult)
    
    @classmethod
    def examples(cls):
        return (
            Example("x ** y", "x * y"),
            Example("2 ** 3", "2 * 3"),
        )
        

class ReplaceBinaryOperator_Mult_Pow(BinaryOperatorReplacer):
    """Replace * with **."""
    
    def __init__(self):
        super().__init__(ast.Mult, ast.Pow)
    
    @classmethod
    def examples(cls):
        return (
            Example("x * y", "x ** y"),
            Example("2 * 3", "2 ** 3"),
        )


# Bitwise operators
class ReplaceBinaryOperator_LShift_RShift(BinaryOperatorReplacer):
    """Replace << with >>."""
    
    def __init__(self):
        super().__init__(ast.LShift, ast.RShift)
    
    @classmethod
    def examples(cls):
        return (
            Example("x << y", "x >> y"),
            Example("10 << 2", "10 >> 2"),
        )
        

class ReplaceBinaryOperator_RShift_LShift(BinaryOperatorReplacer):
    """Replace >> with <<."""
    
    def __init__(self):
        super().__init__(ast.RShift, ast.LShift)
    
    @classmethod
    def examples(cls):
        return (
            Example("x >> y", "x << y"),
            Example("10 >> 2", "10 << 2"),
        )
        

class ReplaceBinaryOperator_BitOr_BitXor(BinaryOperatorReplacer):
    """Replace | with ^."""
    
    def __init__(self):
        super().__init__(ast.BitOr, ast.BitXor)
    
    @classmethod
    def examples(cls):
        return (
            Example("x | y", "x ^ y"),
            Example("10 | 2", "10 ^ 2"),
        )   
        

class ReplaceBinaryOperator_BitXor_BitOr(BinaryOperatorReplacer):
    """Replace ^ with |."""
    
    def __init__(self):
        super().__init__(ast.BitXor, ast.BitOr)
    
    @classmethod
    def examples(cls):
        return (
            Example("x ^ y", "x | y"),
            Example("10 ^ 2", "10 | 2"),
        )
        

class ReplaceBinaryOperator_BitAnd_BitXor(BinaryOperatorReplacer):
    """Replace & with ^."""
    
    def __init__(self):
        super().__init__(ast.BitAnd, ast.BitXor)
    
    @classmethod
    def examples(cls):
        return (
            Example("x & y", "x ^ y"),
            Example("10 & 2", "10 ^ 2"),
        )
        

class ReplaceBinaryOperator_BitXor_BitAnd(BinaryOperatorReplacer):
    """Replace ^ with &."""
    
    def __init__(self):
        super().__init__(ast.BitXor, ast.BitAnd)
    
    @classmethod
    def examples(cls):
        return (
            Example("x ^ y", "x & y"),
            Example("10 ^ 2", "10 & 2"),
        )
        

class ReplaceBinaryOperator_BitOr_BitAnd(BinaryOperatorReplacer):
    """Replace | with &."""
    
    def __init__(self):
        super().__init__(ast.BitOr, ast.BitAnd)
    
    @classmethod
    def examples(cls):
        return (
            Example("x | y", "x & y"),
            Example("10 | 2", "10 & 2"),
        )


class ReplaceBinaryOperator_BitAnd_BitOr(BinaryOperatorReplacer):
    """Replace & with |."""
    
    def __init__(self):
        super().__init__(ast.BitAnd, ast.BitOr)
    
    @classmethod
    def examples(cls):
        return (
            Example("x & y", "x | y"),
            Example("10 & 2", "10 | 2"),
        )


# Augmented assignment operators
class AugAssignOperatorReplacer(Operator):
    """Base class for augmented assignment operator replacement."""
    
    def __init__(self, from_op: type, to_op: type):
        self.from_op = from_op
        self.to_op = to_op
    
    def mutation_positions(self, node: ast.AST) -> Iterable[Tuple[int, int]]:
        """Find augmented assignments that match our target operator."""
        if isinstance(node, ast.AugAssign) and isinstance(node.op, self.from_op):
            yield (getattr(node, 'lineno', 0), getattr(node, 'col_offset', 0))
    
    def mutate(self, node: ast.AST, index: int) -> ast.AST:
        """Replace the augmented assignment operator."""
        assert isinstance(node, ast.AugAssign)
        assert isinstance(node.op, self.from_op)
        assert index == 0
        
        node.op = self.to_op()
        return node


class ReplaceAugAssign_AddEq_SubEq(AugAssignOperatorReplacer):
    """Replace += with -=."""
    
    def __init__(self):
        super().__init__(ast.Add, ast.Sub)
    
    @classmethod
    def examples(cls):
        return (
            Example("x += 1", "x -= 1"),
            Example("count += step", "count -= step"),
        )


class ReplaceAugAssign_SubEq_AddEq(AugAssignOperatorReplacer):
    """Replace -= with +=."""
    
    def __init__(self):
        super().__init__(ast.Sub, ast.Add)
    
    @classmethod
    def examples(cls):
        return (
            Example("x -= 1", "x += 1"),
            Example("count -= step", "count += step"),
        )


class ReplaceAugAssign_MulEq_DivEq(AugAssignOperatorReplacer):
    """Replace *= with /=."""
    
    def __init__(self):
        super().__init__(ast.Mult, ast.Div)
    
    @classmethod
    def examples(cls):
        return (
            Example("x *= 2", "x /= 2"),
            Example("value *= factor", "value /= factor"),
        )


class ReplaceAugAssign_DivEq_MulEq(AugAssignOperatorReplacer):
    """Replace /= with *=."""
    
    def __init__(self):
        super().__init__(ast.Div, ast.Mult)
    
    @classmethod
    def examples(cls):
        return (
            Example("x /= 2", "x *= 2"),
            Example("value /= factor", "value *= factor"),
        )
