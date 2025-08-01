"""
Mutation operators package.

This package contains all the specific mutation operators organized by category:
- arithmetic: Basic arithmetic and binary operations
- comparison: Comparison operators  
- logical: Boolean and logical operations
- control_flow: Break, continue, return mutations
- values: Number, string, boolean constant mutations
- collections: List, dict mutations
- exceptions: Exception handling mutations
"""

# Import all operators for easy access
from .arithmetic import *
from .comparison import *
from .logical import *
from .control_flow import *
from .values import *
from .collections import *
from .exceptions import *

# Export all operator classes
__all__ = [
    # Arithmetic operators
    'NumberReplacer',
    'BinaryOperatorReplacer',
    'ReplaceBinaryOperator_Add_Sub',
    'ReplaceBinaryOperator_Sub_Add',
    'ReplaceBinaryOperator_Mult_Div',
    'ReplaceBinaryOperator_Div_Mult',
    'ReplaceBinaryOperator_FloorDiv_Mod',
    'ReplaceBinaryOperator_Mod_FloorDiv',
    'ReplaceBinaryOperator_Pow_Mult',
    'ReplaceBinaryOperator_Mult_Pow',
    'ReplaceBinaryOperator_LShift_RShift',
    'ReplaceBinaryOperator_RShift_LShift',
    'ReplaceBinaryOperator_BitOr_BitXor',
    'ReplaceBinaryOperator_BitXor_BitOr',
    'ReplaceBinaryOperator_BitAnd_BitXor',
    'ReplaceBinaryOperator_BitXor_BitAnd',
    'ReplaceBinaryOperator_BitOr_BitAnd',
    'ReplaceBinaryOperator_BitAnd_BitOr',
    'AugAssignOperatorReplacer',
    'ReplaceAugAssign_AddEq_SubEq',
    'ReplaceAugAssign_SubEq_AddEq',
    'ReplaceAugAssign_MulEq_DivEq',
    'ReplaceAugAssign_DivEq_MulEq',
    
    # Comparison operators
    'ComparisonOperatorReplacer',
    'ReplaceEqWithNotEq',
    'ReplaceLtWithGt',
    'ReplaceGtWithLt',
    'ReplaceLtEqWithGtEq',
    'ReplaceGtEqWithLtEq',
    'ReplaceNotEqWithEq',
    
    # Logical operators
    'LogicalOperatorReplacer',
    'ReplaceAndWithOr',
    'ReplaceOrWithAnd',
    'UnaryOperatorRemover',
    'RemoveNotOperator',
    'RemoveUnaryMinus',
    'RemoveUnaryPlus',
    
    # Control flow
    'ControlFlowMutator',
    'ReplaceBreakWithContinue',
    'ReplaceContinueWithBreak',
    'ReturnStatementMutator',
    'RemoveReturnValue',
    
    # Values
    'BooleanReplacer',
    'ReplaceTrueWithFalse',
    'ReplaceFalseWithTrue',
    'BoundaryValueMutator',
    'ReplaceZeroWithOne',
    'ReplaceOneWithZero',
    'ReplaceMinusOneWithZero',
    'StringMutator',
    'StringToEmpty',
    'StringToX',
    
    # Collections
    'CollectionMutator',
    'ListToEmpty',
    'DictToEmpty',
    'ConditionalBoundaryMutator',
    'ReplaceTernaryCondition',
    
    # Exceptions
    'ExceptionMutator',
    'ReplaceRaiseWithPass',
    
    # Enums and utilities
    'BinaryOperators',
]
