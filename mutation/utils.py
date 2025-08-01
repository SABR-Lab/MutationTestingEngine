"""
Utility functions for AST manipulation and code conversion.

This module provides helper functions for working with Python ASTs,
including parsing source code and converting ASTs back to source code.
"""

import ast
import textwrap
import re


def get_ast(source_code: str) -> ast.AST:
    """Parse source code into an AST, handling indentation automatically."""
    # Remove common leading whitespace to handle indented code blocks
    dedented_code = textwrap.dedent(source_code)
    return ast.parse(dedented_code)


def get_indentation_info(source_code: str) -> tuple[str, str]:
    """
    Extract indentation information from source code.
    
    Returns:
        tuple: (common_indent, dedented_code)
            - common_indent: the common leading whitespace that was removed
            - dedented_code: the code with common indentation removed
    """
    lines = source_code.splitlines()
    
    # Find the minimum indentation of non-empty lines
    indents = []
    for line in lines:
        if line.strip():  # Skip empty lines
            indent = len(line) - len(line.lstrip())
            indents.append(indent)
    
    if not indents:
        return "", source_code
    
    min_indent = min(indents)
    common_indent = " " * min_indent
    
    # Remove the common indentation
    dedented_lines = []
    for line in lines:
        if line.strip():  # Non-empty line
            dedented_lines.append(line[min_indent:])
        else:  # Empty line
            dedented_lines.append("")
    
    dedented_code = "\n".join(dedented_lines)
    return common_indent, dedented_code


def ast_to_code(node: ast.AST, preserve_indent: str = "") -> str:
    """
    Convert an AST back to source code with optional indentation preservation.
    
    Args:
        node: The AST node to convert
        preserve_indent: Optional indentation to add to each line
        
    Returns:
        The source code as a string
    """
    # Try to use astor first, fallback to ast.unparse
    try:
        import astor
        code = astor.to_source(node).rstrip()
    except ImportError:
        try:
            code = ast.unparse(node)
        except AttributeError:
            # For Python < 3.9, provide a basic implementation
            import sys
            if sys.version_info >= (3, 9):
                code = ast.unparse(node)
            else:
                raise ImportError("astor package required for Python < 3.9, or use Python 3.9+")
    
    # Apply preserved indentation if provided
    if preserve_indent:
        lines = code.splitlines()
        indented_lines = []
        for line in lines:
            if line.strip():  # Non-empty line
                indented_lines.append(preserve_indent + line)
            else:  # Empty line
                indented_lines.append("")
        code = "\n".join(indented_lines)
    
    return code


def get_ast_with_indent_info(source_code: str) -> tuple[ast.AST, str]:
    """
    Parse source code and return both AST and indentation info.
    
    Returns:
        tuple: (ast_tree, common_indent)
    """
    common_indent, dedented_code = get_indentation_info(source_code)
    ast_tree = ast.parse(dedented_code)
    return ast_tree, common_indent