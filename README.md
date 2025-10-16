# DJ Mutation Engine

A modular and extensible mutation testing framework for Python code, featuring integration with the HumanEval dataset for comprehensive code quality assessment.

## Features

- **Multiple Mutation Operators**: Support for numeric, binary operator, boolean, and comparison mutations
- **HumanEval Integration**: Built-in support for testing against the OpenAI HumanEval dataset
- **Comprehensive Testing**: Both individual and batch testing capabilities
- **Mutation Scoring**: Detailed metrics on mutation test effectiveness

## Project Structure

```
dj-mutation-engine/
├── main.py                      # Main entry point with demos
├── simple_humaneval_demo.py     # Simple demo with embedded problems
├── humaneval_tester.py          # Full HumanEval dataset integration
├── requirements.txt             # Dependencies
├── mutation/                    # Core mutation engine package
│   ├── __init__.py             # Package exports
│   ├── core.py                 # Base classes and infrastructure
│   ├── operators.py            # Concrete mutation operators
│   ├── engine.py               # Main mutation engine
│   └── utils.py                # Utility functions
└── README.md                   # This file
```

## Installation

0. You need to have UV installed [https://docs.astral.sh/uv/guides/install-python]

2. Clone the repository:
```bash
git clone <repository-url>
cd dj-mutation-engine
```

2. Install dependencies (optional):
```bash
source .venv/bin/activate
uv sync 
```

## Quick Start

### Basic Mutation Testing

```python
from mutation import MutationEngine, NumberReplacer

# Create engine and operator
engine = MutationEngine()
operator = NumberReplacer()

# Test code
code = """
def add_numbers(a, b):
    return a + b + 1
"""

# Get mutation count
count = engine.get_mutation_count(code, operator)
print(f"Found {count} possible mutations")

# Generate mutations
for original, mutated, occurrence in engine.parse_and_yield_mutations(code, operator):
    if mutated:
        print(f"Mutation {occurrence}:")
        print(mutated)
```

### Running Demos

#### 1. Basic Demo
```bash
python main.py
```

#### 2. Comprehensive Demo
```bash
python main.py --comprehensive
```

#### 3. Simple HumanEval Demo
```bash
# Test specific problem
python simple_humaneval_demo.py 0

# Run full demo
python simple_humaneval_demo.py
```

### HumanEval Integration

#### Download Dataset
```bash
python humaneval_tester.py --download
```

#### Test Single Problem
```bash
# Test problem 0
python humaneval_tester.py --test-one 0
```

#### Run Mutation Testing
```bash
# Run mutation testing on problem 0
python humaneval_tester.py --mutate 0
```

#### Test All Problems
```bash
# Test first 5 problems
python humaneval_tester.py --test-all --max-problems 5

# Test all problems (164 total)
python humaneval_tester.py --test-all
```

## Available Mutation Operators

### Numeric Mutations
- **NumberReplacer**: Modifies numeric constants by ±1

### Binary Operator Mutations
- **ReplaceBinaryOperator_Add_Sub**: Changes + to -
- **ReplaceBinaryOperator_Sub_Add**: Changes - to +
- **ReplaceBinaryOperator_Mult_Div**: Changes * to /

### Boolean Mutations
- **ReplaceTrueWithFalse**: Changes True to False
- **ReplaceFalseWithTrue**: Changes False to True

### Comparison Mutations
- **ReplaceEqWithNotEq**: Changes == to !=
- **ReplaceLtWithGt**: Changes < to >

### Control Flow Mutations
- **ReplaceBreakWithContinue**: Changes break to continue
- **ReplaceContinueWithBreak**: Changes continue to break
- **RemoveReturnValue**: Remove return

### Exception Mutations
- **ReplaceRaiseWithPass**: Changes Raise With Pass

### Logical Mutations
- **ReplaceAndWithOr**: Changes `x and y` to `x or y`
- **ReplaceOrWithAnd**: Changes `x or y` to `x and y`
- **RemoveNotOperator**: Changes not x to x
- **RemoveUnaryMinus**: Changes +5 to -5 
- **RemoveUnaryPlus**: Changes +5 to 5

## API Reference

### MutationEngine

The main class for orchestrating mutations:

```python
engine = MutationEngine()

# Count possible mutations
count = engine.get_mutation_count(source_code, operator)

# Generate all mutations
for original, mutated, occurrence in engine.parse_and_yield_mutations(source_code, operator):
    # Process mutations
    pass

# Apply single mutation
original, mutated = engine.apply_single_mutation(source_code, operator, occurrence_index)
```

### Creating Custom Operators

```python
from mutation import Operator
import ast

class MyCustomOperator(Operator):
    def mutation_positions(self, node):
        # Find nodes that can be mutated
        if isinstance(node, ast.SomeNodeType):
            yield (node.lineno, node.col_offset)
    
    def mutate(self, node, index):
        # Apply mutation to node
        # Return modified node
        return node
    
    @classmethod
    def examples(cls):
        return (
            Example("original code", "mutated code"),
        )
```

## Examples

### Example 1: Basic Number Mutation
```python
from mutation import MutationEngine, NumberReplacer

code = "x = 42"
engine = MutationEngine()
operator = NumberReplacer()

for original, mutated, occurrence in engine.parse_and_yield_mutations(code, operator):
    print(f"Original: {original}")
    print(f"Mutated:  {mutated}")
    # Output: x = 43, x = 41
```

### Example 2: Testing Function Robustness
```python
from mutation import MutationEngine, ReplaceBinaryOperator_Add_Sub

def test_function():
    code = """
def calculate_total(price, tax):
    return price + tax
"""
    
    engine = MutationEngine()
    operator = ReplaceBinaryOperator_Add_Sub()
    
    # This will change + to -, helping test if your tests catch the error
    original, mutated = engine.apply_single_mutation(code, operator, 0)
    print(mutated)
    # Output: return price - tax
```

## Mutation Testing Workflow

1. **Start with working code** that passes all tests
2. **Apply mutations** using various operators
3. **Run tests** against mutated code
4. **Count "killed" mutations** (tests that fail with mutated code)
5. **Calculate mutation score**: killed_mutations / total_mutations
6. **Improve tests** for mutations that survive

A high mutation score indicates robust test coverage.

## Contributing

To add new mutation operators:

1. Create a new class inheriting from `Operator` in `mutation/operators.py`
2. Implement `mutation_positions()` and `mutate()` methods
3. Add the operator to `mutation/__init__.py`
4. Add examples and tests

## Dependencies

- Python 3.8+
- `astor` (optional, for better AST-to-code conversion)
- Standard library only for core functionality

## License

[Your License Here]

## Acknowledgments

- OpenAI HumanEval dataset: https://github.com/openai/human-eval
- Inspired by mutation testing frameworks like Cosmic Ray
