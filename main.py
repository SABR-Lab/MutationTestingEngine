from tqdm import tqdm

#!/usr/bin/env python3
"""
DJ Mutation Engine - Main Entry Point

A modular mutation testing framework for Python code with clean separation
of concerns and well-organized components.

This is the main entry point that demonstrates the usage of the mutation
engine and its various operators.

Usage:
    python main.py  # Run built-in demo
    
Or import and use the modular components:
    from mutation import MutationEngine, NumberReplacer
    
    code = "x = 1 + 2"
    engine = MutationEngine()
    operator = NumberReplacer()
    for original, mutated, occurrence in engine.parse_and_yield_mutations(code, operator):
        print(f"Mutation {occurrence}: {mutated}")
"""

if __name__ == "__main__":
    import subprocess
    for i in tqdm(range(1,163)):
        # Run mutation testing demo for problem 1
        print(f"Running mutation testing demo for problem {i}...")
        subprocess.run(["python", "humaneval_tester.py", "--mutate", str(i), "--max-mutations", "10"], check=True)

