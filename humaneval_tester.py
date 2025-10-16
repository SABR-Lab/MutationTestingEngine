#!/usr/bin/env python3
"""
HumanEval and HumanEval+ Dataset Tester with Mutation Engine

This script downloads the HumanEval and HumanEval+ datasets and executes
code solutions against test cases, optionally applying mutations to
test the robustness of the solutions.

Features:
- Downloads HumanEval and HumanEval+ datasets from GitHub
- Executes solutions against test cases (original and enhanced test suites)
- Applies mutation testing to evaluate code quality
- Provides detailed reports on test results and mutation scores

Usage:
    python humaneval_tester.py --download    # Download HumanEval dataset
    python humaneval_tester.py --download --dataset plus    # Download HumanEval+ dataset
    python humaneval_tester.py --test-all    # Test all problems with HumanEval
    python humaneval_tester.py --test-all --dataset plus    # Test all problems with HumanEval+
    python humaneval_tester.py --test-one 0  # Test specific problem
    python humaneval_tester.py --mutate 0    # Run mutation testing on problem
    python humaneval_tester.py --mutate 0 --max-mutations 10  # Run with 10 mutations per operator
    python humaneval_tester.py --list-results  # List all saved results
    python humaneval_tester.py --view-result filename.json  # View specific result
"""

import argparse
import json
import os
import sys
import subprocess
import tempfile
import traceback
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import urllib.request
import gzip
from datetime import datetime

try:
    from datasets import load_dataset
    DATASETS_AVAILABLE = True
except ImportError:
    DATASETS_AVAILABLE = False

from mutation import MutationEngine
from mutation.operators import (
    # Arithmetic operators
    NumberReplacer,
    ReplaceBinaryOperator_Add_Sub,
    ReplaceBinaryOperator_Sub_Add,
    ReplaceBinaryOperator_Mult_Div,
    ReplaceBinaryOperator_Div_Mult,
    ReplaceBinaryOperator_FloorDiv_Mod,
    ReplaceBinaryOperator_Mod_FloorDiv,
    ReplaceBinaryOperator_Pow_Mult,
    ReplaceBinaryOperator_Mult_Pow,
    ReplaceBinaryOperator_LShift_RShift,
    ReplaceBinaryOperator_RShift_LShift,
    ReplaceBinaryOperator_BitOr_BitXor,
    ReplaceBinaryOperator_BitXor_BitOr,
    ReplaceBinaryOperator_BitAnd_BitXor,
    ReplaceBinaryOperator_BitXor_BitAnd,
    ReplaceBinaryOperator_BitOr_BitAnd,
    ReplaceBinaryOperator_BitAnd_BitOr,
    ReplaceAugAssign_AddEq_SubEq,
    ReplaceAugAssign_SubEq_AddEq,
    ReplaceAugAssign_MulEq_DivEq,
    ReplaceAugAssign_DivEq_MulEq,
    
    # Comparison operators
    ReplaceEqWithNotEq,
    ReplaceNotEqWithEq,
    ReplaceLtWithGt,
    ReplaceGtWithLt,
    ReplaceLtEqWithGtEq,
    ReplaceGtEqWithLtEq,
    
    # Logical operators
    ReplaceAndWithOr,
    ReplaceOrWithAnd,
    RemoveNotOperator,
    RemoveUnaryMinus,
    RemoveUnaryPlus,
    
    # Control flow
    ReplaceBreakWithContinue,
    ReplaceContinueWithBreak,
    RemoveReturnValue,
    
    # Values
    ReplaceTrueWithFalse,
    ReplaceFalseWithTrue,
    ReplaceZeroWithOne,
    ReplaceOneWithZero,
    ReplaceMinusOneWithZero,
    StringToEmpty,
    StringToX,
    
    # Collections
    ListToEmpty,
    DictToEmpty,
    ReplaceTernaryCondition,
    
    # Exceptions
    ReplaceRaiseWithPass,
)


class HumanEvalTester:
    """
    A comprehensive tester for the HumanEval and HumanEval+ datasets that can execute
    code against test cases and perform mutation testing.
    """
    
    def __init__(self, data_dir: str = "humaneval_data", dataset_type: str = "original"):
        """
        Initialize the tester with a data directory and dataset type.
        
        Args:
            data_dir: Directory to store datasets and results
            dataset_type: "original" for HumanEval or "plus" for HumanEval+
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        
        # Set dataset type and file paths
        self.dataset_type = dataset_type
        if dataset_type == "plus":
            self.dataset_file = self.data_dir / "human-eval-plus.jsonl"
        else:
            self.dataset_file = self.data_dir / "human-eval-v1.0.0.jsonl"
            
        self.mutation_engine = MutationEngine()
        
        # Create results directory
        self.results_dir = self.data_dir / "results"
        self.results_dir.mkdir(exist_ok=True)
        
        # Available mutation operators - comprehensive list
        self.mutation_operators = [
            # Arithmetic operators
            ("NumberReplacer", NumberReplacer()),
            ("Add->Sub", ReplaceBinaryOperator_Add_Sub()),
            ("Sub->Add", ReplaceBinaryOperator_Sub_Add()),
            ("Mult->Div", ReplaceBinaryOperator_Mult_Div()),
            ("Div->Mult", ReplaceBinaryOperator_Div_Mult()),
            ("FloorDiv->Mod", ReplaceBinaryOperator_FloorDiv_Mod()),
            ("Mod->FloorDiv", ReplaceBinaryOperator_Mod_FloorDiv()),
            ("Pow->Mult", ReplaceBinaryOperator_Pow_Mult()),
            ("Mult->Pow", ReplaceBinaryOperator_Mult_Pow()),
            ("LShift->RShift", ReplaceBinaryOperator_LShift_RShift()),
            ("RShift->LShift", ReplaceBinaryOperator_RShift_LShift()),
            ("BitOr->BitXor", ReplaceBinaryOperator_BitOr_BitXor()),
            ("BitXor->BitOr", ReplaceBinaryOperator_BitXor_BitOr()),
            ("BitAnd->BitXor", ReplaceBinaryOperator_BitAnd_BitXor()),
            ("BitXor->BitAnd", ReplaceBinaryOperator_BitXor_BitAnd()),
            ("BitOr->BitAnd", ReplaceBinaryOperator_BitOr_BitAnd()),
            ("BitAnd->BitOr", ReplaceBinaryOperator_BitAnd_BitOr()),
            ("AddEq->SubEq", ReplaceAugAssign_AddEq_SubEq()),
            ("SubEq->AddEq", ReplaceAugAssign_SubEq_AddEq()),
            ("MulEq->DivEq", ReplaceAugAssign_MulEq_DivEq()),
            ("DivEq->MulEq", ReplaceAugAssign_DivEq_MulEq()),
            
            # Comparison operators
            ("Eq->NotEq", ReplaceEqWithNotEq()),
            ("NotEq->Eq", ReplaceNotEqWithEq()),
            ("Lt->Gt", ReplaceLtWithGt()),
            ("Gt->Lt", ReplaceGtWithLt()),
            ("LtEq->GtEq", ReplaceLtEqWithGtEq()),
            ("GtEq->LtEq", ReplaceGtEqWithLtEq()),
            
            # Logical operators
            ("And->Or", ReplaceAndWithOr()),
            ("Or->And", ReplaceOrWithAnd()),
            ("Remove Not", RemoveNotOperator()),
            ("Remove UnaryMinus", RemoveUnaryMinus()),
            ("Remove UnaryPlus", RemoveUnaryPlus()),
            
            # Control flow
            ("Break->Continue", ReplaceBreakWithContinue()),
            ("Continue->Break", ReplaceContinueWithBreak()),
            ("Remove ReturnValue", RemoveReturnValue()),
            
            # Values
            ("True->False", ReplaceTrueWithFalse()),
            ("False->True", ReplaceFalseWithTrue()),
            ("Zero->One", ReplaceZeroWithOne()),
            ("One->Zero", ReplaceOneWithZero()),
            ("MinusOne->Zero", ReplaceMinusOneWithZero()),
            ("String->Empty", StringToEmpty()),
            ("String->X", StringToX()),
            
            # Collections
            ("List->Empty", ListToEmpty()),
            ("Dict->Empty", DictToEmpty()),
            ("Ternary Condition", ReplaceTernaryCondition()),
            
            # # Exceptions
            ("Raise->Pass", ReplaceRaiseWithPass()),
        ]
    
    def save_test_result_to_json(self, result: Dict[str, Any], filename_prefix: str = "test_result") -> str:
        """Save a single test result to JSON file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # Clean task_id for filename (replace problematic characters)
        clean_task_id = result['task_id'].replace('/', '_').replace('\\', '_')
        filename = f"{filename_prefix}_{clean_task_id}_{self.dataset_type}_{timestamp}.json"
        filepath = self.results_dir / filename
        
        # Add metadata
        result_with_metadata = {
            "timestamp": timestamp,
            "test_type": "single_test",
            "metadata": {
                "task_id": result['task_id'],
                "timestamp": timestamp,
                "tester_version": "1.0"
            },
            "result": result
        }
        
        with open(filepath, 'w') as f:
            json.dump(result_with_metadata, f, indent=2)
        
        print(f"Test result saved to: {filepath}")
        return str(filepath)
    
    def save_mutation_results_to_json(self, results: Dict[str, Any], filename_prefix: str = "mutation_results") -> str:
        """Save mutation testing results to JSON file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # Clean task_id for filename (replace problematic characters)
        clean_task_id = results['task_id'].replace('/', '_').replace('\\', '_')
        filename = f"{filename_prefix}_{clean_task_id}_{self.dataset_type}_{timestamp}.json"
        filepath = self.results_dir / filename
        
        # Add metadata
        results_with_metadata = {
            "timestamp": timestamp,
            "test_type": "mutation_testing",
            "metadata": {
                "task_id": results['task_id'],
                "timestamp": timestamp,
                "tester_version": "1.0",
                "total_operators_tested": len([op for op in results.get('mutation_results', {}) if 'error' not in results['mutation_results'][op]]),
                "operators_with_errors": len([op for op in results.get('mutation_results', {}) if 'error' in results['mutation_results'][op]])
            },
            "summary": {
                "original_passes": results.get('original_passes', False),
                "total_mutations": results.get('total_mutations', 0),
                "killed_mutations": results.get('killed_mutations', 0),
                "mutation_score": results.get('mutation_score', 0),
                "survival_rate": 1 - results.get('mutation_score', 0)
            },
            "detailed_results": results
        }
        
        with open(filepath, 'w') as f:
            json.dump(results_with_metadata, f, indent=2)
        
        print(f"Mutation results saved to: {filepath}")
        return str(filepath)
    
    def save_batch_results_to_json(self, results: List[Dict[str, Any]], test_type: str = "batch_test") -> str:
        """Save batch test results to JSON file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"batch_results_{test_type}_{timestamp}.json"
        filepath = self.results_dir / filename
        
        # Calculate summary statistics
        total_tests = len(results)
        passed_tests = sum(1 for r in results if r.get('success', False))
        
        batch_results = {
            "timestamp": timestamp,
            "test_type": test_type,
            "metadata": {
                "timestamp": timestamp,
                "tester_version": "1.0",
                "total_problems_tested": total_tests
            },
            "summary": {
                "total_tests": total_tests,
                "passed_tests": passed_tests,
                "failed_tests": total_tests - passed_tests,
                "pass_rate": passed_tests / total_tests if total_tests > 0 else 0
            },
            "detailed_results": results
        }
        
        with open(filepath, 'w') as f:
            json.dump(batch_results, f, indent=2)
        
        print(f"Batch results saved to: {filepath}")
        return str(filepath)
    
    def load_results_from_json(self, filepath: str) -> Dict[str, Any]:
        """Load results from a JSON file."""
        try:
            with open(filepath, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading results from {filepath}: {e}")
            return {}
    
    def list_saved_results(self) -> List[str]:
        """List all saved result files."""
        json_files = list(self.results_dir.glob("*.json"))
        return [str(f) for f in sorted(json_files, reverse=True)]  # Most recent first
    
    def download_dataset(self) -> bool:
        """Download the HumanEval or HumanEval+ dataset."""
        if self.dataset_file.exists():
            print(f"Dataset already exists at {self.dataset_file}")
            return True
        
        if self.dataset_type == "plus":
            print("Downloading HumanEval+ dataset...")
            if not DATASETS_AVAILABLE:
                print("Error: datasets library not available. Please install it with: pip install datasets")
                return False
                
            try:
                # Load HumanEval+ dataset from Hugging Face
                ds = load_dataset("evalplus/humanevalplus")
                
                # Save to JSONL format
                with open(self.dataset_file, 'w') as f:
                    for item in ds['test']:  # HumanEval+ uses 'test' split
                        f.write(json.dumps(item) + '\n')
                        
                print(f"HumanEval+ dataset downloaded successfully to {self.dataset_file}")
                return True
                
            except Exception as e:
                print(f"Error downloading HumanEval+ dataset: {e}")
                return False
        else:
            print("Downloading HumanEval dataset...")
            url = "https://github.com/openai/human-eval/raw/master/data/HumanEval.jsonl.gz"
            
            try:
                # Download the compressed file
                with urllib.request.urlopen(url) as response:
                    compressed_data = response.read()
                
                # Decompress and save
                decompressed_data = gzip.decompress(compressed_data)
                with open(self.dataset_file, 'wb') as f:
                    f.write(decompressed_data)
                        
                print(f"HumanEval dataset downloaded successfully to {self.dataset_file}")
                return True
                
            except Exception as e:
                print(f"Error downloading HumanEval dataset: {e}")
                return False
    
    def load_dataset(self) -> List[Dict[str, Any]]:
        """Load the HumanEval or HumanEval+ dataset from the JSONL file."""
        if not self.dataset_file.exists():
            dataset_name = "HumanEval+" if self.dataset_type == "plus" else "HumanEval"
            print(f"{dataset_name} dataset not found. Please download it first with --download")
            return []
        
        problems = []
        try:
            with open(self.dataset_file, 'r') as f:
                for line in f:
                    if line.strip():
                        problem = json.loads(line)
                        
                        # Handle different formats between HumanEval and HumanEval+
                        if self.dataset_type == "plus":
                            # HumanEval+ has both 'base_input' and 'plus_input' for tests
                            # We'll use 'plus_input' which contains enhanced test cases
                            if 'plus_input' in problem:
                                # Convert plus_input format to standard test format
                                problem['test'] = problem['plus_input']
                            elif 'base_input' in problem:
                                # Fallback to base tests if plus tests not available
                                problem['test'] = problem['base_input']
                        
                        problems.append(problem)
                        
            dataset_name = "HumanEval+" if self.dataset_type == "plus" else "HumanEval"
            print(f"Loaded {len(problems)} problems from {dataset_name} dataset")
            return problems
        except Exception as e:
            print(f"Error loading dataset: {e}")
            return []
    
    def execute_code_with_tests(self, code: str, test_code: str, entry_point: str = None) -> Tuple[bool, str]:
        """
        Execute code with test cases in a safe environment.
        
        Args:
            code: The function implementation
            test_code: The test cases (could be check function or direct test cases)
            entry_point: The function name to test (optional)
            
        Returns:
            Tuple of (success, output/error message)
        """
        if entry_point:
            # Original HumanEval format with check function
            full_code = f"{code}\n\n{test_code}\n\n# Execute the check function\nif __name__ == '__main__':\n    try:\n        check({entry_point})\n        print('All tests passed!')\n    except Exception as e:\n        print(f'Test failed: {{e}}')\n        raise"
        else:
            # Fallback: just combine code and tests
            full_code = f"{code}\n\n{test_code}"
            return False, "No entry point provided for execution"
        try:
            # Create a temporary file to execute the code
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                f.write(full_code)
                temp_file = f.name
            
            # Execute the code
            result = subprocess.run(
                [sys.executable, temp_file],
                capture_output=True,
                text=True,
                timeout=10  # 10 second timeout
            )
            
            # Clean up
            # os.unlink(temp_file)
            
            if result.returncode == 0:
                return True, result.stdout
            else:
                return False, result.stderr
                
        except subprocess.TimeoutExpired:
            return False, "Execution timeout"
        except Exception as e:
            return False, f"Execution error: {e}"
    
    def _get_entry_point(self, code: str) -> str:
        """Extract the function name from the code."""
        import re
        match = re.search(r'def\s+(\w+)\s*\(', code)
        return match.group(1) if match else 'unknown_function'
    
    def test_single_problem(self, problem: Dict[str, Any], verbose: bool = True) -> Dict[str, Any]:
        """
        Test a single problem from the dataset.
        
        Args:
            problem: Problem dictionary from HumanEval or HumanEval+
            verbose: Whether to print detailed output
            
        Returns:
            Dictionary with test results
        """
        task_id = problem['task_id']
        prompt = problem['prompt']
        canonical_solution = problem['canonical_solution']
        test = problem['test']
        print(test)
        entry_point = problem['entry_point']
        
        dataset_name = "HumanEval+" if self.dataset_type == "plus" else "HumanEval"
        
        if verbose:
            print(f"\n{'='*60}")
            print(f"Testing Problem: {task_id} ({dataset_name})")
            print(f"Entry Point: {entry_point}")
            print(f"{'='*60}")
            print("Prompt:")
            print(prompt)
            print("\nCanonical Solution:")
            print(canonical_solution)
        
        # Combine prompt and solution
        complete_code = prompt + canonical_solution
        
        # Test the solution
        success, output = self.execute_code_with_tests(complete_code, test, entry_point)
        
        result = {
            'task_id': task_id,
            'dataset_type': self.dataset_type,
            'success': success,
            'output': output,
            'code': complete_code,
            'prompt': prompt,
            'canonical_solution': canonical_solution,
            'test_code': test,
            'entry_point': entry_point,
            'execution_timestamp': datetime.now().isoformat()
        }
        
        # Save result to JSON
        self.save_test_result_to_json(result)
        
        if verbose:
            print(f"\nTest Result: {'PASSED' if success else 'FAILED'} ({dataset_name})")
            if not success:
                print(f"Error: {output}")
            print(f"{'='*60}")
        
        return result
    
    def run_mutation_testing(self, problem: Dict[str, Any], max_mutations_per_operator: int = 5) -> Dict[str, Any]:
        """
        Run mutation testing on a problem.
        
        Args:
            problem: Problem dictionary from HumanEval or HumanEval+
            max_mutations_per_operator: Maximum mutations to test per operator
            
        Returns:
            Dictionary with mutation testing results
        """
        task_id = problem['task_id']
        prompt = problem['prompt']
        canonical_solution = problem['canonical_solution']
        test = problem['test']
        
        dataset_name = "HumanEval+" if self.dataset_type == "plus" else "HumanEval"
        
        print(f"\n{'='*60}")
        print(f"Mutation Testing: {task_id} ({dataset_name})")
        complete_code = prompt + canonical_solution
        print(f"{'='*60}")

        
        # First, verify the original code passes
        original_success, original_output = self.execute_code_with_tests(complete_code, test, problem['entry_point'])
        if not original_success:
            print(f"Original code fails tests: {original_output}")
            return {
                'task_id': task_id,
                'dataset_type': self.dataset_type,
                'original_passes': False,
                'mutation_results': {}
            }
        
        print("Original code passes all tests ✓")
        
        mutation_results = {}
        total_mutations = 0
        killed_mutations = 0
        
        # Test each mutation operator
        for op_name, operator in self.mutation_operators:
            print(f"\nTesting {op_name} mutations...")
            
            try:
                mutation_count = self.mutation_engine.get_mutation_count(complete_code, operator)
                if mutation_count == 0:
                    print(f"  No mutations possible with {op_name}")
                    continue
                
                op_mutations = 0
                op_killed = 0
                op_detailed_mutations = []  # Store detailed info for each mutation
                max_to_test = min(mutation_count, max_mutations_per_operator)
                
                for i in range(max_to_test):
                    original, mutated_solution = self.mutation_engine.apply_single_mutation(
                        complete_code, operator, i
                    )
                    
                    if mutated_solution is None:
                        continue
                    
                    # Create mutated complete code
                    mutated_complete_code = prompt + mutated_solution
                    
                    # Test the mutated code
                    mutated_success, mutated_output = self.execute_code_with_tests(
                        mutated_complete_code, test, problem['entry_point']
                    )
                    
                    op_mutations += 1
                    total_mutations += 1
                    
                    # Create detailed mutation record
                    mutation_record = {
                        'mutation_index': i,
                        'original_code': original,
                        'mutated_code': mutated_solution,
                        'mutated_complete_code': mutated_complete_code,
                        'execution_success': mutated_success,
                        'execution_output': mutated_output,
                        'timestamp': datetime.now().isoformat()
                    }
                    
                    if not mutated_success:
                        # Check if it's a test failure (AssertionError) vs code error
                        if "AssertionError" in mutated_output:
                            # Mutation was killed by test failure
                            op_killed += 1
                            killed_mutations += 1
                            mutation_record['status'] = 'KILLED'
                            mutation_record['reason'] = 'test_failure'
                            print(f"  Mutation {i}: KILLED ✓ (test failure)")
                        else:
                            op_mutations -= 1  # Don't count this as a mutation if it failed to apply
                            total_mutations -= 1 # Don't count this as a mutation if it failed to apply
                            mutation_record['status'] = 'BROKEN'
                            mutation_record['reason'] = 'code_error'
                            # Mutation caused code to break (syntax error, runtime error, etc.)
                            print(f"  Mutation {i}: BROKEN ✗ code error: \n {60*"-"} \n {mutated_output} \n {60*"-"}")
                    else:
                        mutation_record['status'] = 'SURVIVED'
                        mutation_record['reason'] = 'tests_passed'
                        print(f"  Mutation {i}: SURVIVED ✗ (tests passed)")
                    
                    op_detailed_mutations.append(mutation_record)

                mutation_results[op_name] = {
                    'total_mutations': op_mutations,
                    'killed_mutations': op_killed,
                    'survival_rate': (op_mutations - op_killed) / op_mutations if op_mutations > 0 else 0,
                    'detailed_mutations': op_detailed_mutations,
                    'mutation_count_available': mutation_count,
                    'mutations_tested': max_to_test
                }
                
                print(f"  {op_name}: {op_killed}/{op_mutations} killed ({op_killed/op_mutations*100:.1f}%)")
                
            except Exception as e:
                print(f"  Error testing {op_name}: {e}")
                mutation_results[op_name] = {'error': str(e)}
        
        mutation_score = killed_mutations / total_mutations if total_mutations > 0 else 0
        
        print(f"\nMutation Testing Summary ({dataset_name}):")
        print(f"Total mutations tested: {total_mutations}")
        print(f"Mutations killed: {killed_mutations}")
        print(f"Mutation score: {mutation_score:.2%}")
        
        # Prepare comprehensive results
        comprehensive_results = {
            'task_id': task_id,
            'dataset_type': self.dataset_type,
            'original_passes': True,
            'total_mutations': total_mutations,
            'killed_mutations': killed_mutations,
            'mutation_score': mutation_score,
            'mutation_results': mutation_results,
            'problem_info': {
                'prompt': prompt,
                'canonical_solution': canonical_solution,
                'test_code': test,
                'entry_point': problem['entry_point'],
                'complete_code': complete_code
            },
            'execution_info': {
                'max_mutations_per_operator': max_mutations_per_operator,
                'total_operators_tested': len(self.mutation_operators),
                'operators_with_mutations': len([op for op in mutation_results if mutation_results[op].get('total_mutations', 0) > 0]),
                'execution_timestamp': datetime.now().isoformat()
            }
        }
        
        # Save results to JSON
        self.save_mutation_results_to_json(comprehensive_results)
        
        return comprehensive_results
    
    def test_all_problems(self, max_problems: Optional[int] = None) -> List[Dict[str, Any]]:
        """Test all problems in the dataset."""
        problems = self.load_dataset()
        if not problems:
            return []
        
        if max_problems:
            problems = problems[:max_problems]
        
        dataset_name = "HumanEval+" if self.dataset_type == "plus" else "HumanEval"
        print(f"Testing {len(problems)} problems from {dataset_name}...")
        results = []
        
        for i, problem in enumerate(problems):
            print(f"\nProgress: {i+1}/{len(problems)} ({dataset_name})")
            result = self.test_single_problem(problem, verbose=False)
            results.append(result)
        
        # Summary
        passed = sum(1 for r in results if r['success'])
        print(f"\n{'='*60}")
        print(f"SUMMARY ({dataset_name}): {passed}/{len(results)} problems passed ({passed/len(results)*100:.1f}%)")
        print(f"{'='*60}")
        
        # Save batch results to JSON
        self.save_batch_results_to_json(results, f"all_problems_test_{self.dataset_type}")
        
        return results


def main():
    """Main function to handle command line arguments."""
    parser = argparse.ArgumentParser(description="HumanEval and HumanEval+ Dataset Tester with Mutation Engine")
    parser.add_argument("--download", action="store_true", help="Download the specified dataset")
    parser.add_argument("--dataset", choices=["original", "plus"], default="original", 
                       help="Dataset type: 'original' for HumanEval or 'plus' for HumanEval+ (default: original)")
    parser.add_argument("--test-all", action="store_true", help="Test all problems")
    parser.add_argument("--test-one", type=int, help="Test a specific problem by index")
    parser.add_argument("--mutate", type=int, help="Run mutation testing on a specific problem")
    parser.add_argument("--max-problems", type=int, help="Maximum number of problems to test")
    parser.add_argument("--max-mutations", type=int, default=5, help="Maximum number of mutations per operator (default: 5)")
    parser.add_argument("--data-dir", default="humaneval_data", help="Directory to store data")
    parser.add_argument("--list-results", action="store_true", help="List all saved result files")
    parser.add_argument("--view-result", type=str, help="View a specific result file (provide filename)")
    
    args = parser.parse_args()
    
    tester = HumanEvalTester(args.data_dir, args.dataset)
    
    if args.download:
        tester.download_dataset()
        return
    
    if args.list_results:
        results = tester.list_saved_results()
        print(f"\nSaved result files ({len(results)} total):")
        for i, result_file in enumerate(results):
            print(f"{i+1:3d}. {Path(result_file).name}")
        return
    
    if args.view_result:
        result_path = args.view_result
        if not result_path.startswith('/'):
            result_path = tester.results_dir / result_path
        
        result_data = tester.load_results_from_json(result_path)
        if result_data:
            print(f"\nResult file: {result_path}")
            print("=" * 60)
            print(json.dumps(result_data, indent=2))
        return
    
    if args.test_all:
        tester.test_all_problems(args.max_problems)
        return
    
    if args.test_one is not None:
        problems = tester.load_dataset()
        if 0 <= args.test_one < len(problems):
            tester.test_single_problem(problems[args.test_one])
        else:
            print(f"Problem index {args.test_one} out of range (0-{len(problems)-1})")
        return
    
    if args.mutate is not None:
        problems = tester.load_dataset()
        if 0 <= args.mutate < len(problems):
            tester.run_mutation_testing(problems[args.mutate], args.max_mutations)
        else:
            print(f"Problem index {args.mutate} out of range (0-{len(problems)-1})")
        return
    
if __name__ == "__main__":
    main()
