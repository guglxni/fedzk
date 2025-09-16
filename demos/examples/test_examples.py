#!/usr/bin/env python3
"""
Test Script for FEDzk Real-World Examples (Task 5.1.2)
=====================================================

This script validates the structure and correctness of all real-world examples
created in task 5.1.2 without requiring external dependencies like PyTorch.
"""

import os
import ast
import re
from pathlib import Path
from typing import Dict, List, Tuple, Set
import json

class ExampleValidator:
    """Validates FEDzk example files for correctness and structure."""

    def __init__(self):
        self.examples_dir = Path("examples")
        self.results = {}

    def validate_all_examples(self) -> Dict:
        """Validate all task 5.1.2 examples."""
        examples_to_test = [
            "healthcare_federated_learning.py",
            "financial_risk_assessment.py",
            "iot_device_network.py",
            "production_configuration.py",
            "troubleshooting_guide.py",
            "deployment_tutorial.py"
        ]

        print("🔍 FEDzk Real-World Examples Validation")
        print("=" * 50)

        all_results = {}

        for example_file in examples_to_test:
            print(f"\n📄 Testing: {example_file}")
            result = self.validate_example(example_file)
            all_results[example_file] = result

            # Print summary
            checks_passed = sum(1 for check in result.values() if check.get('status') == 'pass')
            total_checks = len(result)
            status = "✅ PASS" if checks_passed == total_checks else "❌ ISSUES"
            print(f"   {status}: {checks_passed}/{total_checks} checks passed")

        # Overall summary
        print(f"\n{'='*50}")
        print("VALIDATION SUMMARY")
        print('='*50)

        total_passed = 0
        total_checks = 0

        for example, checks in all_results.items():
            example_passed = sum(1 for check in checks.values() if check.get('status') == 'pass')
            example_total = len(checks)
            total_passed += example_passed
            total_checks += example_total

            status = "✅" if example_passed == example_total else "❌"
            print(f"{status} {example}: {example_passed}/{example_total}")

        overall_status = "✅ ALL EXAMPLES VALID" if total_passed == total_checks else "⚠️ SOME ISSUES FOUND"
        print(f"\nOverall: {total_passed}/{total_checks} checks passed")
        print(f"Status: {overall_status}")

        return all_results

    def validate_example(self, filename: str) -> Dict:
        """Validate a single example file."""
        filepath = self.examples_dir / filename

        if not filepath.exists():
            return {"file_exists": {"status": "fail", "message": "File not found"}}

        results = {}

        try:
            # Read file content
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()

            # 1. Check file structure
            results.update(self._validate_file_structure(content, filename))

            # 2. Check imports (without executing)
            results.update(self._validate_imports(content))

            # 3. Check class definitions
            results.update(self._validate_classes(content))

            # 4. Check function definitions
            results.update(self._validate_functions(content))

            # 5. Check documentation
            results.update(self._validate_documentation(content))

            # 6. Check FEDzk-specific patterns
            results.update(self._validate_fedzk_patterns(content))

            # 7. Check security patterns
            results.update(self._validate_security_patterns(content))

        except Exception as e:
            results["parsing_error"] = {
                "status": "fail",
                "message": f"Failed to parse file: {e}"
            }

        return results

    def _validate_file_structure(self, content: str, filename: str) -> Dict:
        """Validate basic file structure."""
        results = {}

        # Check shebang
        has_shebang = content.startswith("#!/usr/bin/env python3")
        results["shebang"] = {
            "status": "pass" if has_shebang else "fail",
            "message": "Has proper shebang" if has_shebang else "Missing shebang"
        }

        # Check docstring
        lines = content.split('\n')
        has_docstring = len(lines) > 2 and '"""' in lines[1] and '"""' in lines[2]
        results["docstring"] = {
            "status": "pass" if has_docstring else "fail",
            "message": "Has module docstring" if has_docstring else "Missing module docstring"
        }

        # Check main guard
        has_main_guard = "if __name__ == \"__main__\":" in content
        results["main_guard"] = {
            "status": "pass" if has_main_guard else "fail",
            "message": "Has main guard" if has_main_guard else "Missing main guard"
        }

        return results

    def _validate_imports(self, content: str) -> Dict:
        """Validate import statements."""
        results = {}

        # Check for FEDzk imports
        fedzk_imports = [
            "from fedzk.client import",
            "from fedzk.mpc.client import",
            "from fedzk.coordinator import",
            "from fedzk.utils import",
            "from fedzk.config import",
            "from fedzk.prover."
        ]

        has_fedzk_imports = any(imp in content for imp in fedzk_imports)
        results["fedzk_imports"] = {
            "status": "pass" if has_fedzk_imports else "fail",
            "message": "Has FEDzk imports" if has_fedzk_imports else "Missing FEDzk imports"
        }

        # Check for standard library imports
        std_imports = ["import torch", "import logging", "import time"]
        has_std_imports = any(imp in content for imp in std_imports)
        results["std_imports"] = {
            "status": "pass" if has_std_imports else "warn",
            "message": "Has standard imports" if has_std_imports else "Minimal imports (may be expected)"
        }

        return results

    def _validate_classes(self, content: str) -> Dict:
        """Validate class definitions."""
        results = {}

        # Look for class definitions
        class_pattern = r'class\s+(\w+)\s*\([^)]*\)\s*:'
        classes = re.findall(class_pattern, content)

        has_classes = len(classes) > 0
        results["classes_defined"] = {
            "status": "pass" if has_classes else "fail",
            "message": f"Classes defined: {classes}" if has_classes else "No classes defined",
            "classes": classes
        }

        # Check for FEDzk-specific class patterns
        fedzk_class_patterns = [
            "FLClient", "Coordinator", "MPCClient", "Validator"
        ]
        has_fedzk_classes = any(pattern in ' '.join(classes) for pattern in fedzk_class_patterns)
        results["fedzk_classes"] = {
            "status": "pass" if has_fedzk_classes else "warn",
            "message": "Has FEDzk-style classes" if has_fedzk_classes else "No FEDzk-style classes found"
        }

        return results

    def _validate_functions(self, content: str) -> Dict:
        """Validate function definitions."""
        results = {}

        # Look for function definitions
        func_pattern = r'def\s+(\w+)\s*\([^)]*\)\s*:'
        functions = re.findall(func_pattern, content)

        has_functions = len(functions) > 0
        results["functions_defined"] = {
            "status": "pass" if has_functions else "fail",
            "message": f"Functions defined: {len(functions)}" if has_functions else "No functions defined",
            "functions": functions[:5]  # Show first 5
        }

        # Check for main function
        has_main_func = "def main()" in content
        results["main_function"] = {
            "status": "pass" if has_main_func else "fail",
            "message": "Has main function" if has_main_func else "Missing main function"
        }

        return results

    def _validate_documentation(self, content: str) -> Dict:
        """Validate documentation."""
        results = {}

        # Check for docstrings in functions
        func_blocks = re.findall(r'def\s+\w+\s*\([^)]*\)\s*:\s*"""[^"]*"""', content, re.DOTALL)
        has_func_docstrings = len(func_blocks) > 0

        results["function_docstrings"] = {
            "status": "pass" if has_func_docstrings else "warn",
            "message": f"Function docstrings: {len(func_blocks)}" if has_func_docstrings else "No function docstrings found"
        }

        # Check for comments
        comment_lines = len([line for line in content.split('\n') if line.strip().startswith('#')])
        has_comments = comment_lines > 5  # Reasonable threshold

        results["comments"] = {
            "status": "pass" if has_comments else "warn",
            "message": f"Comment lines: {comment_lines}" if has_comments else "Few comments found"
        }

        return results

    def _validate_fedzk_patterns(self, content: str) -> Dict:
        """Validate FEDzk-specific patterns."""
        results = {}

        # Check for ZK proof patterns
        zk_patterns = [
            "generate_proof",
            "verify_proof",
            "zk_proof",
            "zero_knowledge"
        ]
        has_zk_patterns = any(pattern in content for pattern in zk_patterns)
        results["zk_patterns"] = {
            "status": "pass" if has_zk_patterns else "fail",
            "message": "Has ZK proof patterns" if has_zk_patterns else "Missing ZK proof patterns"
        }

        # Check for federated learning patterns
        fl_patterns = [
            "federated_learning",
            "model_update",
            "gradient",
            "aggregation"
        ]
        has_fl_patterns = any(pattern in content for pattern in fl_patterns)
        results["fl_patterns"] = {
            "status": "pass" if has_fl_patterns else "fail",
            "message": "Has FL patterns" if has_fl_patterns else "Missing FL patterns"
        }

        # Check for real crypto (no mocks)
        mock_patterns = ["mock", "fake", "dummy", "simulation"]
        has_mocks = any(pattern in content.lower() for pattern in mock_patterns)
        results["no_mocks"] = {
            "status": "pass" if not has_mocks else "fail",
            "message": "No mock implementations" if not has_mocks else "Contains mock implementations"
        }

        return results

    def _validate_security_patterns(self, content: str) -> Dict:
        """Validate security-related patterns."""
        results = {}

        # Check for security imports
        security_imports = [
            "cryptography",
            "secrets",
            "hashlib",
            "hmac"
        ]
        has_security_imports = any(imp in content for imp in security_imports)
        results["security_imports"] = {
            "status": "pass" if has_security_imports else "warn",
            "message": "Has security imports" if has_security_imports else "No security imports found"
        }

        # Check for error handling
        error_patterns = ["try:", "except", "raise"]
        has_error_handling = any(pattern in content for pattern in error_patterns)
        results["error_handling"] = {
            "status": "pass" if has_error_handling else "warn",
            "message": "Has error handling" if has_error_handling else "No error handling found"
        }

        # Check for logging
        has_logging = "logger." in content or "logging." in content
        results["logging"] = {
            "status": "pass" if has_logging else "warn",
            "message": "Has logging" if has_logging else "No logging found"
        }

        return results

def main():
    """Main validation function."""
    validator = ExampleValidator()

    try:
        results = validator.validate_all_examples()

        # Save detailed results
        with open("examples_validation_results.json", "w") as f:
            json.dump(results, f, indent=2)

        print(f"\n📄 Detailed results saved to: examples_validation_results.json")

        # Check if all examples passed
        all_passed = True
        for example_results in results.values():
            for check in example_results.values():
                if check.get('status') == 'fail':
                    all_passed = False
                    break

        if all_passed:
            print("\n🎉 All examples validation PASSED!")
            return 0
        else:
            print("\n⚠️ Some examples have issues - check results above")
            return 1

    except Exception as e:
        print(f"\n💥 Validation failed with error: {e}")
        return 1

if __name__ == "__main__":
    exit(main())

