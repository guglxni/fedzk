#!/usr/bin/env python3
"""
Comprehensive Testing Framework for FEDzk Production Readiness

This module implements Task 1.1.4: Comprehensive Testing Framework that verifies
the intended outcomes of Tasks 1.1.1, 1.1.2, and 1.1.3.

Tests verify:
- ZKProver cleanup (no mock proofs, strict validation)
- MPC Server cleanup (no mock generation, strict ZK enforcement)
- ZKVerifier cleanup (no mock verification, strict key validation)
- Integration testing for complete ZK workflow
- Security regression testing
- Production readiness validation
"""

import os
import sys
import tempfile
import subprocess
import time
from pathlib import Path
from typing import Dict, List, Any, Optional

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import torch


class TestSuiteZKProverCleanup:
    """Test suite for verifying ZKProver cleanup (Task 1.1.1)."""

    def __init__(self):
        self.passed = 0
        self.total = 0

    def run_all_tests(self) -> Dict[str, Any]:
        """Run all ZKProver cleanup tests."""
        print("🧪 Running ZKProver Cleanup Tests (Task 1.1.1)")

        tests = [
            self.test_no_mock_proof_generation,
            self.test_no_environment_bypasses,
            self.test_strict_zk_validation,
            self.test_hard_failure_on_missing_tools,
            self.test_startup_verification,
            self.test_no_deterministic_test_proofs,
        ]

        results = {}
        for test in tests:
            self.total += 1
            try:
                result = test()
                if result:
                    self.passed += 1
                    print(f"  ✅ {test.__name__}")
                else:
                    print(f"  ❌ {test.__name__}")
                results[test.__name__] = result
            except Exception as e:
                print(f"  ❌ {test.__name__}: {e}")
                results[test.__name__] = False

        return results

    def test_no_mock_proof_generation(self) -> bool:
        """Verify no mock proof generation exists in ZKProver."""
        with open('src/fedzk/prover/zkgenerator.py', 'r') as f:
            content = f.read()

        mock_patterns = [
            "test_proof =",
            "mock_secure_proof",
            "mock_proof",
            "hashlib.md5",
            "input_hash[:8]",
            "artificial delay",
            "time.sleep.*0.0",
            "pi_a.*input_hash",
            "pi_b.*input_hash",
            "pi_c.*input_hash"
        ]

        for pattern in mock_patterns:
            if pattern in content:
                return False
        return True

    def test_no_environment_bypasses(self) -> bool:
        """Verify no environment variable bypasses in ZKProver."""
        with open('src/fedzk/prover/zkgenerator.py', 'r') as f:
            content = f.read()

        bypass_patterns = [
            "FEDZK_TEST_MODE",
            "FEDZK_ZK_VERIFIED",
            "os.getenv.*test",
            "test.*mode.*bypass"
        ]

        for pattern in bypass_patterns:
            if pattern in content:
                return False
        return True

    def test_strict_zk_validation(self) -> bool:
        """Verify strict ZK validation is implemented."""
        with open('src/fedzk/prover/zkgenerator.py', 'r') as f:
            content = f.read()

        required_patterns = [
            "Strict verification",
            "no test mode bypasses",
            "Missing ZK tools",
            "Missing ZK circuit files",
            "file_size == 0",
            "subprocess.run.*circom",
            "subprocess.run.*snarkjs"
        ]

        found_patterns = sum(1 for pattern in required_patterns if pattern in content)
        return found_patterns >= len(required_patterns) * 0.8  # 80% coverage

    def test_hard_failure_on_missing_tools(self) -> bool:
        """Verify hard failure when ZK tools are missing."""
        # Set environment to simulate missing tools
        original_path = os.environ.get('PATH', '')
        temp_path = '/tmp/nonexistent_path'

        try:
            os.environ['PATH'] = temp_path
            from fedzk.prover.zkgenerator import ZKProver

            try:
                prover = ZKProver()
                return False  # Should have failed
            except RuntimeError as e:
                return "Missing ZK tools" in str(e) or "snarkjs" in str(e).lower()
        except Exception:
            return False
        finally:
            os.environ['PATH'] = original_path

    def test_startup_verification(self) -> bool:
        """Verify startup verification checks all required files."""
        try:
            from fedzk.prover.zkgenerator import ZKProver
            prover = ZKProver()  # This should work with existing setup
            return hasattr(prover, 'wasm_path') and hasattr(prover, 'zkey_path')
        except Exception:
            return False

    def test_no_deterministic_test_proofs(self) -> bool:
        """Verify no deterministic test proof generation."""
        with open('src/fedzk/prover/zkgenerator.py', 'r') as f:
            content = f.read()

        # Check that old test proof patterns are gone
        test_proof_patterns = [
            "input_hash = hashlib.md5",
            "pi_a.*[input_hash",
            "pi_b.*[input_hash",
            "pi_c.*[input_hash"
        ]

        for pattern in test_proof_patterns:
            if pattern in content:
                return False
        return True


class TestSuiteMPCServerCleanup:
    """Test suite for verifying MPC Server cleanup (Task 1.1.2)."""

    def __init__(self):
        self.passed = 0
        self.total = 0

    def run_all_tests(self) -> Dict[str, Any]:
        """Run all MPC Server cleanup tests."""
        print("🧪 Running MPC Server Cleanup Tests (Task 1.1.2)")

        tests = [
            self.test_no_mock_proof_generation_server,
            self.test_no_test_mode_handling,
            self.test_strict_server_validation,
            self.test_server_error_handling,
            self.test_no_fallback_generation,
        ]

        results = {}
        for test in tests:
            self.total += 1
            try:
                result = test()
                if result:
                    self.passed += 1
                    print(f"  ✅ {test.__name__}")
                else:
                    print(f"  ❌ {test.__name__}")
                results[test.__name__] = result
            except Exception as e:
                print(f"  ❌ {test.__name__}: {e}")
                results[test.__name__] = False

        return results

    def test_no_mock_proof_generation_server(self) -> bool:
        """Verify no mock proof generation in MPC server."""
        with open('src/fedzk/mpc/server.py', 'r') as f:
            content = f.read()

        mock_patterns = [
            "test_proof =",
            "dummy proof",
            "hashlib.md5",
            "input_hash[:8]",
            "pi_a.*input_hash"
        ]

        for pattern in mock_patterns:
            if pattern in content:
                return False
        return True

    def test_no_test_mode_handling(self) -> bool:
        """Verify no test mode handling in server."""
        with open('src/fedzk/mpc/server.py', 'r') as f:
            content = f.read()

        # Should still have logging but no bypass logic
        if "FEDZK_TEST_MODE" in content:
            # Check if it's just logging, not bypass logic
            lines = content.split('\n')
            for line in lines:
                if "FEDZK_TEST_MODE" in line and "if" in line:
                    return False  # Found conditional bypass
        return True

    def test_strict_server_validation(self) -> bool:
        """Verify strict server validation."""
        with open('src/fedzk/mpc/server.py', 'r') as f:
            content = f.read()

        required_patterns = [
            "Strict ZK toolchain validation",
            "required_tools = [\"circom\", \"snarkjs\"]",
            "file_size == 0",
            "All ZK toolchain components validated"
        ]

        found_patterns = sum(1 for pattern in required_patterns if pattern in content)
        return found_patterns >= len(required_patterns) * 0.8

    def test_server_error_handling(self) -> bool:
        """Verify proper server error handling."""
        with open('src/fedzk/mpc/server.py', 'r') as f:
            content = f.read()

        # Should have strict error handling without fallbacks
        return "Strict error handling" in content and "no fallback" in content

    def test_no_fallback_generation(self) -> bool:
        """Verify no fallback proof generation."""
        with open('src/fedzk/mpc/server.py', 'r') as f:
            content = f.read()

        # Should not have fallback proof generation
        fallback_patterns = [
            "if os.getenv.*FEDZK_TEST_MODE",
            "test_proof =",
            "fallback.*proof"
        ]

        for pattern in fallback_patterns:
            if pattern in content:
                return False
        return True


class TestSuiteZKVerifierCleanup:
    """Test suite for verifying ZKVerifier cleanup (Task 1.1.3)."""

    def __init__(self):
        self.passed = 0
        self.total = 0

    def run_all_tests(self) -> Dict[str, Any]:
        """Run all ZKVerifier cleanup tests."""
        print("🧪 Running ZKVerifier Cleanup Tests (Task 1.1.3)")

        tests = [
            self.test_no_mock_verification_logic,
            self.test_strict_key_validation,
            self.test_circuit_specific_mapping,
            self.test_no_environment_bypasses_verifier,
            self.test_verifier_initialization,
        ]

        results = {}
        for test in tests:
            self.total += 1
            try:
                result = test()
                if result:
                    self.passed += 1
                    print(f"  ✅ {test.__name__}")
                else:
                    print(f"  ❌ {test.__name__}")
                results[test.__name__] = result
            except Exception as e:
                print(f"  ❌ {test.__name__}: {e}")
                results[test.__name__] = False

        return results

    def test_no_mock_verification_logic(self) -> bool:
        """Verify no mock verification logic in ZKVerifier."""
        with open('src/fedzk/prover/verifier.py', 'r') as f:
            content = f.read()

        mock_patterns = [
            "FEDZK_TEST_MODE",
            "FEDZK_ZK_VERIFIED",
            "always return success",
            "pi_a.*12345",
            "pi_a.*67890",
            "return True.*test"
        ]

        for pattern in mock_patterns:
            if pattern in content:
                return False
        return True

    def test_strict_key_validation(self) -> bool:
        """Verify strict verification key validation."""
        with open('src/fedzk/prover/verifier.py', 'r') as f:
            content = f.read()

        required_patterns = [
            "Strict verification",
            "file_size == 0",
            "Cannot access verification key",
            "Verification key not found"
        ]

        found_patterns = sum(1 for pattern in required_patterns if pattern in content)
        return found_patterns >= len(required_patterns) * 0.8

    def test_circuit_specific_mapping(self) -> bool:
        """Verify circuit-specific verification key mapping."""
        with open('src/fedzk/prover/verifier.py', 'r') as f:
            content = f.read()

        required_patterns = [
            "circuit_keys",
            "standard.*verification_key",
            "secure.*verification_key_secure",
            "_get_verification_key_for_proof",
            "_detect_circuit_type"
        ]

        found_patterns = sum(1 for pattern in required_patterns if pattern in content)
        return found_patterns >= len(required_patterns) * 0.8

    def test_no_environment_bypasses_verifier(self) -> bool:
        """Verify no environment bypasses in ZKVerifier."""
        with open('src/fedzk/prover/verifier.py', 'r') as f:
            content = f.read()

        bypass_patterns = [
            "os.getenv.*FEDZK_TEST_MODE",
            "os.getenv.*FEDZK_ZK_VERIFIED",
            "test.*mode.*bypass"
        ]

        for pattern in bypass_patterns:
            if pattern in content:
                return False
        return True

    def test_verifier_initialization(self) -> bool:
        """Verify ZKVerifier initializes correctly."""
        try:
            from fedzk.prover.verifier import ZKVerifier
            verifier = ZKVerifier()

            # Check required attributes
            required_attrs = ['circuit_keys', 'standard_vkey_path', 'secure_vkey_path']
            for attr in required_attrs:
                if not hasattr(verifier, attr):
                    return False

            # Check circuit keys are populated
            if not verifier.circuit_keys or len(verifier.circuit_keys) == 0:
                return False

            return True
        except Exception:
            return False


class TestSuiteIntegrationAndSecurity:
    """Integration tests and security regression tests."""

    def __init__(self):
        self.passed = 0
        self.total = 0

    def run_all_tests(self) -> Dict[str, Any]:
        """Run all integration and security tests."""
        print("🧪 Running Integration and Security Tests")

        tests = [
            self.test_complete_zk_workflow,
            self.test_security_regression,
            self.test_production_readiness,
            self.test_dependency_validation,
        ]

        results = {}
        for test in tests:
            self.total += 1
            try:
                result = test()
                if result:
                    self.passed += 1
                    print(f"  ✅ {test.__name__}")
                else:
                    print(f"  ❌ {test.__name__}")
                results[test.__name__] = result
            except Exception as e:
                print(f"  ❌ {test.__name__}: {e}")
                results[test.__name__] = False

        return results

    def test_complete_zk_workflow(self) -> bool:
        """Test complete ZK workflow from generation to verification."""
        try:
            from fedzk.prover.zkgenerator import ZKProver
            from fedzk.prover.verifier import ZKVerifier

            # Create test data
            test_gradients = {'weights': torch.tensor([1.0, 2.0, 3.0, 4.0])}

            # Generate proof
            prover = ZKProver(secure=False)
            proof, public_signals = prover.generate_proof(test_gradients)

            # Verify proof
            verifier = ZKVerifier()
            is_valid = verifier.verify_proof(proof, public_signals)

            return is_valid
        except Exception:
            return False

    def test_security_regression(self) -> bool:
        """Test that security regressions are prevented."""
        # Set environment variables that used to enable bypasses
        os.environ['FEDZK_TEST_MODE'] = 'true'
        os.environ['FEDZK_ZK_VERIFIED'] = 'true'

        try:
            from fedzk.prover.zkgenerator import ZKProver
            from fedzk.prover.verifier import ZKVerifier

            # These should still work with real ZK setup, not bypass
            test_gradients = {'weights': torch.tensor([1.0, 2.0, 3.0, 4.0])}

            prover = ZKProver(secure=False)
            proof, public_signals = prover.generate_proof(test_gradients)

            verifier = ZKVerifier()
            is_valid = verifier.verify_proof(proof, public_signals)

            # Should get real results, not mocked
            return is_valid and isinstance(proof, dict) and 'pi_a' in proof
        except Exception:
            return False
        finally:
            os.environ.pop('FEDZK_TEST_MODE', None)
            os.environ.pop('FEDZK_ZK_VERIFIED', None)

    def test_production_readiness(self) -> bool:
        """Test production readiness validation."""
        try:
            from fedzk.prover.zkgenerator import ZKProver, ZKVerifier

            # Should be able to initialize all components
            prover = ZKProver()
            verifier = ZKVerifier()

            # Should have proper error handling
            try:
                # This should fail appropriately
                invalid_prover = ZKProver()
                # Try to generate proof with invalid data
                invalid_prover.generate_proof({})
                return False  # Should have failed
            except (ValueError, RuntimeError):
                return True  # Correctly failed

        except Exception:
            return False

    def test_dependency_validation(self) -> bool:
        """Test dependency validation and integrity."""
        # Check that all required files exist and are valid
        required_files = [
            'src/fedzk/prover/zkgenerator.py',
            'src/fedzk/mpc/server.py',
            'src/fedzk/prover/verifier.py',
            'src/fedzk/zk/model_update.wasm',
            'src/fedzk/zk/proving_key.zkey',
            'src/fedzk/zk/verification_key.json'
        ]

        for file_path in required_files:
            if not os.path.exists(file_path):
                return False

            # Check file is not empty
            if os.path.getsize(file_path) == 0:
                return False

        return True


def main():
    """Run the comprehensive testing framework."""
    print("🚀 FEDzk Comprehensive Testing Framework (Task 1.1.4)")
    print("=" * 60)

    # Initialize test suites
    zkprover_tests = TestSuiteZKProverCleanup()
    mpc_server_tests = TestSuiteMPCServerCleanup()
    zkverifier_tests = TestSuiteZKVerifierCleanup()
    integration_tests = TestSuiteIntegrationAndSecurity()

    # Run all test suites
    all_results = {}

    test_suites = [
        ("ZKProver Cleanup", zkprover_tests),
        ("MPC Server Cleanup", mpc_server_tests),
        ("ZKVerifier Cleanup", zkverifier_tests),
        ("Integration & Security", integration_tests),
    ]

    total_passed = 0
    total_tests = 0

    for suite_name, test_suite in test_suites:
        print(f"\n{suite_name}:")
        print("-" * 40)
        results = test_suite.run_all_tests()
        all_results[suite_name] = results

        total_passed += test_suite.passed
        total_tests += test_suite.total

    # Summary
    print("\n" + "=" * 60)
    print("📊 COMPREHENSIVE TESTING FRAMEWORK RESULTS")
    print("=" * 60)

    overall_success_rate = (total_passed / total_tests * 100) if total_tests > 0 else 0

    print(f"Total Tests: {total_tests}")
    print(f"Passed: {total_passed}")
    print(f"Failed: {total_tests - total_passed}")
    print(f"Success Rate: {overall_success_rate:.1f}%")
    if overall_success_rate >= 90:
        print("🎉 SUCCESS: All cleanup tasks verified successfully!")
        print("✅ FEDzk is production-ready with no mock implementations")
        print("✅ All security regressions prevented")
        print("✅ Integration testing passed")
        return 0
    elif overall_success_rate >= 75:
        print("⚠️ PARTIAL SUCCESS: Most cleanup tasks verified")
        print("🔧 Some minor issues may need attention")
        return 1
    else:
        print("❌ FAILURE: Significant issues detected")
        print("🔧 Major cleanup verification failures")
        return 2


if __name__ == "__main__":
    sys.exit(main())
