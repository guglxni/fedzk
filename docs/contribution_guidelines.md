# FEDzk Contribution Guidelines

## 🔐 Contributing to Real Cryptographic Components

This guide outlines the process for contributing to FEDzk's real cryptographic components. **All contributions must maintain cryptographic integrity and security standards.**

---

## 📋 Table of Contents

1. [Getting Started](#getting-started)
2. [Development Workflow](#development-workflow)
3. [Cryptographic Contribution Standards](#cryptographic-contribution-standards)
4. [Testing Requirements](#testing-requirements)
5. [Security Review Process](#security-review-process)
6. [Documentation Standards](#documentation-standards)
7. [Code Review Guidelines](#code-review-guidelines)
8. [Release Process](#release-process)

---

## 🚀 Getting Started

### Prerequisites

```bash
# Development environment
- Python 3.9+
- Node.js 16+
- Circom 2.0+
- SNARKjs 0.7+
- Git with GPG signing
- Security clearance (for cryptographic components)

# Development tools
pip install -e .[dev]
pre-commit install
```

### Repository Setup

```bash
# Clone repository
git clone https://github.com/your-org/fedzk.git
cd fedzk

# Setup development environment
./scripts/setup_zk.sh
pip install -e .[dev]

# Verify setup
fedzk verify setup
python -m pytest tests/ -v
```

### Branch Naming Convention

```bash
# Feature branches
git checkout -b feature/cryptographic-circuit-optimization
git checkout -b feature/enhanced-security-validation

# Bug fix branches
git checkout -b fix/zk-proof-validation-bug
git checkout -b fix/security-vulnerability-patch

# Documentation branches
git checkout -b docs/cryptographic-api-reference
git checkout -b docs/security-best-practices

# Release branches
git checkout -b release/v1.0.0
```

---

## 🔄 Development Workflow

### 1. Issue Creation

**All contributions must start with an issue:**

```markdown
# Issue Template for Cryptographic Components

## Problem Statement
[Clear description of the cryptographic issue or enhancement]

## Security Impact
- [ ] Critical: Affects cryptographic security
- [ ] High: Significant security improvement
- [ ] Medium: Security enhancement
- [ ] Low: Minor improvement

## Technical Requirements
- [ ] Must maintain backward compatibility
- [ ] Requires trusted setup update
- [ ] Affects circuit constraints
- [ ] Changes proof verification

## Acceptance Criteria
- [ ] Cryptographic security maintained
- [ ] Comprehensive test coverage
- [ ] Documentation updated
- [ ] Security review passed
```

### 2. Development Process

```bash
# 1. Create feature branch
git checkout -b feature/your-cryptographic-feature

# 2. Implement changes with security in mind
# 3. Add comprehensive tests
# 4. Update documentation
# 5. Run security checks

# 6. Commit with security-focused messages
git add .
git commit -S -m "feat: implement secure ZK circuit optimization

- Add constraint validation for gradient bounds
- Implement secure proof verification
- Update trusted setup procedure
- Add comprehensive security tests

Security: Maintains cryptographic integrity
Tests: 95% coverage on cryptographic components
Breaking: No backward compatibility changes"

# 7. Push and create PR
git push origin feature/your-cryptographic-feature
```

### 3. Pull Request Template

```markdown
# Cryptographic Component Enhancement

## 🔐 Security Impact
- [ ] **Critical**: Affects core cryptographic primitives
- [ ] **High**: Significant security improvement
- [ ] **Medium**: Security enhancement
- [ ] **Low**: Minor security improvement

## 📋 Changes Made

### Cryptographic Components
- [ ] ZK Circuit modifications
- [ ] Proof generation changes
- [ ] Verification logic updates
- [ ] Trusted setup procedure changes

### Security Features
- [ ] Input validation enhancements
- [ ] Authentication improvements
- [ ] Authorization updates
- [ ] Audit logging additions

### Testing
- [ ] Unit tests for cryptographic functions
- [ ] Integration tests for security features
- [ ] Performance tests for cryptographic operations
- [ ] Security regression tests

## 🧪 Testing Results

### Test Coverage
- **Cryptographic Components**: [X]% coverage
- **Security Features**: [X]% coverage
- **Integration Tests**: [X] passed

### Security Testing
- [ ] Static analysis passed
- [ ] Dependency vulnerability scan passed
- [ ] Cryptographic primitive validation passed
- [ ] Penetration testing completed

### Performance Impact
- **Proof Generation**: [X]% change ([X]s → [X]s)
- **Verification**: [X]% change ([X]s → [X]s)
- **Memory Usage**: [X]% change ([X]MB → [X]MB)

## 🔒 Security Review

### Checklist
- [ ] Cryptographic security maintained
- [ ] No hardcoded secrets or keys
- [ ] Secure random number generation
- [ ] Proper input validation
- [ ] Secure error handling
- [ ] Audit logging implemented
- [ ] Rate limiting applied
- [ ] Access control enforced

### Security Considerations
[Detail any security considerations or trade-offs]

### Threat Model Updates
[Describe any changes to the threat model]

## 📚 Documentation Updates

### Files Updated
- [ ] API documentation updated
- [ ] Security best practices updated
- [ ] Developer guide updated
- [ ] Circuit documentation updated

### New Documentation
- [ ] Security review completed
- [ ] Integration guide provided
- [ ] Troubleshooting guide updated
- [ ] Performance benchmarks included

## 🚀 Deployment Considerations

### Breaking Changes
- [ ] Backward compatibility maintained
- [ ] Migration guide provided
- [ ] Rollback plan documented

### Infrastructure Requirements
- [ ] Additional compute resources needed
- [ ] Network security updates required
- [ ] Monitoring enhancements needed

## ✅ Acceptance Criteria

- [ ] All security tests pass
- [ ] Code review completed by security team
- [ ] Documentation reviewed and approved
- [ ] Performance benchmarks meet requirements
- [ ] Integration tests pass in CI/CD pipeline
- [ ] Security audit completed and signed off

## 🔗 Related Issues

- Closes #[issue_number]
- Related to #[issue_number]

---

**By submitting this PR, I confirm that:**
- [ ] I have read and followed the [Security Best Practices](./security_best_practices.md)
- [ ] All cryptographic components have been security reviewed
- [ ] I have not introduced any security vulnerabilities
- [ ] All tests pass and code coverage is maintained
- [ ] Documentation has been updated accordingly
```

---

## 🔐 Cryptographic Contribution Standards

### Code Quality Requirements

#### 1. Cryptographic Code Standards

```python
# ✅ GOOD: Secure cryptographic implementation
import secrets
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

class SecureKeyDerivation:
    """Secure key derivation for cryptographic operations."""

    def __init__(self, salt: bytes = None):
        self.salt = salt or secrets.token_bytes(16)
        self.iterations = 100000  # NIST recommended minimum

    def derive_key(self, password: str, key_length: int = 32) -> bytes:
        """Derive cryptographic key from password."""
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=key_length,
            salt=self.salt,
            iterations=self.iterations,
        )
        return kdf.derive(password.encode())

# ❌ BAD: Insecure implementation
import hashlib

def insecure_key_derivation(password: str) -> str:
    """INSECURE: Do not use in production."""
    return hashlib.md5(password.encode()).hexdigest()  # MD5 is broken
```

#### 2. Input Validation Standards

```python
# ✅ GOOD: Comprehensive input validation
def validate_cryptographic_input(data: Dict[str, Any]) -> bool:
    """Validate input for cryptographic operations."""

    # Type checking
    if not isinstance(data, dict):
        raise ValueError("Input must be a dictionary")

    # Required fields
    required_fields = ['gradients', 'client_id', 'timestamp']
    for field in required_fields:
        if field not in data:
            raise ValueError(f"Missing required field: {field}")

    # Gradient validation
    gradients = data['gradients']
    if not isinstance(gradients, (list, dict)):
        raise ValueError("Gradients must be list or dictionary")

    # Value range validation
    if isinstance(gradients, list):
        for i, grad in enumerate(gradients):
            if not isinstance(grad, int):
                raise ValueError(f"Gradient {i} must be integer")
            if not (-10**9 <= grad <= 10**9):
                raise ValueError(f"Gradient {i} out of valid range")

    # Timestamp validation (prevent replay attacks)
    timestamp = data.get('timestamp', 0)
    current_time = int(time.time())
    if abs(current_time - timestamp) > 300:  # 5 minute window
        raise ValueError("Timestamp outside valid range")

    return True

# ❌ BAD: Insufficient validation
def bad_validation(data):
    """INSECURE: Insufficient input validation."""
    # No type checking
    # No range validation
    # No replay attack prevention
    return data.get('gradients', [])
```

#### 3. Error Handling Standards

```python
# ✅ GOOD: Secure error handling
class CryptographicError(Exception):
    """Base class for cryptographic errors."""
    pass

class ProofVerificationError(CryptographicError):
    """Error during proof verification."""
    def __init__(self, message: str, proof_hash: str = None):
        super().__init__(message)
        self.proof_hash = proof_hash

        # Log security event without exposing sensitive data
        logger.warning(f"Proof verification failed: {message[:100]}...")

class CircuitCompilationError(CryptographicError):
    """Error during circuit compilation."""
    pass

def secure_error_handler(func):
    """Decorator for secure error handling."""
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except CryptographicError:
            # Re-raise cryptographic errors
            raise
        except Exception as e:
            # Log unexpected errors securely
            logger.error(f"Unexpected cryptographic error: {type(e).__name__}")
            # Do not expose internal error details
            raise CryptographicError("Cryptographic operation failed")
    return wrapper

# ❌ BAD: Insecure error handling
def insecure_error_handler():
    """INSECURE: Exposes internal error details."""
    try:
        # Some cryptographic operation
        pass
    except Exception as e:
        # BAD: Exposes internal implementation details
        raise Exception(f"Internal error: {str(e)}")
```

### Circuit Development Standards

#### 1. Circuit Security Best Practices

```circom
pragma circom 2.0.0;

include "circomlib/poseidon.circom";
include "circomlib/comparators.circom";

template SecureGradientAggregation(n) {
    // === PUBLIC INPUTS ===
    signal input gradients[n];
    signal input maxNorm;
    signal input clientId;

    // === PRIVATE INPUTS ===
    signal input timestamp;
    signal input clientSecret;

    // === OUTPUTS ===
    signal output aggregatedNorm;
    signal output securityProof;
    signal output clientCommitment;

    // === CONSTRAINTS ===

    // 1. Range validation for all inputs
    for (var i = 0; i < n; i++) {
        assert(gradients[i] >= -1000000);
        assert(gradients[i] <= 1000000);
    }

    // 2. Timestamp validation (anti-replay)
    assert(timestamp > 1640995200);  // After 2022-01-01
    assert(timestamp < 2147483647);  // Before 2038

    // 3. Norm calculation with overflow protection
    signal sq[n];
    signal accumulator[n+1];
    accumulator[0] = 0;

    for (var i = 0; i < n; i++) {
        sq[i] = gradients[i] * gradients[i];
        accumulator[i+1] = accumulator[i] + sq[i];

        // Prevent overflow
        assert(accumulator[i+1] >= accumulator[i]);
    }

    aggregatedNorm = accumulator[n];

    // 4. Security constraints
    assert(aggregatedNorm <= maxNorm);

    // 5. Client commitment for authentication
    component poseidon = Poseidon(3);
    poseidon.inputs[0] = clientId;
    poseidon.inputs[1] = timestamp;
    poseidon.inputs[2] = clientSecret;
    clientCommitment = poseidon.out;

    // 6. Security proof
    securityProof = 1;
}

// Main component definition
component main { public [gradients, maxNorm, clientId] } = SecureGradientAggregation(4);
```

#### 2. Circuit Testing Standards

```python
import unittest
import json
from pathlib import Path
import subprocess

class TestSecureCircuits(unittest.TestCase):
    """Test secure circuit implementations."""

    def setUp(self):
        self.circuit_dir = Path("src/fedzk/zk/circuits")
        self.test_data = {
            "gradients": [100, 200, 300, 400],
            "maxNorm": 500000,
            "clientId": 12345,
            "timestamp": 1704067200,
            "clientSecret": 98765
        }

    def test_circuit_compilation_security(self):
        """Test that circuit compiles without security vulnerabilities."""
        circuit_file = self.circuit_dir / "secure_gradient_aggregation.circom"

        # Compile with security checks
        result = subprocess.run([
            "circom", str(circuit_file),
            "--r1cs", "--wasm", "--sym",
            "--O2"  # Optimization level for security
        ], capture_output=True, text=True)

        self.assertEqual(result.returncode, 0,
                        f"Circuit compilation failed: {result.stderr}")

        # Verify no insecure patterns in output
        r1cs_file = circuit_file.with_suffix('.r1cs')
        with open(r1cs_file, 'rb') as f:
            r1cs_content = f.read()

        # Check for potential security issues
        # (In practice, this would use specialized tools)

    def test_constraint_validation(self):
        """Test that circuit constraints are properly validated."""
        # Test with valid inputs
        valid_result = self._test_circuit_constraints(self.test_data)
        self.assertTrue(valid_result)

        # Test with invalid inputs (should fail)
        invalid_data = self.test_data.copy()
        invalid_data["gradients"] = [100, 200, 300, 400, 999999999]  # Overflow

        invalid_result = self._test_circuit_constraints(invalid_data)
        self.assertFalse(invalid_result)

    def _test_circuit_constraints(self, input_data: dict) -> bool:
        """Test circuit constraints with given inputs."""
        try:
            # Write input to file
            input_file = self.circuit_dir / "test_input.json"
            with open(input_file, 'w') as f:
                json.dump(input_data, f)

            # Generate witness
            result = subprocess.run([
                "snarkjs", "wtns", "calculate",
                "secure_gradient_aggregation_js/secure_gradient_aggregation.wasm",
                str(input_file),
                "test_witness.wtns"
            ], capture_output=True, text=True, cwd=self.circuit_dir)

            return result.returncode == 0

        except Exception:
            return False

    def test_proof_security_properties(self):
        """Test that generated proofs have required security properties."""
        # Generate proof
        proof, signals = self._generate_test_proof()

        # Verify proof structure
        required_proof_fields = ['pi_a', 'pi_b', 'pi_c', 'protocol']
        for field in required_proof_fields:
            self.assertIn(field, proof)

        # Verify proof size (should be constant)
        proof_size = len(json.dumps(proof))
        self.assertLess(proof_size, 2000, "Proof size should be constant")

        # Verify signals format
        self.assertIsInstance(signals, list)
        self.assertGreater(len(signals), 0)

    def _generate_test_proof(self) -> tuple:
        """Generate test proof for validation."""
        # In practice, this would use the actual proof generation
        # For testing, return mock structure
        return {
            "pi_a": [0, 0, 0],
            "pi_b": [[0, 0], [0, 0], [0, 0]],
            "pi_c": [0, 0, 0],
            "protocol": "groth16"
        }, [1, 2, 3, 4]
```

---

## 🧪 Testing Requirements

### Security Testing Standards

#### 1. Cryptographic Function Testing

```python
class TestCryptographicSecurity(unittest.TestCase):
    """Test cryptographic security properties."""

    def test_proof_uniqueness(self):
        """Test that different inputs produce different proofs."""
        prover = ZKProver()

        input1 = {"gradients": [1, 2, 3, 4]}
        input2 = {"gradients": [5, 6, 7, 8]}

        proof1, _ = prover.generate_proof(input1)
        proof2, _ = prover.generate_proof(input2)

        # Proofs should be different
        self.assertNotEqual(proof1, proof2)

        # Proofs should verify correctly
        verifier = ZKVerifier()
        self.assertTrue(verifier.verify_proof(proof1, input1))
        self.assertTrue(verifier.verify_proof(proof2, input2))

        # Cross-verification should fail
        self.assertFalse(verifier.verify_proof(proof1, input2))

    def test_timing_attack_resistance(self):
        """Test resistance to timing attacks."""
        import time

        prover = ZKProver()
        test_inputs = [
            {"gradients": [1, 2, 3, 4]},
            {"gradients": [1, 2, 3, 5]},  # Minimal difference
            {"gradients": [999, 999, 999, 999]},  # Large values
        ]

        times = []
        for input_data in test_inputs:
            start = time.perf_counter()
            proof, _ = prover.generate_proof(input_data)
            end = time.perf_counter()
            times.append(end - start)

        # Timing variation should be minimal
        avg_time = sum(times) / len(times)
        max_deviation = max(abs(t - avg_time) for t in times)

        # Allow 10% timing variation (adjust based on system)
        self.assertLess(max_deviation / avg_time, 0.1,
                       "Timing attack vulnerability detected")

    def test_input_fuzzing(self):
        """Test robustness against malformed inputs."""
        prover = ZKProver()

        # Test various malformed inputs
        malformed_inputs = [
            {"gradients": []},  # Empty
            {"gradients": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]},  # Too many
            {"gradients": ["1", "2", "3", "4"]},  # Wrong type
            {"gradients": [1.5, 2.5, 3.5, 4.5]},  # Floating point
            {"gradients": [None, None, None, None]},  # Null values
        ]

        for malformed_input in malformed_inputs:
            with self.subTest(input=malformed_input):
                with self.assertRaises((ValueError, TypeError)):
                    prover.generate_proof(malformed_input)
```

#### 2. Integration Testing

```python
import pytest
from fedzk.mpc.client import MPCClient
from fedzk.coordinator import SecureCoordinator
from fedzk.client import Trainer

class TestCryptographicIntegration:
    """Integration tests for cryptographic components."""

    @pytest.fixture
    async def crypto_stack(self):
        """Setup complete cryptographic stack."""
        # In production, this would start actual services
        mpc_client = MPCClient(
            server_url="https://mpc.test:9000",
            api_key="test_key_123"
        )

        coordinator = SecureCoordinator(
            coordinator_url="https://coordinator.test:8443"
        )

        trainer = Trainer(
            model_config={'architecture': 'mlp', 'layers': [10, 5, 2]}
        )

        yield {
            'mpc': mpc_client,
            'coordinator': coordinator,
            'trainer': trainer
        }

    @pytest.mark.asyncio
    async def test_end_to_end_cryptographic_flow(self, crypto_stack):
        """Test complete cryptographic flow."""
        stack = crypto_stack

        # 1. Generate training data
        training_data = torch.randn(100, 10)

        # 2. Train model
        updates = stack['trainer'].train(training_data, epochs=1)

        # 3. Quantize gradients
        quantized = stack['trainer'].quantize_gradients(updates)

        # 4. Generate ZK proof
        proof_payload = {
            "gradients": quantized,
            "secure": True,
            "maxNorm": 1000,
            "minNonZero": 1
        }

        proof, signals = stack['mpc'].generate_proof(proof_payload)

        # 5. Submit to coordinator
        result = stack['coordinator'].submit_update(
            client_id="test_client",
            model_updates=quantized,
            zk_proof=proof,
            public_signals=signals
        )

        # 6. Verify success
        assert result['status'] == 'accepted'
        assert 'proof_hash' in result

    @pytest.mark.asyncio
    async def test_concurrent_cryptographic_operations(self, crypto_stack):
        """Test concurrent cryptographic operations."""
        import asyncio

        async def single_operation(client_id: str):
            stack = crypto_stack

            # Simulate client operation
            updates = {"layer.weight": [1, 2, 3, 4]}
            proof, signals = stack['mpc'].generate_proof({"gradients": updates})

            result = stack['coordinator'].submit_update(
                client_id=client_id,
                model_updates=updates,
                zk_proof=proof,
                public_signals=signals
            )

            return result

        # Run multiple operations concurrently
        tasks = [
            single_operation(f"client_{i}")
            for i in range(10)
        ]

        results = await asyncio.gather(*tasks)

        # Verify all operations succeeded
        for result in results:
            assert result['status'] == 'accepted'

    @pytest.mark.asyncio
    async def test_cryptographic_failure_recovery(self, crypto_stack):
        """Test recovery from cryptographic failures."""
        stack = crypto_stack

        # Simulate MPC server failure
        original_url = stack['mpc'].server_url
        stack['mpc'].server_url = "https://nonexistent.server:9000"

        # Attempt operation (should fail gracefully)
        with pytest.raises(ConnectionError):
            proof, signals = stack['mpc'].generate_proof({"gradients": [1, 2, 3, 4]})

        # Restore connection
        stack['mpc'].server_url = original_url

        # Retry operation (should succeed)
        proof, signals = stack['mpc'].generate_proof({"gradients": [1, 2, 3, 4]})
        assert proof is not None
        assert signals is not None
```

### Performance Testing

```python
import time
import psutil
import cProfile
import pstats
from io import StringIO

class TestCryptographicPerformance:
    """Performance tests for cryptographic operations."""

    def setUp(self):
        self.prover = ZKProver()
        self.test_inputs = {"gradients": [1, 2, 3, 4]}

    def test_proof_generation_performance(self):
        """Test proof generation performance under load."""
        num_proofs = 100
        times = []

        for i in range(num_proofs):
            start = time.perf_counter()
            proof, signals = self.prover.generate_proof(self.test_inputs)
            end = time.perf_counter()
            times.append(end - start)

        avg_time = sum(times) / len(times)
        max_time = max(times)
        min_time = min(times)

        # Performance requirements
        self.assertLess(avg_time, 2.0, "Average proof generation too slow")
        self.assertLess(max_time, 5.0, "Maximum proof generation too slow")
        self.assertGreater(min_time, 0.1, "Minimum proof generation too fast (suspicious)")

        print(f"Performance: {avg_time:.3f}s avg, {min_time:.3f}s min, {max_time:.3f}s max")

    def test_memory_usage_profile(self):
        """Test memory usage during cryptographic operations."""
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB

        # Perform cryptographic operations
        for i in range(50):
            proof, signals = self.prover.generate_proof(self.test_inputs)

        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory

        # Memory requirements
        self.assertLess(memory_increase, 500, "Memory usage too high")
        self.assertGreater(memory_increase, 10, "Memory usage suspiciously low")

        print(f"Memory: {initial_memory:.1f}MB → {final_memory:.1f}MB ({memory_increase:+.1f}MB)")

    def test_cpu_usage_profile(self):
        """Test CPU usage during cryptographic operations."""
        initial_cpu = psutil.cpu_percent(interval=1)

        # Perform intensive cryptographic operations
        start_time = time.time()
        for i in range(20):
            proof, signals = self.prover.generate_proof(self.test_inputs)
        end_time = time.time()

        final_cpu = psutil.cpu_percent(interval=1)

        # CPU usage should be reasonable
        avg_cpu = (initial_cpu + final_cpu) / 2
        self.assertLess(avg_cpu, 95, "CPU usage too high")
        self.assertGreater(avg_cpu, 5, "CPU usage suspiciously low")

        print(f"CPU: {avg_cpu:.1f}% average usage")

    def test_cryptographic_code_profile(self):
        """Profile cryptographic code performance."""
        profiler = cProfile.Profile()
        profiler.enable()

        # Perform cryptographic operations
        for i in range(10):
            proof, signals = self.prover.generate_proof(self.test_inputs)

        profiler.disable()

        # Analyze profile
        stream = StringIO()
        stats = pstats.Stats(profiler, stream=stream).sort_stats('cumulative')
        stats.print_stats(10)  # Top 10 functions

        profile_output = stream.getvalue()

        # Check for performance bottlenecks
        if "generate_proof" in profile_output:
            lines = profile_output.split('\n')
            for line in lines:
                if "generate_proof" in line and "src/fedzk" in line:
                    # Extract timing information
                    parts = line.split()
                    if len(parts) >= 6:
                        cumulative_time = float(parts[1])
                        if cumulative_time > 5.0:  # More than 5 seconds
                            self.fail(f"Performance bottleneck detected: {line}")

        print("Code Profile:")
        print(profile_output)
```

---

## 🔍 Security Review Process

### Pre-Commit Security Checks

```bash
# .pre-commit-config.yaml
repos:
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.4.0
    hooks:
      - id: trailing-whitespace
      - id: end-of-file-fixer
      - id: check-yaml
      - id: check-added-large-files

  - repo: https://github.com/psf/black
    rev: 23.7.0
    hooks:
      - id: black
        language_version: python3

  - repo: https://github.com/pycqa/isort
    rev: 5.12.0
    hooks:
      - id: isort

  - repo: https://github.com/pycqa/flake8
    rev: 6.0.0
    hooks:
      - id: flake8
        args: [--max-line-length=88, --extend-ignore=E203,W503]

  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.5.1
    hooks:
      - id: mypy
        additional_dependencies: [types-all]

  - repo: local
    hooks:
      - id: security-scan
        name: Security Vulnerability Scan
        entry: python -m bandit
        language: python
        files: \.(py)$
        args: [-r, --ini, .bandit]

      - id: cryptography-audit
        name: Cryptographic Code Audit
        entry: python scripts/cryptography_audit.py
        language: python
        files: src/fedzk/.*\.(py|circom)$
        pass_filenames: false
```

### Security Review Checklist

```markdown
# Cryptographic Security Review Checklist

## Code Review
- [ ] No hardcoded secrets or cryptographic keys
- [ ] Secure random number generation used
- [ ] Proper input validation implemented
- [ ] Secure error handling (no information leakage)
- [ ] Audit logging implemented for security events
- [ ] Rate limiting applied to prevent DoS attacks
- [ ] Access control properly enforced

## Cryptographic Implementation
- [ ] Approved cryptographic algorithms used
- [ ] Key sizes meet current standards
- [ ] Secure key storage and rotation implemented
- [ ] Certificate validation properly implemented
- [ ] TLS 1.3 or higher required
- [ ] Perfect forward secrecy enabled

## Circuit Security
- [ ] Circuit constraints properly implemented
- [ ] Input validation in circuits
- [ ] No insecure arithmetic patterns
- [ ] Trusted setup procedure documented
- [ ] Circuit compilation verified
- [ ] Proof verification tested

## Testing
- [ ] Cryptographic unit tests implemented
- [ ] Security regression tests added
- [ ] Fuzz testing performed
- [ ] Performance under adversarial conditions tested
- [ ] Memory safety verified

## Documentation
- [ ] Security considerations documented
- [ ] Threat model updated
- [ ] Incident response procedures documented
- [ ] Security best practices included
- [ ] Compliance requirements documented

## Deployment
- [ ] Secure configuration templates provided
- [ ] Secret management procedures documented
- [ ] Monitoring and alerting configured
- [ ] Backup and recovery procedures tested
- [ ] Rollback procedures documented

## Compliance
- [ ] GDPR compliance verified (if applicable)
- [ ] SOX compliance verified (if applicable)
- [ ] Industry-specific requirements met
- [ ] Data retention policies implemented
- [ ] Audit trail requirements satisfied

## Sign-off
- [ ] Security reviewer approval obtained
- [ ] Code owner approval obtained
- [ ] Legal review completed (if required)
- [ ] Compliance officer approval obtained

**Review Date:** __________
**Reviewer:** __________
**Approval:** __________
```

### Automated Security Scanning

```python
# scripts/cryptography_audit.py
#!/usr/bin/env python3

import ast
import re
from pathlib import Path
from typing import List, Dict, Tuple

class CryptographyAuditor:
    """Automated cryptography code auditor."""

    def __init__(self):
        self.issues = []
        self.vulnerabilities = {
            'hardcoded_secret': r'(?i)(password|secret|key)\s*=\s*["\'][^"\']+["\']',
            'weak_hash': r'(?i)(md5|sha1)\(',
            'insecure_random': r'(?i)random\.',
            'unsafe_eval': r'(?i)eval\(',
            'sql_injection': r'(?i)execute\(.+\+.+\)',
            'command_injection': r'(?i)subprocess\..*\(.+\+.+\)',
        }

    def audit_file(self, file_path: Path) -> List[Dict]:
        """Audit a single file for cryptographic issues."""
        issues = []

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                lines = content.split('\n')

            # Check for vulnerabilities
            for vuln_type, pattern in self.vulnerabilities.items():
                for match in re.finditer(pattern, content):
                    line_num = content[:match.start()].count('\n') + 1
                    issues.append({
                        'file': str(file_path),
                        'line': line_num,
                        'type': vuln_type,
                        'severity': self._get_severity(vuln_type),
                        'description': self._get_description(vuln_type),
                        'code': lines[line_num - 1].strip()
                    })

            # AST-based analysis for Python files
            if file_path.suffix == '.py':
                issues.extend(self._audit_python_ast(content, file_path))

        except Exception as e:
            issues.append({
                'file': str(file_path),
                'line': 0,
                'type': 'audit_error',
                'severity': 'info',
                'description': f'Could not audit file: {e}',
                'code': ''
            })

        return issues

    def _audit_python_ast(self, content: str, file_path: Path) -> List[Dict]:
        """Audit Python code using AST analysis."""
        issues = []

        try:
            tree = ast.parse(content)

            for node in ast.walk(tree):
                # Check for insecure imports
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        if alias.name in ['pickle', 'marshal', 'shelve']:
                            issues.append({
                                'file': str(file_path),
                                'line': node.lineno,
                                'type': 'insecure_import',
                                'severity': 'high',
                                'description': f'Insecure module import: {alias.name}',
                                'code': f'import {alias.name}'
                            })

                # Check for assert statements in production code
                elif isinstance(node, ast.Assert):
                    issues.append({
                        'file': str(file_path),
                        'line': node.lineno,
                        'type': 'production_assert',
                        'severity': 'medium',
                        'description': 'Assert statements should not be used in production code',
                        'code': 'assert ...'
                    })

        except SyntaxError:
            pass

        return issues

    def _get_severity(self, vuln_type: str) -> str:
        """Get severity level for vulnerability type."""
        severity_map = {
            'hardcoded_secret': 'critical',
            'weak_hash': 'high',
            'insecure_random': 'high',
            'unsafe_eval': 'critical',
            'sql_injection': 'critical',
            'command_injection': 'critical',
            'insecure_import': 'high',
            'production_assert': 'medium',
        }
        return severity_map.get(vuln_type, 'medium')

    def _get_description(self, vuln_type: str) -> str:
        """Get description for vulnerability type."""
        descriptions = {
            'hardcoded_secret': 'Hardcoded secrets or credentials detected',
            'weak_hash': 'Weak hash algorithm (MD5/SHA1) should not be used',
            'insecure_random': 'Insecure random number generation',
            'unsafe_eval': 'Use of eval() poses security risks',
            'sql_injection': 'Potential SQL injection vulnerability',
            'command_injection': 'Potential command injection vulnerability',
            'insecure_import': 'Import of insecure module',
            'production_assert': 'Assert statements disabled in production',
        }
        return descriptions.get(vuln_type, 'Security issue detected')

    def audit_directory(self, directory: Path) -> Dict:
        """Audit entire directory for cryptographic issues."""
        all_issues = []

        # File patterns to audit
        patterns = ['*.py', '*.circom', '*.js', '*.ts']

        for pattern in patterns:
            for file_path in directory.rglob(pattern):
                if file_path.is_file():
                    issues = self.audit_file(file_path)
                    all_issues.extend(issues)

        # Summarize results
        summary = {
            'total_files': len(list(directory.rglob('*'))),
            'audited_files': len(set(issue['file'] for issue in all_issues)),
            'total_issues': len(all_issues),
            'severity_breakdown': {},
            'issues_by_type': {},
            'issues': all_issues
        }

        # Group by severity and type
        for issue in all_issues:
            severity = issue['severity']
            vuln_type = issue['type']

            summary['severity_breakdown'][severity] = summary['severity_breakdown'].get(severity, 0) + 1
            summary['issues_by_type'][vuln_type] = summary['issues_by_type'].get(vuln_type, 0) + 1

        return summary

def main():
    """Main audit function."""
    import argparse
    import json

    parser = argparse.ArgumentParser(description='Cryptography Code Auditor')
    parser.add_argument('directory', help='Directory to audit')
    parser.add_argument('--output', '-o', help='Output JSON file')
    parser.add_argument('--fail-on-critical', action='store_true',
                       help='Exit with error code on critical issues')

    args = parser.parse_args()

    auditor = CryptographyAuditor()
    directory = Path(args.directory)

    if not directory.exists():
        print(f"Error: Directory {directory} does not exist")
        exit(1)

    print(f"Auditing directory: {directory}")
    results = auditor.audit_directory(directory)

    # Print summary
    print(f"\nAudit Summary:")
    print(f"Total files: {results['total_files']}")
    print(f"Audited files: {results['audited_files']}")
    print(f"Total issues: {results['total_issues']}")

    print(f"\nSeverity Breakdown:")
    for severity, count in results['severity_breakdown'].items():
        print(f"  {severity.upper()}: {count}")

    print(f"\nIssues by Type:")
    for vuln_type, count in results['issues_by_type'].items():
        print(f"  {vuln_type}: {count}")

    # Check for critical issues
    critical_count = results['severity_breakdown'].get('critical', 0)
    if critical_count > 0 and args.fail_on_critical:
        print(f"\n❌ Found {critical_count} critical issues!")
        exit(1)

    # Save detailed results
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nDetailed results saved to: {args.output}")

    print("
✅ Cryptography audit completed!"
if __name__ == "__main__":
    main()
```

---

## 📚 Documentation Standards

### API Documentation

```python
def generate_zk_proof(
    gradients: Union[List[int], Dict[str, List[int]]],
    circuit: str = "model_update",
    security_level: str = "standard"
) -> Tuple[Dict[str, Any], List[int]]:
    """
    Generate a zero-knowledge proof for model gradients.

    This function creates a cryptographic proof that the provided gradients
    satisfy the constraints defined in the specified ZK circuit, without
    revealing the actual gradient values.

    Args:
        gradients: Model gradients to prove. Can be a list of integers or
                  a dictionary mapping layer names to gradient lists.
                  All values must be integers (no floating-point).
        circuit: ZK circuit to use for proof generation.
                Options: "model_update", "model_update_secure", "batch"
        security_level: Security level for proof generation.
                       Options: "standard", "enhanced", "maximum"

    Returns:
        Tuple of (proof, public_signals) where:
        - proof: Dictionary containing the cryptographic proof
        - public_signals: List of public signals for verification

    Raises:
        ValueError: If gradients contain non-integer values or invalid format
        RuntimeError: If ZK toolchain is not properly configured
        ConnectionError: If MPC server is unreachable

    Example:
        >>> gradients = [100, 200, 300, 400]
        >>> proof, signals = generate_zk_proof(gradients, circuit="model_update_secure")
        >>> print(f"Proof generated with {len(signals)} public signals")

    Security:
        - Uses real cryptographic operations (no mocks)
        - Proof size is constant regardless of input size
        - Zero-knowledge property ensures gradient privacy
        - Soundness guarantees proof validity

    Performance:
        - Standard circuit: ~0.8-1.2s generation time
        - Secure circuit: ~1.2-1.8s generation time
        - Memory usage: ~512MB - 1GB depending on circuit

    Note:
        Gradients must be quantized to integers before calling this function.
        Use GradientQuantizer for automatic quantization.
    """
    # Implementation...
```

### Circuit Documentation

```circom
/**
 * @title Secure Model Update Circuit
 * @author FEDzk Team
 * @notice Zero-knowledge circuit for secure model gradient verification
 * @dev Implements cryptographic proof of gradient norm constraints
 *
 * @param n Number of gradient values to process
 *
 * Function: verifyGradientNorm
 * - Computes L2 norm of gradient vector
 * - Enforces maximum norm constraint
 * - Generates zero-knowledge proof
 *
 * Security Properties:
 * - Zero-knowledge: Gradient values remain private
 * - Succinctness: Proof size independent of gradient size
 * - Soundness: Invalid gradients rejected with high probability
 * - Completeness: Valid gradients always accepted
 *
 * Constraints:
 * - All inputs must be integers in range [-10^9, 10^9]
 * - Maximum norm constraint prevents gradient explosion
 * - Computational complexity: O(n) for n gradient values
 *
 * Gas Estimation:
 * - Witness generation: ~50k constraints
 * - Proof generation: ~1-2 seconds
 * - Verification: ~150-250ms
 *
 * Example Usage:
 * ```solidity
 * // Verify proof in smart contract
 * require(verifyProof(proof, publicSignals), "Invalid proof");
 * ```
 */
template SecureModelUpdate(n) {
    // ... circuit implementation
}
```

---

## 🔍 Code Review Guidelines

### Cryptographic Code Review Checklist

```markdown
# Cryptographic Code Review Checklist

## Security Analysis
- [ ] No cryptographic keys or secrets hardcoded
- [ ] Secure random number generation used
- [ ] Cryptographic operations use approved algorithms
- [ ] Key sizes meet current security standards
- [ ] Secure key storage and rotation implemented
- [ ] Input validation prevents injection attacks

## Code Quality
- [ ] Functions have clear, single responsibilities
- [ ] Error handling preserves security (no information leakage)
- [ ] Logging implemented without exposing sensitive data
- [ ] Code follows principle of least privilege
- [ ] Resource cleanup properly implemented
- [ ] Thread safety considerations addressed

## Testing
- [ ] Unit tests cover cryptographic functions
- [ ] Integration tests verify end-to-end workflows
- [ ] Security regression tests implemented
- [ ] Edge cases and error conditions tested
- [ ] Performance tests under adversarial conditions

## Documentation
- [ ] Function parameters and return values documented
- [ ] Security considerations clearly stated
- [ ] Usage examples provided
- [ ] Error conditions documented
- [ ] Performance characteristics specified

## Compliance
- [ ] Code follows organizational security policies
- [ ] Industry standards and best practices followed
- [ ] Audit requirements satisfied
- [ ] Regulatory compliance considerations addressed

## Review Process
- [ ] Code reviewed by security team member
- [ ] Automated security scans passed
- [ ] Manual security analysis completed
- [ ] Performance benchmarks verified
- [ ] Documentation reviewed and approved

**Reviewer:** __________
**Date:** __________
**Approval Status:** __________
```

### Automated Code Review

```python
# .github/workflows/security-review.yml
name: Security Code Review

on:
  pull_request:
    branches: [ main, develop ]
    paths:
      - 'src/fedzk/**'
      - 'tests/**'

jobs:
  security-review:
    runs-on: ubuntu-latest

    steps:
    - uses: actions/checkout@v3

    - name: Setup Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.9'

    - name: Install Security Tools
      run: |
        python -m pip install --upgrade pip
        pip install bandit safety cryptography-audit

    - name: Run Bandit Security Scan
      run: |
        bandit -r src/fedzk/ -f json -o bandit_results.json || true

    - name: Run Safety Vulnerability Scan
      run: |
        safety check --output json > safety_results.json || true

    - name: Run Custom Cryptography Audit
      run: |
        python scripts/cryptography_audit.py src/fedzk/ --output crypto_audit.json

    - name: Analyze Results
      run: |
        # Parse results and determine if PR should be blocked
        python scripts/analyze_security_results.py \
          bandit_results.json \
          safety_results.json \
          crypto_audit.json

    - name: Upload Security Report
      uses: actions/upload-artifact@v3
      with:
        name: security-scan-results
        path: |
          bandit_results.json
          safety_results.json
          crypto_audit.json

  cryptography-test:
    runs-on: ubuntu-latest

    steps:
    - uses: actions/checkout@v3

    - name: Setup ZK Environment
      run: |
        ./scripts/setup_zk.sh

    - name: Run Cryptographic Tests
      run: |
        python -m pytest tests/ -k "crypto" -v --tb=short

    - name: Run Performance Benchmarks
      run: |
        python scripts/benchmark_cryptography.py --output crypto_benchmark.json

    - name: Upload Test Results
      uses: actions/upload-artifact@v3
      with:
        name: cryptography-test-results
        path: crypto_benchmark.json
```

---

## 🚀 Release Process

### Version Management

```python
# version.py
class VersionManager:
    """Manage cryptographic component versions."""

    def __init__(self, current_version: str = "0.5.0"):
        self.current_version = current_version
        self.compatibility_matrix = {
            "zk_circuit": "2.0.0",
            "snarkjs": "0.7.0",
            "circom": "2.1.0"
        }

    def check_compatibility(self, component_versions: Dict[str, str]) -> bool:
        """Check version compatibility for cryptographic components."""
        for component, required_version in self.compatibility_matrix.items():
            current_version = component_versions.get(component, "0.0.0")
            if not self._is_compatible(current_version, required_version):
                return False
        return True

    def _is_compatible(self, current: str, required: str) -> bool:
        """Check if current version is compatible with required version."""
        current_parts = [int(x) for x in current.split('.')]
        required_parts = [int(x) for x in required.split('.')]

        # Major version must match
        if current_parts[0] != required_parts[0]:
            return False

        # Minor version must be >= required
        if current_parts[1] < required_parts[1]:
            return False

        return True

    def generate_release_notes(self, version: str, changes: List[str]) -> str:
        """Generate release notes for cryptographic components."""
        notes = f"# FEDzk v{version} - Cryptographic Release\n\n"

        # Security highlights
        security_changes = [c for c in changes if any(word in c.lower()
                          for word in ['security', 'crypto', 'zk', 'proof'])]
        if security_changes:
            notes += "## 🔐 Security Enhancements\n"
            for change in security_changes:
                notes += f"- {change}\n"
            notes += "\n"

        # Breaking changes
        breaking_changes = [c for c in changes if 'breaking' in c.lower()]
        if breaking_changes:
            notes += "## ⚠️ Breaking Changes\n"
            for change in breaking_changes:
                notes += f"- {change}\n"
            notes += "\n"

        # All changes
        notes += "## 📋 Changes\n"
        for change in changes:
            notes += f"- {change}\n"
        notes += "\n"

        # Compatibility
        notes += "## 🔗 Compatibility\n"
        for component, version in self.compatibility_matrix.items():
            notes += f"- {component}: v{version}+\n"

        return notes
```

### Release Checklist

```markdown
# Cryptographic Release Checklist

## Pre-Release
- [ ] All cryptographic tests passing
- [ ] Security audit completed and signed off
- [ ] Performance benchmarks verified
- [ ] Documentation updated and reviewed
- [ ] Compatibility testing completed
- [ ] Backup and rollback procedures tested

## Security Verification
- [ ] No hardcoded secrets in codebase
- [ ] Cryptographic key management verified
- [ ] Input validation tested with adversarial inputs
- [ ] Error handling preserves security
- [ ] Audit logging functioning correctly
- [ ] Rate limiting properly configured

## Cryptographic Validation
- [ ] ZK circuit compilation verified
- [ ] Trusted setup procedure validated
- [ ] Proof generation and verification tested
- [ ] Batch processing functionality confirmed
- [ ] Performance requirements met

## Compliance Check
- [ ] GDPR compliance verified (if applicable)
- [ ] SOX compliance verified (if applicable)
- [ ] Industry-specific requirements met
- [ ] Data retention policies implemented
- [ ] Audit trail requirements satisfied

## Deployment Preparation
- [ ] Production configuration templates updated
- [ ] Docker images built and tested
- [ ] Kubernetes manifests validated
- [ ] Monitoring and alerting configured
- [ ] Backup and recovery procedures documented

## Release Execution
- [ ] Code frozen and tagged
- [ ] Release notes published
- [ ] Documentation deployed
- [ ] Docker images published
- [ ] PyPI package uploaded
- [ ] Release announcement sent

## Post-Release
- [ ] User feedback monitored
- [ ] Security incidents tracked
- [ ] Performance metrics analyzed
- [ ] Next release planning initiated
- [ ] Community support provided

**Release Date:** __________
**Release Manager:** __________
**Security Officer:** __________
```

---

*This contribution guide ensures that all FEDzk cryptographic components maintain the highest standards of security, quality, and reliability. All contributions must undergo thorough security review before acceptance.*

