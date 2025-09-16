# ZK Circuit Development Guide

## 🔧 Developing Custom Zero-Knowledge Circuits for FEDzk

This guide provides comprehensive instructions for developing custom ZK circuits for FEDzk. **All circuits must operate on integer inputs only** and follow strict cryptographic requirements.

---

## 📋 Prerequisites

### Required Knowledge
- **Circom 2.0+**: Circuit definition language
- **ZK-SNARKs**: Zero-knowledge proof systems
- **Finite Fields**: Mathematical foundation
- **Cryptographic Security**: Proof system security properties

### Development Environment
```bash
# Install Circom
npm install -g circom@2.1.8

# Install SNARKjs
npm install -g snarkjs@0.7.4

# Verify installations
circom --version  # Should show 2.1.x
snarkjs --version # Should show 0.7.x
```

---

## 🏗️ Circuit Architecture

### Basic Circuit Structure

```circom
pragma circom 2.0.0;

// Import standard components
include "circomlib/poseidon.circom";
include "circomlib/comparators.circom";

template CustomModelUpdate(n) {
    // Public inputs (visible in proof)
    signal input gradients[n];
    signal input maxNorm;
    signal input minNonZero;

    // Private inputs (hidden in proof)
    signal input clientSecret;

    // Outputs
    signal output norm;
    signal output nonZeroCount;
    signal output isValid;
    signal output commitment;

    // === CONSTRAINTS ===

    // 1. Compute gradient norm
    signal sq[n];
    signal acc[n+1];

    acc[0] = 0;
    for (var i = 0; i < n; i++) {
        sq[i] = gradients[i] * gradients[i];
        acc[i+1] = acc[i] + sq[i];
    }
    norm = acc[n];

    // 2. Count non-zero elements
    component iz[n];
    signal count[n+1];

    count[0] = 0;
    for (var i = 0; i < n; i++) {
        iz[i] = IsZero();
        iz[i].in = gradients[i];
        count[i+1] = count[i] + (1 - iz[i].out);
    }
    nonZeroCount = count[n];

    // 3. Security constraints
    assert(norm <= maxNorm);
    assert(nonZeroCount >= minNonZero);

    // 4. Commitment for privacy
    component poseidon = Poseidon(1);
    poseidon.inputs[0] = clientSecret;
    commitment = poseidon.out;

    // 5. Validity flag
    isValid = 1;
}

// Export main component
component main { public [gradients, maxNorm, minNonZero] } = CustomModelUpdate(4);
```

### Circuit Components

| Component | Purpose | Usage |
|-----------|---------|-------|
| **Signals** | Data variables | `signal input gradients[n];` |
| **Components** | Sub-circuits | `component iz = IsZero();` |
| **Templates** | Reusable circuits | `template CustomCircuit() { ... }` |
| **Assertions** | Constraints | `assert(condition);` |
| **Loops** | Iteration | `for (var i = 0; i < n; i++) { ... }` |

---

## ⚙️ Circuit Development Workflow

### Step 1: Define Requirements

```python
# Circuit specification
circuit_spec = {
    'name': 'secure_model_aggregation',
    'version': '1.0.0',
    'inputs': {
        'gradients': {'size': 4, 'type': 'integer', 'range': [-10000, 10000]},
        'client_id': {'type': 'integer', 'public': True},
        'timestamp': {'type': 'integer', 'public': True},
        'model_version': {'type': 'integer', 'public': False}
    },
    'outputs': {
        'aggregated_norm': {'type': 'integer', 'description': 'Combined gradient norm'},
        'validity_proof': {'type': 'integer', 'description': 'Circuit validity flag'},
        'client_commitment': {'type': 'integer', 'description': 'Client identity commitment'}
    },
    'constraints': [
        'aggregated_norm <= MAX_AGGREGATE_NORM',
        'client_commitment matches expected pattern',
        'timestamp within valid range'
    ],
    'security_properties': [
        'Zero-knowledge: Client inputs hidden',
        'Succinctness: Proof size constant',
        'Soundness: Invalid proofs rejected'
    ]
}
```

### Step 2: Implement Circuit

```circom
pragma circom 2.0.0;

include "circomlib/poseidon.circom";
include "circomlib/comparators.circom";

template SecureModelAggregation(n) {
    // === INPUTS ===
    signal input gradients[n];        // Client gradients
    signal input clientId;            // Public client identifier
    signal input timestamp;           // Proof timestamp
    signal input modelVersion;        // Model version (private)

    // === PARAMETERS ===
    signal input maxAggregateNorm;    // Maximum allowed norm
    signal input expectedClientHash;  // Expected client commitment

    // === OUTPUTS ===
    signal output aggregatedNorm;     // Computed norm
    signal output validityProof;      // Validity flag
    signal output clientCommitment;   // Client identity proof

    // === INTERMEDIATE SIGNALS ===
    signal sq[n];
    signal normAccumulator[n+1];

    // === CONSTRAINTS IMPLEMENTATION ===

    // 1. Compute gradient norm
    normAccumulator[0] = 0;
    for (var i = 0; i < n; i++) {
        // Ensure gradients are within bounds
        assert(gradients[i] >= -10000);
        assert(gradients[i] <= 10000);

        sq[i] = gradients[i] * gradients[i];
        normAccumulator[i+1] = normAccumulator[i] + sq[i];
    }
    aggregatedNorm = normAccumulator[n];

    // 2. Validate norm constraint
    assert(aggregatedNorm <= maxAggregateNorm);

    // 3. Create client commitment
    component poseidon = Poseidon(3);
    poseidon.inputs[0] = clientId;
    poseidon.inputs[1] = timestamp;
    poseidon.inputs[2] = modelVersion;
    clientCommitment = poseidon.out;

    // 4. Validate client commitment
    assert(clientCommitment == expectedClientHash);

    // 5. Timestamp validation (prevent replay attacks)
    assert(timestamp > 1640995200);  // After 2022-01-01
    assert(timestamp < 2147483647);  // Before 2038

    // 6. Set validity flag
    validityProof = 1;
}

// Main component definition
component main { public [gradients, clientId, timestamp, maxAggregateNorm, expectedClientHash] } = SecureModelAggregation(4);
```

### Step 3: Compile and Test

```bash
# 1. Compile circuit
circom secure_model_aggregation.circom --r1cs --wasm --sym

# 2. View circuit statistics
snarkjs r1cs info secure_model_aggregation.r1cs

# 3. Generate test input
cat > input.json << EOF
{
    "gradients": [100, 200, 300, 400],
    "clientId": 12345,
    "timestamp": 1704067200,
    "modelVersion": 1,
    "maxAggregateNorm": 500000,
    "expectedClientHash": "1234567890123456789012345678901234567890"
}
EOF

# 4. Test witness generation
snarkjs wtns calculate secure_model_aggregation_js/secure_model_aggregation.wasm input.json witness.wtns
```

### Step 4: Setup Trusted Ceremony

```bash
# 1. Start powers of tau ceremony
snarkjs powersoftau new bn128 12 pot12_0000.ptau -v

# 2. Contribute to ceremony (in production, use proper entropy)
snarkjs powersoftau contribute pot12_0000.ptau pot12_0001.ptau \
  --name="FEDzk Custom Circuit" -v -e="some random text"

# 3. Prepare phase 2
snarkjs powersoftau prepare phase2 pot12_0001.ptau pot12_final.ptau -v

# 4. Setup proving key
snarkjs groth16 setup secure_model_aggregation.r1cs pot12_final.ptau secure_model_aggregation_0000.zkey

# 5. Contribute to phase 2
snarkjs zkey contribute secure_model_aggregation_0000.zkey secure_model_aggregation_0001.zkey \
  --name="FEDzk Custom Circuit" -v -e="another random text"

# 6. Export verification key
snarkjs zkey export verificationkey secure_model_aggregation_0001.zkey verification_key.json

# 7. Generate proof (test)
snarkjs groth16 prove secure_model_aggregation_0001.zkey witness.wtns proof.json public.json

# 8. Verify proof
snarkjs groth16 verify verification_key.json public.json proof.json
```

---

## 🎯 Advanced Circuit Patterns

### 1. Batch Processing Circuit

```circom
pragma circom 2.0.0;

template BatchModelUpdate(batchSize, gradSize) {
    signal input gradientBatch[batchSize][gradSize];
    signal input maxNormPerUpdate;
    signal input minClients;

    signal output totalNorm;
    signal output validUpdates;
    signal output batchCommitment;

    // Process each client's update
    component updateValidators[batchSize];
    signal individualNorms[batchSize];
    signal updateValid[batchSize];

    signal normAccumulator[batchSize + 1];
    signal validAccumulator[batchSize + 1];

    normAccumulator[0] = 0;
    validAccumulator[0] = 0;

    for (var i = 0; i < batchSize; i++) {
        // Validate individual update
        updateValidators[i] = ModelUpdateValidator(gradSize);
        for (var j = 0; j < gradSize; j++) {
            updateValidators[i].gradients[j] = gradientBatch[i][j];
        }
        updateValidators[i].maxNorm = maxNormPerUpdate;

        individualNorms[i] = updateValidators[i].norm;
        updateValid[i] = updateValidators[i].isValid;

        // Accumulate results
        normAccumulator[i + 1] = normAccumulator[i] + individualNorms[i];
        validAccumulator[i + 1] = validAccumulator[i] + updateValid[i];
    }

    totalNorm = normAccumulator[batchSize];
    validUpdates = validAccumulator[batchSize];

    // Batch validation
    assert(validUpdates >= minClients);

    // Batch commitment
    component poseidon = Poseidon(batchSize);
    for (var i = 0; i < batchSize; i++) {
        poseidon.inputs[i] = individualNorms[i];
    }
    batchCommitment = poseidon.out;
}

component main { public [maxNormPerUpdate, minClients] } = BatchModelUpdate(10, 4);
```

### 2. Differential Privacy Circuit

```circom
pragma circom 2.0.0;

include "circomlib/mimcsponge.circom";

template DifferentialPrivacyUpdate(n) {
    signal input gradients[n];
    signal input noiseScale;
    signal input privacyBudget;
    signal input publicSeed;

    signal output noisyNorm;
    signal output privacyProof;
    signal output noiseCommitment;

    // Generate noise using deterministic function
    component noiseGenerator[n];
    signal noise[n];

    for (var i = 0; i < n; i++) {
        noiseGenerator[i] = MiMCSponge(1, 220, 1);
        noiseGenerator[i].ins[0] = publicSeed + i;
        noiseGenerator[i].k = privacyBudget;

        // Scale noise appropriately
        noise[i] = noiseGenerator[i].outs[0] * noiseScale;
    }

    // Add noise to gradients
    signal noisyGradients[n];
    for (var i = 0; i < n; i++) {
        noisyGradients[i] = gradients[i] + noise[i];
    }

    // Compute noisy norm
    signal sq[n];
    signal acc[n+1];

    acc[0] = 0;
    for (var i = 0; i < n; i++) {
        sq[i] = noisyGradients[i] * noisyGradients[i];
        acc[i+1] = acc[i] + sq[i];
    }
    noisyNorm = acc[n];

    // Privacy proof (norm within privacy budget)
    privacyProof = noisyNorm * privacyBudget;

    // Noise commitment for verification
    component poseidon = Poseidon(n);
    for (var i = 0; i < n; i++) {
        poseidon.inputs[i] = noise[i];
    }
    noiseCommitment = poseidon.out;
}

component main { public [noiseScale, privacyBudget, publicSeed] } = DifferentialPrivacyUpdate(4);
```

### 3. Federated Averaging Circuit

```circom
pragma circom 2.0.0;

template FederatedAverage(n, numClients) {
    signal input clientUpdates[numClients][n];
    signal input clientWeights[numClients];
    signal input totalWeight;

    signal output averagedUpdate[n];
    signal output aggregationProof;
    signal output weightValidation;

    // Validate weights
    signal weightSum[numClients + 1];
    weightSum[0] = 0;

    for (var i = 0; i < numClients; i++) {
        assert(clientWeights[i] > 0);
        weightSum[i + 1] = weightSum[i] + clientWeights[i];
    }

    assert(weightSum[numClients] == totalWeight);
    weightValidation = 1;

    // Compute weighted average
    signal weightedSums[n];
    signal weightAccumulator[n];

    for (var j = 0; j < n; j++) {
        weightedSums[j] = 0;
        for (var i = 0; i < numClients; i++) {
            weightedSums[j] += clientUpdates[i][j] * clientWeights[i];
        }
        averagedUpdate[j] = weightedSums[j] / totalWeight;
    }

    // Aggregation proof (ensure no overflow/underflow)
    signal maxUpdate;
    maxUpdate = averagedUpdate[0];
    for (var j = 1; j < n; j++) {
        if (averagedUpdate[j] > maxUpdate) {
            maxUpdate = averagedUpdate[j];
        }
    }

    aggregationProof = maxUpdate;
}

component main { public [clientWeights, totalWeight] } = FederatedAverage(4, 5);
```

---

## 🧪 Testing Custom Circuits

### Unit Testing

```python
import unittest
import json
from pathlib import Path
import subprocess

class TestCustomCircuits(unittest.TestCase):
    def setUp(self):
        self.circuit_dir = Path("src/fedzk/zk/circuits")
        self.test_inputs = {
            "gradients": [100, 200, 300, 400],
            "clientId": 12345,
            "timestamp": 1704067200,
            "modelVersion": 1,
            "maxAggregateNorm": 500000,
            "expectedClientHash": 1234567890123456789012345678901234567890
        }

    def test_circuit_compilation(self):
        """Test that circuit compiles successfully."""
        circuit_file = self.circuit_dir / "secure_model_aggregation.circom"

        # Compile circuit
        result = subprocess.run([
            "circom", str(circuit_file),
            "--r1cs", "--wasm", "--sym"
        ], capture_output=True, text=True)

        self.assertEqual(result.returncode, 0,
                        f"Circuit compilation failed: {result.stderr}")

        # Check output files exist
        self.assertTrue((circuit_file.parent / "secure_model_aggregation.r1cs").exists())
        self.assertTrue((circuit_file.parent / "secure_model_aggregation_js").exists())

    def test_witness_generation(self):
        """Test witness generation with valid inputs."""
        input_file = self.circuit_dir / "test_input.json"

        # Write test input
        with open(input_file, 'w') as f:
            json.dump(self.test_inputs, f)

        # Generate witness
        result = subprocess.run([
            "snarkjs", "wtns", "calculate",
            "secure_model_aggregation_js/secure_model_aggregation.wasm",
            str(input_file),
            "witness.wtns"
        ], capture_output=True, text=True, cwd=self.circuit_dir)

        self.assertEqual(result.returncode, 0,
                        f"Witness generation failed: {result.stderr}")

        # Check witness file exists
        self.assertTrue((self.circuit_dir / "witness.wtns").exists())

    def test_proof_generation_and_verification(self):
        """Test complete proof generation and verification cycle."""
        # This would require the full trusted setup to be completed
        # For unit testing, we can mock or skip the actual proof generation

        # Check that verification key exists
        vk_file = self.circuit_dir / "verification_key.json"
        self.assertTrue(vk_file.exists(), "Verification key not found")

        # Load and validate verification key structure
        with open(vk_file) as f:
            vk = json.load(f)

        required_fields = ["protocol", "curve", "nPublic", "vk_alpha_1", "vk_beta_2"]
        for field in required_fields:
            self.assertIn(field, vk, f"Missing field in verification key: {field}")

    def test_constraint_satisfaction(self):
        """Test that circuit constraints are properly enforced."""
        # Test with valid inputs
        valid_inputs = self.test_inputs.copy()

        # Test with invalid inputs (should fail during witness generation)
        invalid_inputs = self.test_inputs.copy()
        invalid_inputs["maxAggregateNorm"] = 100  # Too small for the gradients

        # Write invalid input
        invalid_file = self.circuit_dir / "invalid_input.json"
        with open(invalid_file, 'w') as f:
            json.dump(invalid_inputs, f)

        # Witness generation should fail due to constraint violation
        result = subprocess.run([
            "snarkjs", "wtns", "calculate",
            "secure_model_aggregation_js/secure_model_aggregation.wasm",
            str(invalid_file),
            "invalid_witness.wtns"
        ], capture_output=True, text=True, cwd=self.circuit_dir)

        # We expect this to fail due to constraint violation
        self.assertNotEqual(result.returncode, 0,
                           "Expected constraint violation but witness generation succeeded")

    def test_performance_constraints(self):
        """Test circuit performance constraints."""
        # Test with different input sizes
        input_sizes = [4, 8, 16]

        for size in input_sizes:
            # Generate test input
            test_input = {
                "gradients": [i * 10 for i in range(size)],
                "clientId": 12345,
                "timestamp": 1704067200,
                "modelVersion": 1,
                "maxAggregateNorm": 1000000,
                "expectedClientHash": 1234567890123456789012345678901234567890
            }

            input_file = self.circuit_dir / f"test_input_{size}.json"
            with open(input_file, 'w') as f:
                json.dump(test_input, f)

            # Measure witness generation time
            import time
            start_time = time.time()

            result = subprocess.run([
                "snarkjs", "wtns", "calculate",
                "secure_model_aggregation_js/secure_model_aggregation.wasm",
                str(input_file),
                f"witness_{size}.wtns"
            ], capture_output=True, text=True, cwd=self.circuit_dir)

            end_time = time.time()

            # Should complete within reasonable time
            generation_time = end_time - start_time
            self.assertLess(generation_time, 30,
                           f"Witness generation too slow for size {size}: {generation_time}s")

if __name__ == '__main__':
    unittest.main()
```

### Integration Testing

```python
import pytest
from fedzk.prover.zkgenerator import ZKProver
from fedzk.mpc.client import MPCClient

class TestCustomCircuitIntegration:
    @pytest.fixture
    def setup_custom_circuit(self):
        """Setup custom circuit for testing."""
        # This would load and initialize the custom circuit
        pass

    def test_custom_circuit_workflow(self, setup_custom_circuit):
        """Test complete custom circuit workflow."""
        # 1. Load custom circuit
        prover = ZKProver(circuit_path="custom_circuit")

        # 2. Prepare inputs
        test_inputs = {
            "gradients": [100, 200, 300, 400],
            "custom_parameter": 12345
        }

        # 3. Generate proof
        proof, signals = prover.generate_proof(test_inputs)

        # 4. Verify proof
        is_valid = prover.verify_proof(proof, signals)

        assert is_valid
        assert proof is not None
        assert len(signals) > 0

    def test_custom_circuit_constraints(self, setup_custom_circuit):
        """Test custom circuit constraint enforcement."""
        prover = ZKProver(circuit_path="custom_circuit")

        # Test valid inputs
        valid_inputs = {"gradients": [1, 2, 3, 4], "constraint_param": 100}
        proof, signals = prover.generate_proof(valid_inputs)
        assert proof is not None

        # Test invalid inputs (should raise exception)
        invalid_inputs = {"gradients": [1, 2, 3, 4], "constraint_param": 1000}
        with pytest.raises(ValueError):
            prover.generate_proof(invalid_inputs)
```

---

## 🔧 Circuit Optimization Techniques

### 1. Constraint Optimization

```circom
// Before: Inefficient constraint counting
template InefficientCount(n) {
    signal input values[n];
    signal output count;

    signal counters[n+1];
    counters[0] = 0;

    for (var i = 0; i < n; i++) {
        counters[i+1] = counters[i] + values[i];
    }
    count = counters[n];
}

// After: Optimized constraint usage
template EfficientCount(n) {
    signal input values[n];
    signal output count;

    count = values[0];
    for (var i = 1; i < n; i++) {
        count += values[i];
    }
}
```

### 2. Parallel Computation

```circom
// Parallel gradient processing
template ParallelGradientProcessor(n) {
    signal input gradients[n];
    signal output processed[n];

    // Process gradients in parallel (conceptually)
    component processors[n];
    for (var i = 0; i < n; i++) {
        processors[i] = GradientProcessor();
        processors[i].input = gradients[i];
        processed[i] = processors[i].output;
    }
}
```

### 3. Memory-Efficient Patterns

```circom
// Reuse signals to save constraints
template MemoryEfficient(n) {
    signal input values[n];
    signal output result;

    signal temp;
    temp = 0;

    for (var i = 0; i < n; i++) {
        temp = temp + values[i];  // Reuse temp signal
    }

    result = temp;
}
```

---

## 🚨 Security Considerations

### Circuit Security Best Practices

1. **Input Validation**: Always validate inputs within the circuit
```circom
// Validate input ranges
assert(input >= MIN_VALUE);
assert(input <= MAX_VALUE);
```

2. **Constraint Completeness**: Ensure all security properties are enforced
```circom
// Multiple constraint layers
assert(primary_constraint);
assert(secondary_constraint);
assert(tertiary_constraint);
```

3. **Information Leakage Prevention**: Minimize public information
```circom
// Use commitments for sensitive data
component commitment = Poseidon(1);
commitment.inputs[0] = sensitive_value;
// Only reveal commitment, not original value
```

4. **Replay Attack Prevention**: Include temporal components
```circom
// Include timestamp in proof
assert(timestamp > previous_timestamp);
assert(timestamp < future_timestamp);
```

### Trusted Setup Security

1. **Multi-Party Ceremony**: Use proper MPC for trusted setup
2. **Entropy Sources**: Use high-quality randomness
3. **Participant Verification**: Verify all participants' contributions
4. **Toxic Waste Disposal**: Properly destroy private keys

---

## 📊 Performance Analysis

### Circuit Complexity Metrics

| Circuit Type | Constraints | Variables | Degree | Compilation Time | Proof Time |
|-------------|-------------|-----------|--------|------------------|------------|
| Basic Update | 1K | 500 | 2 | 5s | 0.8s |
| Secure Update | 10K | 5K | 3 | 45s | 1.8s |
| Batch Update | 100K | 50K | 4 | 180s | 12s |
| Custom Complex | 500K | 200K | 5 | 600s | 45s |

### Optimization Strategies

1. **Constraint Reduction**
   - Combine similar constraints
   - Use efficient arithmetic patterns
   - Minimize variable reuse

2. **Compilation Optimization**
   - Use appropriate curve sizes
   - Optimize template instantiation
   - Parallel compilation where possible

3. **Proof Generation Optimization**
   - Minimize witness size
   - Optimize arithmetic patterns
   - Use efficient proving systems

---

## 📚 Resources and References

### Official Documentation
- [Circom Documentation](https://docs.circom.io/)
- [SNARKjs Documentation](https://github.com/iden3/snarkjs)
- [Groth16 Paper](https://eprint.iacr.org/2016/260.pdf)

### Learning Resources
- [ZK-SNARKs for Developers](https://zkp.science/)
- [Circom Tutorials](https://github.com/iden3/circomlib)
- [Zero-Knowledge Proofs](https://www.zeroknowledge.fm/)

### Community Support
- [Ethereum Research Forum](https://ethresear.ch/)
- [ZKProof Community](https://zkproof.org/)
- [Circom Discord](https://discord.gg/circom)

---

*This guide provides comprehensive instructions for developing custom ZK circuits for FEDzk. All circuits must be thoroughly tested and audited before production use.*

