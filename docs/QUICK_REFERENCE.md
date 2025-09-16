# FEDzk Quick Reference

## 🚀 Getting Started

### Prerequisites
```bash
# Required tools
- Node.js 16+
- circom 2.0+
- snarkjs 0.7+
- Python 3.8+

# Verify installation
circom --version
snarkjs --version
python --version
```

### Quick Setup
```bash
# 1. Setup ZK toolchain
cd scripts
./setup_zk.sh

# 2. Complete trusted setup
./complete_trusted_setup.sh

# 3. Start MPC server
fedzk server start --port 9000

# 4. Generate proof
fedzk client prove --input gradients.json --mpc-server http://localhost:9000
```

## ⚠️ Critical Limitations

### ❌ What Won't Work
- **Floating-point gradients**: `snarkJS: RangeError: The number X cannot be converted to a BigInt`
- **Variable input sizes**: Circuits expect exactly 4 gradient values
- **MPC server down**: No automatic fallbacks to local generation
- **Uncompiled circuits**: Must run setup scripts first

### ✅ What Works
- **Integer gradients**: `[1, 2, 3, 4]` format
- **Real MPC servers**: Remote proof generation
- **Secure circuits**: With `maxNorm` and `minNonZero` constraints
- **Batch processing**: Multiple proofs in one operation

## 📋 Input Format Requirements

### Standard Proof
```json
{
  "gradients": {
    "layer.weight": [1, 2, 3, 4],
    "layer.bias": [5, 6, 7, 8]
  },
  "secure": false
}
```

### Secure Proof
```json
{
  "gradients": {
    "layer.weight": [1, 2, 3, 4],
    "layer.bias": [5, 6, 7, 8]
  },
  "secure": true,
  "maxNorm": 1000,
  "minNonZero": 1
}
```

## 🛠️ Common Solutions

### Convert Floating-Point Gradients
```python
# Quantize floating-point to integers
def quantize_gradients(gradients, scale=1000):
    quantized = {}
    for name, values in gradients.items():
        quantized[name] = [int(round(float(v) * scale)) for v in values]
    return quantized

# Usage
original = {"weight": [0.1, 0.2, 0.3, 0.4]}
quantized = quantize_gradients(original)  # {"weight": [100, 200, 300, 400]}
```

### Handle Input Size
```python
# Normalize to exactly 4 values
def normalize_size(values, target=4):
    if len(values) > target:
        return values[:target]  # Truncate
    elif len(values) < target:
        return values + [0] * (target - len(values))  # Pad
    return values

# Usage
original = [1, 2, 3, 4, 5, 6]  # Too many
normalized = normalize_size(original)  # [1, 2, 3, 4]
```

### Validate Inputs
```python
def validate_inputs(gradients):
    errors = []
    for name, values in gradients.items():
        if not isinstance(values, list):
            errors.append(f"{name}: not a list")
        elif len(values) != 4:
            errors.append(f"{name}: wrong size {len(values)}, need 4")
        elif not all(isinstance(v, int) for v in values):
            errors.append(f"{name}: non-integer values")
    return errors

# Usage
errors = validate_inputs(gradients)
if errors:
    print("Fix these errors:", errors)
else:
    print("Inputs are valid!")
```

## 🚨 Error Solutions

### `RangeError: The number X cannot be converted to a BigInt`
**Problem**: Floating-point gradients
**Solution**:
```python
# Quantize before proof generation
quantized = quantize_gradients(floating_gradients, scale=1000)
proof, signals = mpc_client.generate_proof(quantized)
```

### `HTTPConnectionPool: Max retries exceeded`
**Problem**: MPC server unavailable
**Solution**:
```bash
# Check server status
fedzk server health --url http://localhost:9000

# Start server if needed
fedzk server start --port 9000
```

### `Circuit file not found`
**Problem**: Circuits not compiled
**Solution**:
```bash
# Run setup scripts
cd scripts
./setup_zk.sh
./complete_trusted_setup.sh
```

### `Array size mismatch`
**Problem**: Wrong gradient count
**Solution**:
```python
# Normalize each parameter to 4 values
normalized = {}
for name, values in gradients.items():
    normalized[name] = normalize_size(values, 4)
```

## 📊 Circuit Specifications

| Circuit | Input Size | Data Type | Public Inputs | Constraints |
|---------|------------|-----------|---------------|-------------|
| Standard | 4 integers | BigInt | gradients | None |
| Secure | 4 integers | BigInt | gradients, maxNorm, minNonZero | norm ≤ maxNorm, nonZeroCount ≥ minNonZero |
| Quantized | 4 integers | BigInt | quantized_gradients, scale_factor | scale_factor matches circuit parameter |

## 🔧 CLI Commands

### Server Management
```bash
# Start MPC server
fedzk server start --port 9000

# Check server health
fedzk server health --url http://localhost:9000

# Stop server
fedzk server stop
```

### Proof Generation
```bash
# Generate proof with local gradients
fedzk client prove --input gradients.json

# Generate proof with MPC server
fedzk client prove --input gradients.json --mpc-server http://localhost:9000

# Generate secure proof
fedzk client prove --input gradients.json --secure --max-norm 1000
```

### Benchmarking
```bash
# Run benchmark
fedzk benchmark run --clients 5 --secure --mpc-server http://localhost:9000

# Run with custom output
fedzk benchmark run --clients 10 --output results.json --csv results.csv
```

## 🎯 Key Principles

### Real Cryptography Only
- ✅ **Real ZK proofs**: No mocks or simulations
- ✅ **Real MPC integration**: No fallback mechanisms
- ✅ **Real cryptographic validation**: No shortcuts
- ❌ **No automatic conversions**: Manual input preparation required

### Error Handling Philosophy
- **Fail Fast**: Clear errors for invalid inputs
- **No Silent Failures**: Explicit error messages
- **Manual Recovery**: User must fix issues and retry
- **No Automatic Fallbacks**: System integrity preserved

### Input Requirements
- **Integer Only**: No floating-point support in base circuits
- **Fixed Size**: Exactly 4 values per gradient vector
- **Exact Format**: Must match circuit specifications
- **Validation Required**: Check inputs before proof generation

## 📚 Additional Resources

- **[Technical Limitations](./TECHNICAL_LIMITATIONS.md)**: Detailed constraints and known issues
- **[Circuit Requirements](./CIRCUIT_REQUIREMENTS.md)**: Complete input format specifications
- **[Setup Guide](./SETUP.md)**: Installation and configuration instructions
- **[Troubleshooting](./TROUBLESHOOTING.md)**: Solutions for common problems

## ⚡ Quick Checklist

### Pre-Proof Generation
- [ ] ZK toolchain installed (`circom --version`)
- [ ] Circuits compiled (`ls src/fedzk/zk/circuits/`)
- [ ] Trusted setup completed (`ls setup_artifacts/`)
- [ ] MPC server running (`fedzk server health`)
- [ ] Gradients are integers (`[1, 2, 3, 4]` format)
- [ ] Exactly 4 values per parameter
- [ ] Network connectivity to MPC server

### Post-Error Checklist
- [ ] Check gradient data types (integers only)
- [ ] Verify gradient sizes (exactly 4 values)
- [ ] Confirm MPC server accessibility
- [ ] Validate circuit compilation
- [ ] Review quantization scale factors

---

*Remember: FEDzk uses real cryptographic operations with no compromises. All limitations are by design to ensure cryptographic integrity.*

🚀 **Happy proving!**

