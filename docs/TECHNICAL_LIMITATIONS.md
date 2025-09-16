# FEDzk Technical Limitations

## Overview

FEDzk is a production-ready federated learning framework with real cryptographic operations. This document outlines the current technical limitations and constraints that users should be aware of when deploying and using the system.

## 🚨 Critical Limitations

### 1. ZK Circuit Input Format Requirements

#### **Floating-Point Gradient Limitation**
- **Issue**: Current ZK circuits require **integer inputs only**
- **Problem**: Machine learning gradients are **floating-point values**
- **Impact**: Proof generation will **fail** when using standard ML gradients
- **Error**: `RangeError: The number X cannot be converted to a BigInt because it is not an integer`

#### **Expected Behavior**
```bash
# This WILL fail:
fedzk client prove --input gradients.json
# Error: snarkJS: RangeError: The number 0.00868903286755085 cannot be converted to a BigInt

# This is EXPECTED behavior - no fallbacks or mocks are used
```

#### **Why This Happens**
- ZK circuits (Circom) operate on finite fields
- Floating-point numbers cannot be directly represented in finite fields
- Conversion requires quantization/normalization (not implemented in base circuits)

### 2. No Automatic Fallbacks or Mocks

#### **Design Principle**
FEDzk operates with **real cryptographic operations only**. There are:
- ❌ **No mock proofs** for testing
- ❌ **No fallback mechanisms** to local generation
- ❌ **No simulation modes** for development
- ✅ **Real cryptographic primitives** always

#### **System Behavior**
- **MPC Server Unavailable**: System fails with connection errors
- **ZK Circuit Issues**: System fails with cryptographic errors
- **No Automatic Recovery**: Manual intervention required

### 3. Circuit-Specific Constraints

#### **Standard Circuits**
- **Input Size**: Fixed at 4 gradient values per proof
- **Data Type**: Integers only (BigInt compatible)
- **Range**: Limited by circuit field size
- **Format**: Array of integers `[int, int, int, int]`

#### **Secure Circuits**
- **Additional Parameters**: `maxNorm`, `minNonZero`
- **Input Size**: Same as standard circuits
- **Validation**: Built-in constraint checking

#### **Batch Circuits**
- **Input Format**: Multi-dimensional arrays
- **Size Constraints**: Must match circuit specifications
- **Performance**: Higher computational requirements

## 🔧 Current Circuit Specifications

### Model Update Circuit (`model_update.circom`)

```circom
template ModelUpdate(n) {
    signal input gradients[n];  // Array of integers
    signal output norm;         // Computed norm
}
```

**Requirements:**
- `gradients`: Array of integers
- `n`: Fixed size (currently 4)
- No floating-point values allowed

### Secure Model Update Circuit (`model_update_secure.circom`)

```circom
template ModelUpdateSecure(n, maxNorm, minNonZero) {
    signal input gradients[n];
    signal input maxNorm;           // Maximum allowed norm
    signal input minNonZero;        // Minimum non-zero elements
    signal output norm;
    signal output nonZeroCount;
}
```

**Requirements:**
- All inputs: Integers only
- `maxNorm`: Upper bound for gradient norm
- `minNonZero`: Minimum active gradient elements

## 📋 Input Format Requirements

### Gradient Format
```json
{
  "layer.weight": [1, 2, 3, 4],
  "layer.bias": [5, 6, 7, 8]
}
```

**Constraints:**
- ✅ **Allowed**: Integer arrays
- ❌ **Not Allowed**: Floating-point values
- ❌ **Not Allowed**: Nested objects
- ❌ **Not Allowed**: Strings or other types

### MPC Server Payload
```json
{
  "gradients": {
    "layer.weight": [1, 2, 3, 4],
    "layer.bias": [5, 6, 7, 8]
  },
  "secure": false,
  "batch": false,
  "maxNorm": 1000,
  "minNonZero": 1
}
```

## 🚦 Operational Requirements

### Prerequisites

#### **1. ZK Toolchain**
```bash
# Required tools
- Node.js (v16+)
- npm
- circom (v2.0+)
- snarkjs (v0.7+)

# Verify installation
circom --version
snarkjs --version
```

#### **2. Circuit Compilation**
```bash
# Circuits must be compiled and trusted setup completed
cd scripts
./setup_zk.sh
./complete_trusted_setup.sh
```

#### **3. MPC Server**
```bash
# Server must be running and accessible
fedzk server start --port 9000
# Or external MPC server at known URL
```

#### **4. Network Connectivity**
- Stable network connection to MPC server
- No proxy/firewall blocking cryptographic operations
- Reliable DNS resolution

### Environment Setup

#### **Required Environment Variables**
```bash
# Optional but recommended
export FEDZK_COORDINATOR_URL="http://localhost:8000"
export FEDZK_MPC_SERVER_URL="http://localhost:9000"
```

#### **File System Requirements**
```
fedzk/
├── src/fedzk/zk/circuits/     # Compiled circuits
├── setup_artifacts/           # Trusted setup files
├── temp_test/                 # Working directory
└── logs/                      # Operation logs
```

## ⚠️ Common Failure Scenarios

### 1. Floating-Point Gradients
```
Error: snarkJS: RangeError: The number 0.123456 cannot be converted to a BigInt
Solution: Convert gradients to integers before proof generation
```

### 2. MPC Server Unavailable
```
Error: HTTPConnectionPool: Max retries exceeded
Solution: Ensure MPC server is running and accessible
```

### 3. Circuit Not Compiled
```
Error: Circuit file not found
Solution: Run setup_zk.sh and complete_trusted_setup.sh
```

### 4. Network Issues
```
Error: Failed to resolve hostname
Solution: Check network connectivity and DNS
```

## 🔄 Future Enhancements

### Planned Improvements

#### **1. Quantized Circuits (Medium Priority)**
- Implementation of `model_update_quantized.circom`
- Support for scaled floating-point inputs
- Automatic quantization with integrity preservation

#### **2. Enhanced Error Handling (Medium Priority)**
- Better error messages and troubleshooting guidance
- Automatic detection of common issues
- Improved logging and diagnostics

#### **3. Extended Circuit Support (Low Priority)**
- Additional circuit templates for different use cases
- Batch processing optimizations
- Performance improvements

## 📚 Best Practices

### Development Guidelines

#### **1. Input Validation**
```python
# Always validate gradient format before proof generation
def validate_gradients(gradients):
    for name, values in gradients.items():
        if not isinstance(values, list):
            raise ValueError(f"Gradient {name} must be a list")
        if not all(isinstance(x, int) for x in values):
            raise ValueError(f"Gradient {name} must contain integers only")
```

#### **2. Error Handling**
```python
try:
    proof, signals = mpc_client.generate_proof(gradients)
except ConnectionError:
    print("MPC server unavailable - check server status")
except RuntimeError as e:
    if "BigInt" in str(e):
        print("Floating-point gradients detected - convert to integers")
    else:
        print(f"Cryptographic error: {e}")
```

#### **3. System Monitoring**
```python
# Regularly check system health
health = mpc_client.get_server_health()
if not health["server_reachable"]:
    print("MPC server health check failed")
```

### Production Deployment

#### **1. Pre-deployment Checklist**
- [ ] ZK toolchain installed and verified
- [ ] Circuits compiled and trusted setup completed
- [ ] MPC server running and accessible
- [ ] Network connectivity verified
- [ ] Test proofs generated successfully

#### **2. Monitoring**
- Monitor MPC server health endpoints
- Track proof generation success rates
- Log all cryptographic failures
- Set up alerts for system unavailability

#### **3. Backup Strategies**
- Multiple MPC server instances
- Redundant network connections
- Regular backup of trusted setup artifacts

## 🎯 Summary

FEDzk provides **real cryptographic operations** with **no compromises** on security or functionality. The current limitations are well-defined and expected behavior:

- **✅ Real ZK Proofs**: No mocks or simulations
- **✅ Real MPC Integration**: No fallback mechanisms
- **✅ Production Ready**: Enterprise-grade cryptographic operations
- **⚠️ Integer Inputs Required**: Current circuits don't support floating-point
- **⚠️ All Services Required**: No automatic fallbacks when components unavailable

These limitations ensure **cryptographic integrity** while providing clear guidance for proper system usage and future enhancements.

---

*Last Updated: December 2024*
*Document Version: 1.0*

