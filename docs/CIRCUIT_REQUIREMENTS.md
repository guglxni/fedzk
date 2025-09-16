# FEDzk Circuit Requirements

## Overview

This document specifies the exact input format requirements for FEDzk's ZK circuits. All circuits operate on **integer inputs only** and require specific data formats for successful proof generation.

## 🔧 Circuit Specifications

### 1. Model Update Circuit

#### **File**: `model_update.circom`
#### **Purpose**: Basic gradient norm computation and verification

```circom
pragma circom 2.0.0;

template ModelUpdate(n) {
    signal input gradients[n];
    signal output norm;
    signal acc[n+1];
    signal sq[n];

    // Initialize accumulator
    acc[0] <== 0;

    // Compute squares and accumulate
    for (var i = 0; i < n; i++) {
        sq[i] <== gradients[i] * gradients[i];
        acc[i+1] <== acc[i] + sq[i];
    }

    // Output final norm
    norm <== acc[n];
}

component main { public [gradients] } = ModelUpdate(4);
```

#### **Input Requirements**
- **Parameter `n`**: Fixed at 4 (gradient vector size)
- **Input `gradients`**: Array of 4 integers
- **Data Type**: `BigInt` compatible integers only
- **Range**: Limited by circuit field size

#### **Input Format**
```json
{
  "gradients": [1, 2, 3, 4]
}
```

#### **Output**
```json
{
  "norm": 30
}
```

### 2. Secure Model Update Circuit

#### **File**: `model_update_secure.circom`
#### **Purpose**: Gradient verification with security constraints

```circom
pragma circom 2.0.0;

template IsZero() {
    signal input in;
    signal output out;
    signal inv;

    inv <-- in != 0 ? 1/in : 0;
    out <== -in * inv + 1;
    in * out === 0;
}

template ModelUpdateSecure(n) {
    signal input gradients[n];
    signal input maxNorm;
    signal input minNonZero;
    signal output norm;
    signal output nonZeroCount;
    signal acc[n+1];
    signal sq[n];
    signal count[n+1];
    signal isNZ[n];
    component iz[n];

    acc[0] <== 0;
    count[0] <== 0;

    for (var i = 0; i < n; i++) {
        sq[i] <== gradients[i] * gradients[i];
        acc[i+1] <== acc[i] + sq[i];

        iz[i] = IsZero();
        iz[i].in <== gradients[i];
        isNZ[i] <== 1 - iz[i].out;
        count[i+1] <== count[i] + isNZ[i];
    }

    norm <== acc[n];
    nonZeroCount <== count[n];

    // Security constraints
    assert(norm <= maxNorm);
    assert(nonZeroCount >= minNonZero);
}

component main { public [gradients, maxNorm, minNonZero] } = ModelUpdateSecure(4);
```

#### **Input Requirements**
- **Parameter `n`**: Fixed at 4
- **Input `gradients`**: Array of 4 integers
- **Input `maxNorm`**: Maximum allowed norm (integer)
- **Input `minNonZero`**: Minimum non-zero elements (integer)
- **Data Type**: All inputs must be BigInt compatible

#### **Input Format**
```json
{
  "gradients": [1, 2, 3, 4],
  "maxNorm": 1000,
  "minNonZero": 1
}
```

#### **Output**
```json
{
  "norm": 30,
  "nonZeroCount": 4
}
```

### 3. Quantized Model Update Circuit (Future Enhancement)

#### **File**: `model_update_quantized.circom`
#### **Purpose**: Support for quantized floating-point gradients

```circom
pragma circom 2.0.0;

template ModelUpdateQuantized(n, scale_factor) {
    signal input quantized_gradients[n];
    signal input scale_factor_input;
    signal output original_norm;
    signal output quantized_norm;
    signal output gradient_count;

    signal scaled_gradients[n];
    signal sq[n];
    signal acc[n+1];

    scale_factor_input === scale_factor;
    acc[0] <== 0;

    for (var i = 0; i < n; i++) {
        scaled_gradients[i] <== quantized_gradients[i] * scale_factor_input;
        sq[i] <== scaled_gradients[i] * scaled_gradients[i];
        acc[i+1] <== acc[i] + sq[i];
    }

    original_norm <== acc[n];
    quantized_norm <== acc[n] / (scale_factor * scale_factor);
    gradient_count <== n;
    gradient_count === n;
}

component main { public [quantized_gradients, scale_factor_input] } = ModelUpdateQuantized(4, 1000);
```

#### **Input Requirements**
- **Parameter `n`**: Fixed at 4
- **Parameter `scale_factor`**: Quantization scale (fixed at 1000)
- **Input `quantized_gradients`**: Array of 4 quantized integers
- **Input `scale_factor_input`**: Must match circuit scale_factor
- **Data Type**: BigInt compatible integers

#### **Input Format**
```json
{
  "quantized_gradients": [1234, 5678, 9012, 3456],
  "scale_factor_input": 1000
}
```

## 📋 Complete Input Specifications

### FEDzk Proof Generation Payload

#### **Standard Proof Request**
```json
{
  "gradients": {
    "layer.weight": [1, 2, 3, 4],
    "layer.bias": [5, 6, 7, 8]
  },
  "secure": false,
  "batch": false
}
```

#### **Secure Proof Request**
```json
{
  "gradients": {
    "layer.weight": [1, 2, 3, 4],
    "layer.bias": [5, 6, 7, 8]
  },
  "secure": true,
  "batch": false,
  "maxNorm": 1000,
  "minNonZero": 1
}
```

#### **Batch Proof Request**
```json
{
  "gradients": {
    "layer.weight": [1, 2, 3, 4],
    "layer.bias": [5, 6, 7, 8]
  },
  "secure": false,
  "batch": true,
  "chunk_size": 2
}
```

### Data Type Constraints

#### **Allowed Data Types**
- ✅ **Integers**: `1`, `2`, `3`, `-5`, `1000`
- ✅ **BigInt Compatible**: Numbers within JavaScript BigInt range
- ✅ **Arrays**: `[1, 2, 3, 4]` (fixed size arrays only)

#### **Forbidden Data Types**
- ❌ **Floating-point**: `1.23`, `0.456`, `1e-5`
- ❌ **Strings**: `"1.23"`, `"gradient"`
- ❌ **Objects**: `{"value": 1.23}`
- ❌ **Null/Undefined**: `null`, `undefined`
- ❌ **Dynamic Arrays**: Variable-length arrays

### Size and Dimension Requirements

#### **Gradient Vector Size**
- **Fixed Size**: All circuits expect exactly 4 gradient values
- **No Padding**: Input must match exact circuit expectations
- **No Truncation**: Cannot provide more or less than 4 values

#### **Multi-Layer Support**
```json
{
  "conv1.weight": [1, 2, 3, 4],
  "conv1.bias": [5, 6, 7, 8],
  "fc.weight": [9, 10, 11, 12],
  "fc.bias": [13, 14, 15, 16]
}
```
- Each layer parameter must be exactly 4 integers
- Layer names are arbitrary strings
- All layers processed independently

## 🔧 Input Validation Rules

### Pre-Processing Requirements

#### **1. Tensor to Array Conversion**
```python
# PyTorch tensor conversion
gradients = model.gradients
processed = {}
for name, tensor in gradients.items():
    # Flatten and convert to list
    flat_values = tensor.flatten().tolist()
    # Take first 4 values (or pad/truncate as needed)
    processed[name] = flat_values[:4]
```

#### **2. Integer Conversion**
```python
# Convert floating-point to integers
def quantize_gradients(gradients, scale_factor=1000):
    quantized = {}
    for name, values in gradients.items():
        quantized_values = []
        for val in values:
            # Scale and round to integer
            quantized_val = round(float(val) * scale_factor)
            quantized_values.append(int(quantized_val))
        quantized[name] = quantized_values
    return quantized
```

#### **3. Size Normalization**
```python
def normalize_gradient_size(gradients, target_size=4):
    normalized = {}
    for name, values in gradients.items():
        if len(values) > target_size:
            # Truncate
            normalized[name] = values[:target_size]
        elif len(values) < target_size:
            # Pad with zeros
            padded = values + [0] * (target_size - len(values))
            normalized[name] = padded
        else:
            normalized[name] = values
    return normalized
```

### Validation Checks

#### **Circuit Compatibility Validation**
```python
def validate_circuit_inputs(gradients, circuit_type="standard"):
    """
    Validate that gradients meet circuit requirements.
    """
    errors = []

    for layer_name, values in gradients.items():
        # Check data type
        if not isinstance(values, list):
            errors.append(f"{layer_name}: must be a list")

        # Check element types
        for i, val in enumerate(values):
            if not isinstance(val, int):
                errors.append(f"{layer_name}[{i}]: must be integer, got {type(val)}")

        # Check size
        if len(values) != 4:
            errors.append(f"{layer_name}: must have exactly 4 values, got {len(values)}")

    return errors
```

## 🚦 Proof Generation Workflow

### 1. Input Preparation
```python
# Step 1: Extract gradients from model
gradients = extract_model_gradients(model)

# Step 2: Validate and normalize
errors = validate_circuit_inputs(gradients)
if errors:
    raise ValueError(f"Input validation failed: {errors}")

# Step 3: Convert to circuit format
circuit_input = prepare_circuit_input(gradients)
```

### 2. Proof Generation
```python
# Step 4: Generate proof
from fedzk.mpc.client import MPCClient

mpc_client = MPCClient(
    server_url="http://localhost:9000",
    api_key="your_api_key"
)

proof, signals = mpc_client.generate_proof(circuit_input)
```

### 3. Verification
```python
# Step 5: Verify proof
from fedzk.prover.verifier import ZKVerifier

verifier = ZKVerifier()
is_valid = verifier.verify_proof(proof, signals)
```

## ⚠️ Common Input Errors

### Error: "RangeError: The number X cannot be converted to a BigInt"
**Cause**: Floating-point values in input
**Solution**: Convert to integers using quantization

### Error: "Array size mismatch"
**Cause**: Wrong number of gradient values
**Solution**: Normalize to exactly 4 values per parameter

### Error: "Invalid data type"
**Cause**: Non-integer values in input
**Solution**: Ensure all values are integers

### Error: "Circuit input validation failed"
**Cause**: Input doesn't match circuit expectations
**Solution**: Check input format against circuit specifications

## 🔄 Future Circuit Enhancements

### Planned Circuit Improvements

#### **1. Flexible Input Sizes**
```circom
// Future: Support variable input sizes
template ModelUpdateFlexible(n) {
    signal input gradients[n];
    signal input input_size;
    signal output norm;

    // Dynamic size handling
    // Implementation for variable n
}
```

#### **2. Floating-Point Support**
```circom
// Future: Native floating-point support
template ModelUpdateFloat(n, precision) {
    signal input float_gradients[n];
    signal input precision_bits;
    signal output norm;

    // Fixed-point arithmetic implementation
    // Precision-based calculations
}
```

#### **3. Advanced Batch Processing**
```circom
// Future: Enhanced batch processing
template BatchModelUpdate(batch_size, grad_size) {
    signal input gradient_batch[batch_size][grad_size];
    signal output batch_norms[batch_size];

    // Parallel batch processing
    // Optimized for multiple clients
}
```

## 📚 Best Practices

### Input Preparation Guidelines

#### **1. Consistent Quantization**
```python
# Use consistent scale factors across all operations
SCALE_FACTOR = 1000

def prepare_inputs(gradients):
    return quantize_gradients(gradients, SCALE_FACTOR)
```

#### **2. Input Validation**
```python
# Always validate before proof generation
def safe_proof_generation(gradients):
    # Validate
    errors = validate_circuit_inputs(gradients)
    if errors:
        raise ValueError(f"Invalid inputs: {errors}")

    # Generate proof
    return mpc_client.generate_proof(gradients)
```

#### **3. Error Handling**
```python
# Handle common errors gracefully
try:
    proof, signals = mpc_client.generate_proof(gradients)
except ValueError as e:
    if "BigInt" in str(e):
        print("Convert gradients to integers first")
    elif "size mismatch" in str(e):
        print("Normalize gradient sizes to 4 values")
    else:
        print(f"Circuit error: {e}")
```

### Performance Optimization

#### **1. Batch Processing**
```python
# Use batch processing for multiple proofs
batch_payload = {
    "gradients": [grad1, grad2, grad3],
    "batch": True,
    "chunk_size": 2
}
```

#### **2. Pre-validation**
```python
# Validate inputs before expensive operations
if validate_circuit_inputs(gradients):
    # Only proceed if inputs are valid
    proof, signals = generate_proof(gradients)
```

## 🎯 Summary

### Circuit Requirements Summary

| Circuit Type | Input Size | Data Types | Additional Inputs | Output |
|-------------|------------|------------|-------------------|---------|
| Standard | 4 integers | BigInt only | None | norm |
| Secure | 4 integers | BigInt only | maxNorm, minNonZero | norm, nonZeroCount |
| Quantized | 4 integers | BigInt only | scale_factor | original_norm, quantized_norm |

### Key Takeaways

1. **Integer Only**: All circuit inputs must be integers
2. **Fixed Size**: Exactly 4 gradient values per proof
3. **No Float Support**: Current circuits don't handle floating-point
4. **Validation Required**: Always validate inputs before proof generation
5. **Format Strict**: Must match exact circuit specifications

### Development Recommendations

- **Use Quantization**: Convert floating-point gradients to integers
- **Validate Inputs**: Always check format before proof generation
- **Handle Errors**: Implement proper error handling for common issues
- **Test Thoroughly**: Validate with real circuits before production use

---

*Last Updated: December 2024*
*Document Version: 1.0*

