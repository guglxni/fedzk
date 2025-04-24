# Zero-Knowledge Circuit Implementation in FedZK

This document explains the technical details of the zero-knowledge proof circuits used in FedZK.

## Overview

FedZK uses zero-knowledge proofs to verify that client model updates satisfy certain properties without revealing the actual gradients. This is implemented using circuits that convert numerical computations into constraint systems that can be proven and verified.

## Circuit Types

FedZK implements two main types of circuits:

### 1. Basic Circuit

The basic circuit proves the correctness of gradient values without applying additional constraints. It proves that:

- The public commitment matches the private gradient values
- The values are represented correctly in the circuit

```
Input: Private gradient values [g₁, g₂, ..., gₙ]
Output: Public commitment to these values
```

### 2. Secure Circuit

The secure circuit extends the basic circuit with additional constraints to prevent poisoning attacks. It provides the following guarantees:

- The L2 norm squared of the gradients is below a threshold
- There is a minimum number of non-zero gradient elements
- The values are within an acceptable range

```
Input: Private gradient values [g₁, g₂, ..., gₙ]
Output: Public commitment + verification of constraints
```

## Circuit Implementation

The circuits are implemented using the circom language and compiled using snarkjs. Below is an outline of the main circuit components:

### Gradient Commitment

```circom
template GradientCommitment(n) {
    signal input gradients[n];
    signal output commitment;
    
    // Hash or other commitment mechanism
    // ...
}
```

### L2 Norm Constraint

```circom
template L2NormConstraint(n) {
    signal input gradients[n];
    signal input max_norm;
    signal output norm_squared;
    
    var sum = 0;
    for (var i = 0; i < n; i++) {
        sum += gradients[i] * gradients[i];
    }
    
    norm_squared <== sum;
    norm_squared <= max_norm;
}
```

### Active Elements Constraint

```circom
template ActiveElementsConstraint(n) {
    signal input gradients[n];
    signal input min_active;
    signal output active_count;
    
    var count = 0;
    for (var i = 0; i < n; i++) {
        // Non-zero check
        count += (gradients[i] != 0) ? 1 : 0;
    }
    
    active_count <== count;
    active_count >= min_active;
}
```

## Batching Implementation

For large models, FedZK implements a batching mechanism that breaks large gradient vectors into smaller chunks:

```circom
template BatchedGradientProof(chunk_size, num_chunks) {
    signal input gradients[chunk_size * num_chunks];
    
    component chunks[num_chunks];
    for (var i = 0; i < num_chunks; i++) {
        chunks[i] = GradientProof(chunk_size);
        for (var j = 0; j < chunk_size; j++) {
            chunks[i].gradients[j] <== gradients[i * chunk_size + j];
        }
    }
}
```

## Circuit Compilation

The circuits are compiled in a multi-step process:

1. The circom source code is compiled to an arithmetic circuit
2. A trusted setup is performed (using Powers of Tau)
3. The proving and verification keys are generated

This process creates the necessary artifacts for both proving and verifying ZK proofs.

## Integration with snarkjs

FedZK integrates with the snarkjs library to handle the proof generation and verification:

```javascript
// Generating a proof
const { proof, publicSignals } = await snarkjs.groth16.fullProve(
    { gradients: [g1, g2, ..., gn], max_norm: 100.0, min_active: 10 },
    wasm_file,
    zkey_file
);

// Verifying a proof
const isValid = await snarkjs.groth16.verify(
    verification_key,
    publicSignals,
    proof
);
```

## Circuit Constraints and Security

The secure circuit enforces the following constraints to prevent poisoning attacks:

1. **L2 Norm Constraint**: Prevents clients from submitting gradients with excessively large magnitudes that could disproportionately influence the aggregated model.
   
   ```
   ∑(gᵢ²) ≤ max_norm
   ```

2. **Minimum Active Elements**: Ensures that updates affect a minimum number of model parameters, preventing targeted attacks on specific neurons.
   
   ```
   |{i | gᵢ ≠ 0}| ≥ min_active
   ```

3. **Range Constraint**: Ensures each individual gradient component is within an acceptable range.
   
   ```
   ∀i: -max_value ≤ gᵢ ≤ max_value
   ```

## Performance Considerations

The circuit implementation balances security and performance:

- **Circuit Size**: The number of constraints grows linearly with the number of gradient elements
- **Batch Processing**: Large models are processed in chunks to manage circuit complexity
- **Constraint Selection**: Users can configure which constraints to enable based on their security requirements

## Future Improvements

Potential improvements to the circuit implementation:

1. More sophisticated constraints (cosine similarity, statistical tests)
2. Support for quantized gradients to reduce circuit complexity
3. Recursive SNARKs for more efficient batch processing
4. Integration with other proof systems (Plonk, Bulletproofs, etc.)

## Technical Limitations

1. Circuit complexity increases with the number of gradient elements
2. Floating-point operations are expensive in circuit representations
3. Large models require significant computational resources for proof generation

## Appendix: Circom Circuit Example

A simplified example of the complete secure circuit:

```circom
pragma circom 2.0.0;

template GradientProof(n) {
    signal input gradients[n];
    signal input max_norm;
    signal input min_active;
    
    signal output commitment;
    signal output norm_squared;
    signal output active_count;
    
    // Compute commitment
    var comm = 0;
    for (var i = 0; i < n; i++) {
        comm += gradients[i] * (i + 1);  // Simple commitment scheme
    }
    commitment <== comm;
    
    // Compute norm
    var norm = 0;
    for (var i = 0; i < n; i++) {
        norm += gradients[i] * gradients[i];
    }
    norm_squared <== norm;
    
    // Count active elements
    var active = 0;
    for (var i = 0; i < n; i++) {
        active += (gradients[i] != 0) ? 1 : 0;
    }
    active_count <== active;
    
    // Enforce constraints
    assert(norm_squared <= max_norm);
    assert(active_count >= min_active);
}

component main {public [max_norm, min_active]} = GradientProof(10);
``` 