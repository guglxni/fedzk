# Zero-Knowledge Proofs in FedZK

This document explains the zero-knowledge proof system used in FedZK, including the circuit designs, security properties, and integration with the federated learning system.

## Introduction to Zero-Knowledge Proofs

Zero-knowledge proofs (ZKPs) are cryptographic protocols that allow one party (the prover) to prove to another party (the verifier) that a statement is true without revealing any information beyond the validity of the statement itself.

In FedZK, ZKPs are used to:

1. Prove that client-generated gradients satisfy certain properties (e.g., bounded L2 norm)
2. Verify the integrity of model updates without revealing the underlying training data
3. Prevent malicious clients from poisoning the global model

## Circuit Implementation

FedZK uses [circom](https://docs.circom.io/) to implement its ZK circuits and [snarkjs](https://github.com/iden3/snarkjs) to generate and verify proofs.

### Standard Circuit

The standard circuit (`model_update.circom`) computes the L2 norm of gradient values:

```circom
pragma circom 2.0.0;

template ModelUpdate(n) {
    signal input gradients[n];
    signal output norm;
    signal acc[n+1];
    signal sq[n];

    // Initialize accumulator
    acc[0] <== 0;

    // Compute squares and accumulate step by step
    for (var i = 0; i < n; i++) {
        sq[i] <== gradients[i] * gradients[i];
        acc[i+1] <== acc[i] + sq[i];
    }

    // Output the final accumulated sum
    norm <== acc[n];
}

component main { public [gradients] } = ModelUpdate(4);
```

### Secure Circuit

The secure circuit (`model_update_secure.circom`) extends the standard circuit with additional constraints:

```circom
pragma circom 2.0.0;

template ModelUpdateSecure(n) {
    signal input gradients[n];
    signal output norm;
    signal output nonZeroCount;
    signal acc[n+1];
    signal sq[n];
    signal count[n+1];
    signal isNZ[n];

    // Initialize accumulators
    acc[0] <== 0;
    count[0] <== 0;

    // Compute squares and nonzero count stepwise
    for (var i = 0; i < n; i++) {
        sq[i] <== gradients[i] * gradients[i];
        acc[i+1] <== acc[i] + sq[i];

        isNZ[i] <== gradients[i] != 0;
        count[i+1] <== count[i] + isNZ[i];
    }

    // Assign outputs and enforce bounds
    norm <== acc[n];
    nonZeroCount <== count[n];
    assert(norm <= 100);       // L2 norm is bounded
    assert(nonZeroCount >= 3); // Minimum number of non-zero elements
}

component main { public [gradients] } = ModelUpdateSecure(4);
```

## Security Properties

The secure circuit enforces two key constraints:

1. **Bounded L2 Norm**: Ensures the magnitude of gradients is limited, preventing extreme values that could skew the aggregated model.

2. **Minimum Activity**: Requires at least a certain number of non-zero elements, preventing sparse attacks where only a few parameters are modified.

These constraints help detect and prevent several types of attacks:

- **Model Poisoning**: Malicious updates designed to corrupt the global model
- **Free-riding**: Clients submitting zero or minimal gradients to "free-ride" on the system
- **Gradient Inflation**: Artificially large gradients that would dominate the aggregation

## Batch Processing

For handling large models with many parameters, FedZK implements a batching mechanism that splits gradients into manageable chunks:

1. Gradients are divided into chunks of a fixed size
2. A ZK proof is generated for each chunk
3. A Merkle tree is constructed with the proofs as leaves
4. The Merkle root is used to verify the integrity of the entire batch

### Batch Circuit Integration

The batch processing method doesn't require changes to the circuit itself, but rather involves:

1. Processing gradients in fixed-size chunks
2. Generating individual proofs for each chunk
3. Creating a Merkle tree from the chunk proofs
4. Providing a verification path for each chunk

## Proof Generation and Verification

### Generation Process

1. Client extracts gradients from their local model
2. Gradients are flattened into a one-dimensional array
3. For standard mode:
   - Proof is generated directly using the `model_update.wasm` circuit
4. For secure mode:
   - Client ensures gradients meet L2 norm and minimum activity constraints
   - Proof is generated using the `model_update_secure.wasm` circuit
5. For batch mode:
   - Gradients are chunked into fixed-size pieces
   - Individual proofs are generated for each chunk
   - A Merkle tree is constructed from the chunk proofs
   - The Merkle root and paths are included in the final proof

### Verification Process

1. Coordinator receives a proof and its public inputs
2. For standard and secure proofs:
   - Verification is performed directly using snarkjs
3. For batch proofs:
   - The Merkle root is verified
   - Individual chunk proofs are verified
   - Results are aggregated to determine overall validity

## MPC Server Integration

For clients with limited computational resources, FedZK provides an MPC (Multi-Party Computation) server that can offload proof generation and verification:

1. Client sends gradient data to the MPC server
2. MPC server generates the proof
3. MPC server returns the proof to the client
4. Client submits the proof to the coordinator
5. Coordinator can verify the proof locally or use the MPC server

This design maintains privacy while reducing the computational burden on clients.

## Security Considerations

### Trusted Setup

The proving and verification keys are generated during a trusted setup phase:

```bash
# Setup for standard circuit
snarkjs groth16 setup $BUILD_DIR/$CIRCUIT_NAME.r1cs $PTAU_PATH $BUILD_DIR/circuit_0000.zkey
echo "FedZK" | snarkjs zkey contribute $BUILD_DIR/circuit_0000.zkey $BUILD_DIR/proving_key.zkey
snarkjs zkey export verificationkey $BUILD_DIR/proving_key.zkey $BUILD_DIR/verification_key.json

# Setup for secure circuit
snarkjs groth16 setup $BUILD_DIR/$SECURE_CIRCUIT_NAME.r1cs $PTAU_PATH $BUILD_DIR/secure_circuit_0000.zkey
echo "FedZK-Secure" | snarkjs zkey contribute $BUILD_DIR/secure_circuit_0000.zkey $BUILD_DIR/proving_key_secure.zkey
snarkjs zkey export verificationkey $BUILD_DIR/proving_key_secure.zkey $BUILD_DIR/verification_key_secure.json
```

### API Key Authentication

The MPC server requires API key authentication to prevent unauthorized access:

```python
# Checking API key in server
api_keys = os.environ.get("MPC_API_KEYS", "").split(",")
api_key = request.headers.get("x-api-key")
if not api_key or api_key not in api_keys:
    return JSONResponse(
        status_code=401,
        content={"error": "Unauthorized: Invalid or missing API key"}
    )
```

## Performance Considerations

Zero-knowledge proof generation is computationally intensive. FedZK implements several optimizations:

1. **Batch Processing**: Handles large models by processing gradients in chunks
2. **Remote Proof Generation**: Offloads computation to an MPC server
3. **Fallback Mechanisms**: Gracefully handles MPC server unavailability

## Integration with Federated Learning

FedZK integrates ZK proofs into the federated learning workflow:

1. **Client Training**: Client trains a local model on private data
2. **Proof Generation**: Client generates a ZK proof for the model gradients
3. **Submission**: Client submits gradients and proof to the coordinator
4. **Verification**: Coordinator verifies the proof
5. **Aggregation**: If valid, coordinator incorporates the gradients into the global model
6. **Distribution**: Updated global model is distributed to clients

This integration ensures that only valid gradients are incorporated into the global model, enhancing the security and trustworthiness of the federated learning system.

## Future Enhancements

Potential enhancements to the ZK proof system include:

1. **Advanced Constraints**: Additional properties to verify, such as differential privacy guarantees
2. **Hardware Acceleration**: GPU-based proof generation for improved performance
3. **Recursive SNARKs**: Efficient verification of multiple proofs
4. **zk-SNARKs Alternatives**: Exploring other ZKP systems like zk-STARKs or Bulletproofs 