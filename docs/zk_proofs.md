# Zero-Knowledge Proofs in FedZK

This document explains how zero-knowledge proofs are used in FedZK to ensure model update integrity while preserving privacy.

## What are Zero-Knowledge Proofs?

Zero-knowledge proofs (ZKPs) are cryptographic protocols that allow one party (the prover) to prove to another party (the verifier) that a statement is true, without revealing any information beyond the validity of the statement itself.

In the context of federated learning, ZKPs allow clients to prove that their model updates satisfy certain properties (such as being derived from legitimate training) without revealing their private training data.

## Types of Zero-Knowledge Proofs Used in FedZK

FedZK implements two main types of zero-knowledge proofs:

1. **zk-SNARKs (Zero-Knowledge Succinct Non-Interactive Arguments of Knowledge)**:
   - Succinct: Proofs are small and quick to verify
   - Non-interactive: No back-and-forth communication is needed after initial setup
   - Implemented using the Groth16 protocol

2. **zk-STARKs (Zero-Knowledge Scalable Transparent Arguments of Knowledge)**:
   - Transparent: No trusted setup required
   - Post-quantum secure
   - Used for larger models where setup cost is prohibitive

## Circuit Design for Federated Learning

FedZK uses arithmetic circuits to define the constraints that model updates must satisfy. The main circuits include:

### Basic Model Update Circuit

This circuit verifies that model updates satisfy basic properties:

```
┌───────────────────────────────────────────────┐
│                                               │
│  Inputs:                                      │
│  - Original model weights                     │
│  - Updated model weights                      │
│  - Training hyperparameters                   │
│                                               │
│  Constraints:                                 │
│  - Update magnitude within bounds             │
│  - Gradient consistency                       │
│  - Format validity                            │
│                                               │
│  Output:                                      │
│  - Validity flag                              │
│                                               │
└───────────────────────────────────────────────┘
```

### Secure Model Update Circuit

This circuit adds additional privacy guarantees:

```
┌───────────────────────────────────────────────┐
│                                               │
│  Inputs:                                      │
│  - Original model weights                     │
│  - Updated model weights                      │
│  - Training hyperparameters                   │
│  - Privacy parameters                         │
│                                               │
│  Constraints:                                 │
│  - Update magnitude within bounds             │
│  - Gradient consistency                       │
│  - Format validity                            │
│  - Differential privacy guarantees            │
│  - No memorization of sensitive data          │
│                                               │
│  Output:                                      │
│  - Validity flag                              │
│                                               │
└───────────────────────────────────────────────┘
```

## Workflow in FedZK

### Proof Generation

When a client completes local training, the following steps occur:

1. The client computes model updates (difference between initial and final model weights)
2. The update is formatted as inputs to the appropriate circuit
3. The client generates a zero-knowledge proof using these inputs
4. Both the update and the proof are sent to the coordinator

```python
def generate_proof(updates, model_params):
    # Convert updates to circuit format
    circuit_inputs = prepare_inputs(updates, model_params)
    
    # Generate the proof
    proof = prove(circuit_inputs, proving_key)
    
    # Return the proof and public inputs
    return {
        "proof": proof,
        "public_inputs": extract_public_inputs(circuit_inputs)
    }
```

### Proof Verification

When the coordinator receives updates and proofs from clients:

1. The coordinator validates the structure of the proof
2. The public inputs are extracted
3. The verification algorithm is run
4. If the proof verifies, the updates are accepted for aggregation

```python
def verify_proof(proof_data, verification_key):
    # Extract proof and public inputs
    proof = proof_data["proof"]
    public_inputs = proof_data["public_inputs"]
    
    # Verify the proof
    is_valid = verify(proof, public_inputs, verification_key)
    
    return is_valid
```

## Benefits in Federated Learning

The use of zero-knowledge proofs in FedZK provides several benefits:

1. **Privacy Preservation**: Clients can prove properties of their updates without revealing training data
2. **Malicious Client Detection**: Ensures that clients cannot submit malformed or malicious updates
3. **Trust Minimization**: Reduces the need to trust individual clients
4. **Regulatory Compliance**: Helps meet privacy regulations by minimizing data sharing

## Performance Considerations

Zero-knowledge proofs introduce computational overhead:

1. **Proof Generation**: More computationally intensive, typically performed on client devices
2. **Verification**: Relatively fast, performed by the coordinator
3. **Optimizations**: FedZK implements several optimizations:
   - Circuit optimization to minimize constraint count
   - Batched proof generation
   - Hardware acceleration (GPU support)
   - Selective proving for critical model components

## Customizing Proof Requirements

FedZK allows developers to customize the properties that need to be proved:

1. Create custom circuits using Circom
2. Define specific constraints based on the application requirements
3. Generate new proving and verification keys
4. Integrate with the FedZK framework

For technical implementation details, please refer to the [Implementation Details](implementation_details.md) document. 