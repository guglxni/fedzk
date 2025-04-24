# FedZK Implementation Details

This document provides technical details about the implementation of the FedZK framework for secure federated learning with zero-knowledge proofs.

## Architecture Overview

FedZK is designed to enable secure federated learning with zero-knowledge proofs for gradient verification. The system consists of the following key components:

1. **Client Module**: Handles local model training and gradient extraction
2. **ZK Module**: Manages the generation and verification of zero-knowledge proofs
3. **Coordinator Module**: Orchestrates the federated learning process
4. **MPC Module**: Provides remote proof generation and verification services
5. **CLI**: Command-line interface for interacting with the system
6. **Benchmark Module**: Tools for evaluating system performance

The components interact in the following way:

```
                                 ┌─────────────────┐
                                 │  Coordinator    │
                                 │    Server       │
                                 └───────┬─────────┘
                                         │
                                         │ HTTP API
                                         │
        ┌───────────────────┬────────────┴───────────┬────────────────────┐
        │                   │                        │                    │
┌───────▼───────┐   ┌───────▼───────┐        ┌───────▼───────┐    ┌───────▼───────┐
│               │   │               │        │               │    │               │
│  Client 1     │   │  Client 2     │        │  Client 3     │    │  Client N     │
│               │   │               │        │               │    │               │
└───────┬───────┘   └───────┬───────┘        └───────┬───────┘    └───────┬───────┘
        │                   │                        │                    │
        │                   │                        │                    │
        │                   │        HTTP API        │                    │
        │                   │                        │                    │
        └───────────────────┼────────────────────────┼────────────────────┘
                            │                        │
                     ┌──────▼────────┐      ┌────────▼──────┐
                     │               │      │               │
                     │  MPC Server   │◄─────┤  MPC Server   │
                     │     1         │      │     2         │
                     │               │      │               │
                     └───────────────┘      └───────────────┘
```

## Zero-Knowledge Proof System

### Circuit Implementation

FedZK uses [circom](https://github.com/iden3/circom) for creating zero-knowledge proof circuits and [snarkjs](https://github.com/iden3/snarkjs) for generating and verifying proofs. The framework implements several circuits:

1. **Basic Gradient Circuit**: A simple circuit that computes the sum of gradient elements
2. **Secure Gradient Circuit**: An advanced circuit that enforces constraints on gradients:
   - Maximum L2 norm squared (prevents gradient explosion attacks)
   - Minimum number of active elements (prevents sparse gradient attacks)

The circuits are defined in the `circuits/` directory:

- `gradient.circom`: Basic gradient circuit
- `secure_gradient.circom`: Secure gradient circuit with constraints
- `batch_gradient.circom`: Batch circuit for processing large gradients in chunks

### Proof Generation Process

The proof generation process follows these steps:

1. Extract gradients from a trained model
2. Flatten the gradients into a single array
3. Chunk the gradients if batch processing is enabled
4. For each chunk:
   - Prepare input signals for the circuit
   - Generate a witness
   - Generate a proof using the proving key
5. In batch mode, combine proofs from all chunks

### Verification Process

The verification process involves:

1. Receiving a proof and public inputs
2. Verifying the proof using the verification key
3. For batch proofs, verifying each chunk's proof
4. Checking that all constraints are satisfied

## Batch Processing

### Chunking Mechanism

For large models with many parameters, FedZK implements a chunking mechanism to break down gradient processing into manageable pieces:

1. Gradients are flattened into a single array
2. The array is divided into chunks of configurable size
3. Each chunk is processed separately through the circuit
4. Results from all chunks are combined for final verification

This approach allows processing models of any size, as the circuit size remains constant regardless of the model size.

### Chunk Constraints

For secure batch processing, FedZK enforces constraints on each chunk:

- Maximum L2 norm squared per chunk
- Minimum active elements across all chunks

The final verification ensures that the combined constraints meet the overall requirements.

## MPC Server

### Remote Proof Generation

The MPC (Multi-Party Computation) server provides remote proof generation and verification services for clients with limited computational resources. Key features include:

1. **API Key Authentication**: Secure access to proof generation and verification endpoints
2. **Batch Processing Support**: Processing multiple gradient sets efficiently
3. **Resource Management**: Queuing and prioritization of proof generation requests

### Authentication System

The MPC server implements an API key authentication system:

1. Keys are loaded from the `MPC_API_KEYS` environment variable (comma-separated list)
2. Each request must include the `x-api-key` header with a valid key
3. Invalid or missing keys result in a 401 Unauthorized response

## Coordinator Design

### Aggregation Strategies

The coordinator supports multiple aggregation strategies for combining client updates:

1. **FedAvg**: Standard federated averaging algorithm
2. **FedProx**: Federated optimization with proximal terms
3. **Custom**: Extensible interface for custom aggregation algorithms

### Proof Verification Flow

When a client submits a proof, the coordinator follows this process:

1. Verify the proof's authenticity using the ZK verification system
2. Extract the public inputs (gradients sum, L2 norm, etc.)
3. Add the verified update to the aggregation queue
4. After collecting sufficient updates, trigger the aggregation process
5. Update the global model and make it available to clients

## Client Implementation

### Training Process

The client training process is implemented to:

1. Load local data and the global model
2. Train the model for a configurable number of epochs
3. Extract gradients between the updated model and original model
4. Generate a zero-knowledge proof for the gradients
5. Submit the proof to the coordinator

### Gradient Extraction

Gradient extraction is performed by:

1. Capturing parameter values before training
2. Training the model on local data
3. Computing the difference between pre-training and post-training parameters
4. Converting these differences to the format required by the ZK circuit

## Performance Optimizations

### Circuit Optimizations

Several optimizations have been implemented to improve circuit performance:

1. **Signal Packing**: Combining multiple signals to reduce circuit constraints
2. **Custom Templates**: Specialized circuit templates for common operations
3. **Parallel Computation**: Parallel circuits for independent computations

### Proof Generation Optimizations

Proof generation has been optimized through:

1. **Batching**: Processing multiple gradients in a single operation
2. **GPU Acceleration**: Optional GPU support for faster witness generation
3. **Caching**: Reusing intermediate results when possible

## Benchmark System

### End-to-End Benchmarks

The benchmark system measures:

1. **Training Time**: Local model training duration
2. **Proof Generation Time**: Time to generate ZK proofs
3. **Verification Time**: Time to verify proofs
4. **Communication Overhead**: Size of proofs and model updates
5. **End-to-End Latency**: Total time from training start to model update

### Reporting

Benchmark results are captured in standardized formats:

1. **JSON Reports**: Detailed performance metrics in JSON format
2. **CSV Reports**: Tabulated data for analysis in spreadsheets
3. **Remote Reporting**: Optional submission to a data warehouse endpoint

## Security Considerations

### Threat Model

FedZK addresses the following threats:

1. **Gradient Manipulation**: Clients submitting malicious gradients
2. **Model Poisoning**: Adversaries attempting to poison the global model
3. **Privacy Leakage**: Information leakage from gradients

### Mitigations

Security measures implemented include:

1. **Secure Circuits**: Constraints on gradient norms and sparsity
2. **API Authentication**: Preventing unauthorized access to services
3. **Batch Processing**: Protecting against inference attacks through chunking

## Future Work

Planned enhancements to the FedZK framework include:

1. **Differential Privacy**: Adding DP guarantees to the federated learning process
2. **Advanced Aggregation**: Implementing robust aggregation methods
3. **Recursive SNARKs**: Reducing verification costs for very large models
4. **Cross-Platform Support**: Extended support for mobile and embedded devices 