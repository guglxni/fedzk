# FedZK Architecture

## Overview

FedZK is a federated learning framework with zero-knowledge proofs for secure, verifiable model updates. The architecture follows a client-server model with these key components:

1. **Client Components** - for local training and proof generation
2. **Coordinator** - central server that aggregates model updates
3. **MPC Server** - Multi-Party Computation server for secure proof generation
4. **Zero-Knowledge Components** - for creating and validating proofs 
5. **CLI** - Command-line interface for interacting with the system

## System Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                                                                     │
│  ┌─────────┐   Train   ┌─────────┐   Generate   ┌─────────────────┐ │
│  │         │ ───────▶ │         │ ───────────▶ │                 │ │
│  │ Dataset │           │ Trainer │              │ ZK Generator    │ │
│  │         │ ◀─────── │         │ ◀─────────── │                 │ │
│  └─────────┘   Model   └─────────┘   Gradients  └─────────────────┘ │
│                                                          │          │
│                                                          │ Proof    │
│                          CLIENT                          ▼          │
└────────────────────────────────────────┬────────────────────────────┘
                                         │
                                         │ Submit
                                         │ (Proof + Model Update)
                                         ▼
┌────────────────────────────────────────┴────────────────────────────┐
│                                                                     │
│  ┌─────────────────┐   Verify    ┌─────────────────────────────┐    │
│  │                 │ ◀─────────  │                             │    │
│  │ ZK Verifier     │             │ Coordinator API             │    │
│  │                 │ ───────────▶│ (Model Aggregation)         │    │
│  └─────────────────┘   Result    └─────────────────────────────┘    │
│                                                                     │
│                          SERVER                                     │
└─────────────────────────────────────────────────────────────────────┘
      │                                         ▲
      │                                         │
      │ Outsource                               │ Return
      │ Proof Generation                        │ Proofs
      ▼                                         │
┌─────────────────────────────────────────────────────────────────────┐
│                                                                     │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │                                                             │    │
│  │ MPC Server (Multi-Party Computation)                        │    │
│  │                                                             │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                                                                     │
│                      AUXILIARY SERVICE                              │
└─────────────────────────────────────────────────────────────────────┘
```

## Core Components

### Client Components

#### LocalTrainer

The `LocalTrainer` manages the client-side operations:

- Loads and preprocesses local datasets
- Trains local models using gradient descent
- Generates zero-knowledge proofs of valid model updates
- Submits updates to the coordinator

```python
# Example of client training flow
trainer = LocalTrainer(model, dataset)
trainer.train(epochs=5)
trainer.generate_proof(secure=True)
trainer.submit_update(coordinator_url)
```

#### ZKGenerator

Responsible for generating zero-knowledge proofs that validate model updates without revealing the training data:

- Creates arithmetic circuits for various constraints
- Generates zero-knowledge proofs of gradient bounds
- Supports both secure and fast proving modes

### Server Components

#### CoordinatorAPI

The central server that:

- Receives model updates from clients
- Verifies proofs before accepting updates
- Aggregates model updates using federated averaging
- Distributes the updated global model

```python
# Example coordinator startup
coordinator = CoordinatorAPI(
    model_aggregation_strategy="federated_avg",
    verification_required=True
)
```

#### ZKVerifier

Validates zero-knowledge proofs from clients:

- Verifies that model updates satisfy constraint requirements
- Ensures updates are properly bounded

#### MPCServer

An optional service for secure multi-party computation:

- Provides proof generation as a service
- Useful for clients with limited computational resources
- Ensures proofs are generated without seeing actual data

## Data Flow

1. **Training Phase**
   - Client loads local dataset and model
   - Client trains model locally
   - Client computes gradients

2. **Proof Generation Phase**
   - Client generates ZK proof of valid update (bounded gradient)
   - Alternatively, client may use MPC server for proof generation

3. **Submission Phase**
   - Client submits model update with proof to coordinator
   - Coordinator verifies the proof
   - If valid, coordinator accepts the update

4. **Aggregation Phase**
   - Coordinator aggregates all valid updates
   - Coordinator updates the global model
   - New global model becomes available

## Security Features

FedZK implements several security features:

- **Zero-Knowledge Proofs**: Validate model updates without revealing private data
- **Gradient Bounding**: Ensures updates cannot contain malicious values
- **Secure MPC**: Optional secure computation for proof generation
- **API Key Authentication**: Secures server endpoints
- **Fallback Modes**: Graceful degradation when secure proofs are unavailable

## Implementation Details

### Circuit Design

The system supports multiple ZK circuit designs:

- `BoundedGradient`: Ensures gradients are within bounds
- `L2NormBounded`: Constrains the L2 norm of gradient updates
- `MinActiveElements`: Ensures a minimum number of active training examples

### CLI Structure

The CLI provides a unified interface to all components:

- `setup`: Initialize circuits
- `generate`: Generate proofs
- `verify`: Verify proofs
- `mpc serve`: Run MPC server
- `benchmark`: Performance testing

### Benchmarking

The system includes an end-to-end benchmarking framework:

- Measures training time, proof generation, and verification
- Simulates multiple clients
- Reports metrics in JSON/CSV formats

## Deployment Strategies

FedZK supports multiple deployment models:

### Standalone

Each component runs locally, suitable for development and testing.

### Distributed

- Coordinator runs on a central server
- MPC server runs on a separate high-performance machine
- Clients connect remotely

### Containerized

Docker configurations for easy deployment:

```
docker-compose up -d coordinator mpc-server
```

## Extension Points

FedZK is designed to be extensible:

1. **Custom Circuits**: New ZK constraints can be added
2. **Aggregation Strategies**: Beyond federated averaging
3. **Model Architectures**: Support for different neural network types
4. **Integration APIs**: Connect with existing ML platforms 