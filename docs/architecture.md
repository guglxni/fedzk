# Architecture Overview

This document describes the high-level architecture of FedZK, explaining how the different components interact with each other.

## System Components

FedZK consists of the following main components:

### 1. Client Nodes

Client nodes are responsible for:
- Loading and preprocessing local data
- Training models locally on private data
- Generating zero-knowledge proofs for model updates
- Sending model updates with proofs to the coordinator

### 2. Coordinator

The coordinator is responsible for:
- Managing the federated learning process
- Collecting model updates from clients
- Verifying proofs before accepting updates
- Aggregating validated model updates
- Distributing the global model back to clients

### 3. Zero-Knowledge Proof System

The zero-knowledge proof system:
- Provides cryptographic guarantees for model update integrity
- Ensures that updates adhere to predefined constraints
- Allows verification without revealing private data

### 4. MPC (Multi-Party Computation) Server

The MPC server provides additional privacy guarantees by:
- Enabling secure aggregation of model updates
- Preventing the reconstruction of individual contributions
- Supporting various secure computation protocols

## Architecture Diagram

```
┌────────────────┐     ┌─────────────────┐     ┌───────────────┐
│                │     │                 │     │               │
│  Client Node   │────▶│   Coordinator   │◀────│  Client Node  │
│  (Training)    │     │  (Aggregation)  │     │  (Training)   │
│                │     │                 │     │               │
└────────┬───────┘     └────────┬────────┘     └───────┬───────┘
         │                      │                      │
         │                      ▼                      │
         │              ┌───────────────┐              │
         └─────────────▶│   ZK Proofs   │◀─────────────┘
                        │ (Verification) │
                        └───────────────┘
```

## Data Flow

The data flow in a typical FedZK session follows these steps:

1. **Initialization**:
   - Coordinator configures the learning task and parameters
   - Clients connect to the coordinator and receive initial model
   
2. **Local Training**:
   - Each client trains the model on their local data
   - Clients compute model updates (weight differences)

3. **Proof Generation**:
   - Clients generate zero-knowledge proofs for their model updates
   - Proofs demonstrate that updates satisfy predefined constraints

4. **Verification**:
   - Coordinator receives updates and proofs from clients
   - Updates are accepted only if their proofs verify successfully

5. **Aggregation**:
   - Verified updates are aggregated using secure methods
   - MPC may be used to provide additional privacy guarantees

6. **Model Distribution**:
   - Updated global model is distributed back to clients
   - Process repeats for the next round of training

## Module Structure

FedZK's codebase is organized into the following modules:

- **`fedzk.client`**: Client-side training and proof generation
- **`fedzk.coordinator`**: Coordination and aggregation logic
- **`fedzk.prover`**: Zero-knowledge proof generation and verification
- **`fedzk.mpc`**: Secure multi-party computation for privacy
- **`fedzk.benchmark`**: Performance evaluation and benchmarking tools
- **`fedzk.cli`**: Command-line interface for various operations

## Security Considerations

FedZK's architecture incorporates several security measures:

- **Data Privacy**: Client data never leaves local environments
- **Proof Verification**: Cryptographic guarantees for update integrity
- **Secure Aggregation**: Protection against inference attacks
- **End-to-End Encryption**: Secure communication between components

## Extensibility

The architecture is designed for extensibility:

- Custom verification circuits can be defined
- Various aggregation strategies can be implemented
- Different machine learning models can be integrated
- Additional privacy mechanisms can be incorporated

For implementation details, please refer to the [Implementation Details](implementation_details.md) document. 