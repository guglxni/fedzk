# 🚀 Getting Started with FEDZK

Welcome to FEDZK! This guide will help you get started with federated learning using zero-knowledge proofs for maximum privacy and security.

## Table of Contents

1. [Quick Start](#quick-start)
2. [Installation](#installation)
3. [Basic Concepts](#basic-concepts)
4. [Your First Federation](#your-first-federation)
5. [Advanced Usage](#advanced-usage)
6. [Troubleshooting](#troubleshooting)

## Quick Start

```bash
# Install FEDZK
pip install fedzk

# Start a simple federation
python -c "
import fedzk
from fedzk.core import FederatedLearning

# Create a federation
federation = FederatedLearning.create_federation(
    name='My First Federation',
    model_config={'type': 'neural_network', 'layers': [784, 128, 10]},
    privacy_config={'zk_enabled': True, 'differential_privacy': True}
)

print(f'Federation created: {federation.id}')
"
```

## Installation

### Prerequisites

- Python 3.8+
- pip package manager
- Git (for cloning the repository)

### Install from PyPI

```bash
pip install fedzk
```

### Install from Source

```bash
git clone https://github.com/fedzk/fedzk.git
cd fedzk
pip install -e .
```

### Install with Optional Dependencies

```bash
# For GPU acceleration
pip install fedzk[gpu]

# For development
pip install fedzk[dev]

# For documentation
pip install fedzk[docs]
```

## Basic Concepts

### Federation
A federation is a collaborative group of organizations that train machine learning models together without sharing their raw data.

### Zero-Knowledge Proofs
Cryptographic proofs that verify computations without revealing the underlying data.

### Differential Privacy
Adds noise to data to prevent identification of individuals while preserving statistical properties.

### Multi-Party Computation
Secure computation protocols that allow multiple parties to jointly compute functions on their private data.

## Your First Federation

### Step 1: Import FEDZK

```python
import fedzk
from fedzk.core import FederatedLearning, ModelConfig, PrivacyConfig
```

### Step 2: Configure Your Model

```python
# Define your machine learning model
model_config = ModelConfig(
    model_type='neural_network',
    input_shape=(784,),
    output_shape=(10,),
    hidden_layers=[128, 64],
    activation='relu',
    optimizer='adam',
    learning_rate=0.001
)
```

### Step 3: Set Privacy Parameters

```python
# Configure privacy and security
privacy_config = PrivacyConfig(
    enable_zk_proofs=True,
    differential_privacy={
        'epsilon': 1.0,
        'delta': 1e-5,
        'noise_multiplier': 1.0
    },
    encryption='AES-256',
    secure_aggregation=True
)
```

### Step 4: Create the Federation

```python
# Initialize the federation
federation = FederatedLearning.create_federation(
    name='Healthcare Research Network',
    description='Collaborative ML for medical research',
    model_config=model_config,
    privacy_config=privacy_config,
    min_participants=3,
    max_participants=10
)

print(f"🎉 Federation created with ID: {federation.id}")
```

### Step 5: Join as a Participant

```python
# Generate keys for secure communication
from fedzk.crypto import KeyManager
key_manager = KeyManager()
public_key, private_key = key_manager.generate_keypair()

# Join the federation
participant = federation.join(
    participant_id='hospital_a',
    public_key=public_key,
    data_info={
        'dataset_size': 50000,
        'data_type': 'medical_images',
        'quality_score': 0.95
    }
)

print(f"✅ Joined federation as: {participant.id}")
```

### Step 6: Start Training

```python
# Configure training parameters
training_config = {
    'epochs': 100,
    'batch_size': 32,
    'learning_rate': 0.001,
    'aggregation_method': 'fedavg',
    'early_stopping': {
        'patience': 10,
        'min_delta': 0.001
    }
}

# Start federated training
training_session = federation.start_training(
    training_config=training_config,
    validation_split=0.2,
    save_checkpoints=True
)

print(f"🚀 Training started with session ID: {training_session.id}")
```

### Step 7: Monitor Progress

```python
# Monitor training progress
while training_session.is_active:
    status = training_session.get_status()
    print(f"Epoch {status.current_epoch}/{status.total_epochs}")
    print(f"Loss: {status.loss:.4f}, Accuracy: {status.accuracy:.4f}")
    print(f"Active participants: {status.active_participants}")

    time.sleep(30)  # Check every 30 seconds
```

### Step 8: Get Final Model

```python
# Get the trained model
final_model = federation.get_model()

# Save the model
final_model.save('trained_model.pkl')

# Evaluate on test data
test_accuracy = final_model.evaluate(test_data)
print(f"Final test accuracy: {test_accuracy:.4f}")
```

## Advanced Usage

### Custom Zero-Knowledge Circuits

```python
from fedzk.zk import CircuitBuilder

# Create a custom ZK circuit
circuit = CircuitBuilder()
circuit.add_constraint('input_validation', 'x > 0')
circuit.add_constraint('range_proof', 'x >= min_val AND x <= max_val')
circuit.add_constraint('computation_proof', 'result = x * coefficient')

# Compile and deploy
compiled_circuit = circuit.compile()
federation.set_zk_circuit(compiled_circuit)
```

### Differential Privacy Configuration

```python
# Advanced DP configuration
privacy_config = PrivacyConfig(
    differential_privacy={
        'epsilon': 0.5,  # Stronger privacy
        'delta': 1e-6,   # Tighter bound
        'noise_mechanism': 'gaussian',
        'clipping_norm': 1.0,
        'adaptive_noise': True
    }
)
```

### Multi-Model Training

```python
# Train multiple models simultaneously
models = {
    'classifier': ModelConfig(model_type='neural_network', ...),
    'regressor': ModelConfig(model_type='linear_regression', ...),
    'anomaly_detector': ModelConfig(model_type='autoencoder', ...)
}

federation = FederatedLearning.create_multi_model_federation(
    name='Multi-Task Learning',
    models=models,
    shared_privacy_config=privacy_config
)
```

## Troubleshooting

### Common Issues

#### 1. Import Errors
```bash
# Make sure FEDZK is installed
pip install fedzk

# Check Python version
python --version  # Should be 3.8+
```

#### 2. Network Connectivity
```python
# Test federation connectivity
federation.test_connectivity()

# Check participant status
for participant in federation.participants:
    print(f"{participant.id}: {participant.status}")
```

#### 3. Memory Issues
```python
# Reduce batch size
training_config = {
    'batch_size': 16,  # Try smaller values
    'gradient_accumulation_steps': 4
}

# Enable memory optimization
federation.enable_memory_optimization()
```

#### 4. Privacy Parameter Tuning
```python
# Start with relaxed privacy for testing
privacy_config = PrivacyConfig(
    differential_privacy={
        'epsilon': 5.0,  # Higher epsilon = less privacy but better utility
        'delta': 1e-3
    }
)

# Gradually tighten privacy constraints
```

#### 5. ZK Proof Generation Issues
```python
# Check ZK toolchain
from fedzk.zk import ZKValidator
validator = ZKValidator()
if validator.validate_toolchain():
    print("✅ ZK toolchain is ready")
else:
    print("❌ ZK toolchain issues detected")
    validator.install_dependencies()
```

### Performance Optimization

```python
# Enable GPU acceleration
federation.enable_gpu_acceleration()

# Configure parallel processing
federation.set_parallel_processing(
    num_workers=4,
    use_async=True
)

# Optimize network communication
federation.optimize_network(
    compression=True,
    batch_updates=True
)
```

### Logging and Debugging

```python
import logging

# Enable detailed logging
logging.basicConfig(level=logging.DEBUG)

# Log to file
logging.basicConfig(
    filename='fedzk_debug.log',
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Monitor federation health
health_monitor = federation.get_health_monitor()
health_status = health_monitor.check_all_systems()
print(f"System health: {health_status}")
```

## Next Steps

- 📖 Read the [API Documentation](./api/index.html)
- 🔧 Check out [Deployment Guides](./deployment/)
- 🎯 Explore [Use Case Examples](../examples/)
- 🤝 Join our [Community](../community/)

---

For more advanced topics, visit our [Developer Guide](./developer/) or [API Reference](./api/).
