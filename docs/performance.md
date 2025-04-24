# Performance Tuning

This document provides guidance on optimizing the performance of FedZK for different deployment scenarios.

## Performance Considerations

When using FedZK, you may need to balance several performance aspects:

- **Computational efficiency**: Speed of training, proof generation, and verification
- **Memory usage**: RAM requirements during training and proof generation
- **Network bandwidth**: Size of model updates and proofs transmitted
- **Storage**: Disk space requirements for models and proofs
- **Battery life**: For mobile/edge deployments

## Hardware Recommendations

### Client-Side Hardware

For effective client operation:

| Component | Minimum | Recommended | Optimal |
|-----------|---------|-------------|---------|
| CPU | Dual-core 2GHz | Quad-core 3GHz | 8+ cores 3.5GHz+ |
| RAM | 4GB | 8GB | 16GB+ |
| GPU | Not required | Entry-level GPU | Mid-range CUDA-enabled |
| Storage | 1GB free | 5GB free | 20GB+ SSD |

### Coordinator Hardware

For effective coordinator operation:

| Component | Minimum | Recommended | Optimal |
|-----------|---------|-------------|---------|
| CPU | Quad-core 2.5GHz | 8-core 3GHz | 16+ cores 3.5GHz+ |
| RAM | 8GB | 16GB | 32GB+ |
| GPU | Not required | Mid-range GPU | High-end CUDA-enabled |
| Storage | 10GB free | 50GB SSD | 100GB+ NVMe SSD |
| Network | 100Mbps | 1Gbps | 10Gbps |

## Optimization Strategies

### Training Optimizations

1. **Batch Size Tuning**:
   - Larger batch sizes generally improve computational efficiency
   - Find the optimal batch size for your hardware using:
   ```python
   from fedzk.benchmark import find_optimal_batch_size
   optimal_batch_size = find_optimal_batch_size(model, data_sample, min_size=16, max_size=256)
   ```

2. **Mixed Precision Training**:
   - Enable for significant speed improvements on compatible hardware
   ```python
   trainer = Trainer(model_config=config, use_mixed_precision=True)
   ```

3. **Model Pruning**:
   - Reduce model size while maintaining accuracy
   ```python
   from fedzk.client import ModelPruner
   pruner = ModelPruner(target_sparsity=0.3)
   pruned_model = pruner.prune(model)
   ```

### Proof Generation Optimizations

1. **Circuit Optimization**:
   - Use simplified circuits for faster proof generation
   ```python
   trainer.set_circuit("optimized_model_update")  # Instead of "full_model_update"
   ```

2. **Parallel Proof Generation**:
   - Utilize multiple cores for proof generation
   ```python
   proof = trainer.generate_proof(updates, parallel_workers=4)
   ```

3. **GPU Acceleration**:
   - Leverage GPU for faster proof generation
   ```python
   proof = trainer.generate_proof(updates, use_gpu=True)
   ```

4. **Selective Proving**:
   - Generate proofs only for critical model components
   ```python
   proof = trainer.generate_proof(updates, selective_layers=["dense_1", "dense_2"])
   ```

### Communication Optimizations

1. **Update Compression**:
   - Compress model updates before transmission
   ```python
   from fedzk.client import UpdateCompressor
   compressor = UpdateCompressor(compression_ratio=0.5)
   compressed_updates = compressor.compress(updates)
   ```

2. **Proof Size Reduction**:
   - Use optimized circuits that generate smaller proofs
   ```python
   trainer.set_circuit("small_proof_circuit")
   ```

3. **Batched Updates**:
   - Send updates in batches to reduce connection overhead
   ```python
   coordinator.submit_updates_batch(updates_list, proofs_list)
   ```

## Profiling and Benchmarking

FedZK provides tools to measure and optimize performance:

### Performance Profiling

```python
from fedzk.benchmark import Profiler

# Profile training
with Profiler() as profiler:
    updates = trainer.train(data)
    
profiler.print_stats()
profiler.export_chrome_trace("training_profile.json")
```

### Comparative Benchmarking

```python
from fedzk.benchmark import Benchmarker

benchmarker = Benchmarker()
results = benchmarker.compare_configurations(
    model=model,
    data=data_sample,
    configs=[
        {"batch_size": 32, "use_gpu": False},
        {"batch_size": 64, "use_gpu": False},
        {"batch_size": 32, "use_gpu": True},
        {"batch_size": 64, "use_gpu": True}
    ]
)

benchmarker.visualize_results(results)
```

## Deployment-Specific Optimizations

### Low-Power Devices (IoT, Mobile)

For resource-constrained environments:

1. Use the lightweight model variants
2. Enable model quantization
3. Use simplified circuits
4. Consider server-side proof generation
5. Implement aggressive update compression

```python
# Configuration for low-power devices
trainer = Trainer(
    model_config={"architecture": "lightweight_mlp"},
    quantization="int8",
    circuit="minimal_constraints",
    compression_level="aggressive"
)
```

### High-Performance Clusters

For maximum performance:

1. Utilize distributed training
2. Enable GPU acceleration for all operations
3. Use optimized communication patterns
4. Implement parallel proof generation

```python
# Configuration for high-performance clusters
trainer = Trainer(
    model_config={"architecture": "complex_model"},
    distributed=True,
    num_workers=8,
    use_gpu=True,
    advanced_optimizations=True
)
```

## Memory Management

To minimize memory usage:

1. **Gradient Checkpointing**:
   ```python
   trainer = Trainer(model_config=config, use_gradient_checkpointing=True)
   ```

2. **Proof Stream Processing**:
   ```python
   proof = trainer.generate_proof_stream(updates, max_memory_usage="4G")
   ```

3. **Incremental Verification**:
   ```python
   for proof_chunk in proof_chunks:
       is_valid = verifier.verify_chunk(proof_chunk)
   ```

For more specific optimization guidance, refer to the [Deployment Guide](deployment_guide.md) document. 