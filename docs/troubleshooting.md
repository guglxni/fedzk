# Troubleshooting Guide

This guide provides solutions to common issues you might encounter when using FedZK.

## Installation Issues

### Package Installation Fails

**Problem**: Error installing FedZK or its dependencies.

**Solutions**:

1. Ensure you have the correct Python version (3.8+):
   ```bash
   python --version
   ```

2. Upgrade pip:
   ```bash
   python -m pip install --upgrade pip
   ```

3. Install system prerequisites:
   ```bash
   # On Ubuntu/Debian
   sudo apt-get install build-essential libssl-dev libffi-dev python3-dev
   
   # On macOS
   brew install openssl
   ```

4. Try installing with verbose output to identify specific issues:
   ```bash
   pip install -e . -v
   ```

### Cryptographic Dependencies Issues

**Problem**: Errors related to cryptographic libraries.

**Solution**:
```bash
# Install required system libraries
sudo apt-get install libsodium-dev

# Reinstall cryptographic dependencies
pip uninstall cryptography
pip install cryptography
```

## Runtime Issues

### Circuit Compilation Errors

**Problem**: "Circuit compilation failed" or similar errors.

**Solutions**:

1. Check Circom installation:
   ```bash
   circom --version
   ```
   If not found, install with:
   ```bash
   npm install -g circom
   ```

2. Verify circuit syntax:
   ```bash
   circom --check path/to/circuit.circom
   ```

3. Try with a simpler circuit for testing:
   ```python
   trainer.set_circuit("minimal_test")
   ```

### Memory Errors During Proof Generation

**Problem**: Out of memory errors during proof generation.

**Solutions**:

1. Reduce the model size:
   ```python
   trainer = Trainer(model_config={
       'architecture': 'mlp',
       'layers': [784, 64, 10],  # Smaller hidden layer
   })
   ```

2. Use gradient checkpointing:
   ```python
   trainer = Trainer(model_config=config, use_gradient_checkpointing=True)
   ```

3. Enable stream processing:
   ```python
   proof = trainer.generate_proof_stream(updates, max_memory_usage="4G")
   ```

4. Increase your system's swap space temporarily.

### Verification Failures

**Problem**: Proof verification fails even though the model update seems correct.

**Solutions**:

1. Check that the proof and public inputs match:
   ```python
   from fedzk.debug import verify_proof_structure
   verify_proof_structure(proof, public_inputs)
   ```

2. Ensure the verifier has the correct verification key:
   ```python
   verifier.load_verification_key("path/to/correct/verification_key.json")
   ```

3. Inspect the proof with the debugging tool:
   ```python
   from fedzk.debug import ProofInspector
   inspector = ProofInspector(proof_file="proof.json")
   inspector.validate_structure()
   inspector.analyze_complexity()
   ```

## Communication Issues

### Coordinator Connection Failures

**Problem**: Clients cannot connect to the coordinator.

**Solutions**:

1. Check network connectivity:
   ```bash
   ping coordinator_address
   ```

2. Verify the coordinator is running:
   ```bash
   curl http://coordinator_address:port/health
   ```

3. Check firewall settings to ensure the port is open.

4. Verify SSL certificates if using HTTPS.

### Timeouts During Update Submission

**Problem**: Timeouts when submitting large model updates.

**Solutions**:

1. Increase the timeout setting:
   ```python
   coordinator.submit_update(updates, proof, timeout=120)  # 120 seconds
   ```

2. Use compressed updates:
   ```python
   from fedzk.client import UpdateCompressor
   compressed_updates = UpdateCompressor().compress(updates)
   ```

3. Split large updates into smaller chunks:
   ```python
   from fedzk.client import UpdateSplitter
   update_chunks = UpdateSplitter().split(updates, max_size_mb=10)
   for chunk in update_chunks:
       coordinator.submit_update_chunk(chunk)
   ```

## Performance Issues

### Slow Training or Proof Generation

**Problem**: Training or proof generation is too slow.

**Solutions**:

1. Enable GPU acceleration if available:
   ```python
   trainer = Trainer(model_config=config, use_gpu=True)
   ```

2. Use mixed precision training:
   ```python
   trainer = Trainer(model_config=config, use_mixed_precision=True)
   ```

3. Optimize batch size:
   ```python
   from fedzk.benchmark import find_optimal_batch_size
   optimal_batch_size = find_optimal_batch_size(model, data_sample)
   ```

4. Use a simpler circuit for proof generation:
   ```python
   trainer.set_circuit("optimized_model_update")
   ```

### High Memory Usage

**Problem**: FedZK is using too much memory.

**Solutions**:

1. Use a smaller model:
   ```python
   trainer = Trainer(model_config={"architecture": "lightweight_mlp"})
   ```

2. Enable memory efficient inference:
   ```python
   trainer = Trainer(model_config=config, memory_efficient=True)
   ```

3. Clear cache between operations:
   ```python
   import gc
   
   # After training
   updates = trainer.train(data)
   gc.collect()
   
   # After proof generation
   proof = trainer.generate_proof(updates)
   gc.collect()
   ```

## Circuit-Specific Issues

### Constraint System Errors

**Problem**: "Constraint system not satisfied" or similar errors.

**Solutions**:

1. Check input ranges match circuit expectations:
   ```python
   from fedzk.debug import check_input_ranges
   check_input_ranges(updates, "model_update")
   ```

2. Normalize inputs if needed:
   ```python
   from fedzk.client import normalize_updates
   normalized_updates = normalize_updates(updates, max_value=1000)
   ```

3. Use the circuit debugger to trace constraints:
   ```python
   from fedzk.debug import CircuitDebugger
   debugger = CircuitDebugger("model_update.circom")
   debugger.trace_constraints(sample_inputs)
   ```

## Common Error Messages

### "No proving key found"

**Solution**: Ensure the proving key is in the correct location:
```python
trainer.set_proving_key_path("path/to/proving_key.json")
```

### "Invalid witness generation"

**Solution**: Check that the circuit inputs are correctly formatted:
```python
from fedzk.debug import validate_circuit_inputs
validate_circuit_inputs(inputs, circuit_name="model_update")
```

### "Public inputs don't match"

**Solution**: Ensure public inputs are correctly extracted:
```python
from fedzk.prover import extract_public_inputs
public_inputs = extract_public_inputs(circuit_inputs, circuit_name="model_update")
```

## Getting Further Help

If you encounter issues not covered in this guide:

1. Check the [GitHub Issues](https://github.com/guglxni/fedzk/issues) for similar reports
2. Join our [Slack workspace](https://fedzk-community.slack.com) for real-time support
3. Post your question to our [mailing list](https://groups.google.com/g/fedzk-users)
4. If you suspect a bug, [file a detailed bug report](https://github.com/guglxni/fedzk/issues/new) including:
   - FedZK version
   - Python version
   - OS information
   - Error message and stack trace
   - Steps to reproduce 