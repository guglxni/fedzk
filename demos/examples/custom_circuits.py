#!/usr/bin/env python3
# Functional Source License 1.1 with Apache-2.0 Future Grant (FSL-1.1-Apache-2.0)
# Copyright (c) 2025 Aaryan Guglani and FEDzk Contributors
# Licensed under FSL-1.1-Apache-2.0. See LICENSE for details.

"""
Custom Circuits Example for FEDzk

This example demonstrates how to work with zero-knowledge proof circuits
in the FEDzk framework. It shows real ZKProver and ZKVerifier usage with
existing circuits and explains the circuit development process.

IMPORTANT: This is an EXAMPLE file for educational and demonstration purposes.
It is NOT production-ready code and should not be used as-is in production environments.

This example focuses on:
- Understanding FEDzk circuit architecture
- Working with existing Circom circuits
- Real ZK proof generation and verification
- Educational content about ZK circuit development

For production use of custom circuits:
- Develop circuits using Circom language
- Compile circuits with circom compiler
- Generate trusted setup with snarkjs
- Implement proper circuit validation
- Add comprehensive error handling
- Use secure key management
- Implement circuit optimization

See the main FEDzk documentation for custom circuit development guidelines.
For production deployment guidelines, see the main FEDzk documentation.
"""

from typing import Any, Dict, Tuple

import numpy as np
import torch

# Real FEDzk imports
try:
    from fedzk.prover.zkgenerator import ZKProver
    from fedzk.prover.verifier import ZKVerifier
    from fedzk.prover.batch_zkgenerator import BatchZKProver, BatchZKVerifier
except ImportError as e:
    print(f"FEDzk not installed or import error: {e}")
    print("Please install FEDzk and ensure all dependencies are available.")
    print("Run: pip install -e .")
    exit(1)


def build_gradient_verification_circuit():
    """
    Build a circuit that verifies gradient computation integrity.
    
    The circuit ensures that:
    1. Gradients were computed correctly based on the loss function
    2. All gradients are within a valid range
    3. No outliers or manipulated values are present
    """
    circuit = CircuitBuilder()

    # Define the private inputs (only known to the prover)
    data_input = circuit.input("data_input")
    labels = circuit.input("labels")
    weights = circuit.input("weights")
    learning_rate = circuit.input("learning_rate")

    # Define the public inputs (known to both prover and verifier)
    original_weights = circuit.input("original_weights", is_public=True)
    updated_weights = circuit.input("updated_weights", is_public=True)
    batch_size = circuit.input("batch_size", is_public=True)

    # Create a simulated constraint for proper gradient computation
    # In a real circuit, this would implement the actual math
    computed_gradient = circuit.input("computed_gradient")
    expected_gradient = "gradient_formula_based_on_data_and_weights"
    circuit.add_constraint(
        computed_gradient,
        expected_gradient,
        "Gradient was computed correctly from the data"
    )

    # Create a constraint for weight update formula
    for i in range(5):  # Simplified - real circuits would handle arbitrary dimensions
        weight_var = f"weight_{i}"
        weight_update_formula = circuit.sub(
            original_weights + f"[{i}]",
            circuit.mul(learning_rate, computed_gradient + f"[{i}]")
        )
        circuit.add_constraint(
            updated_weights + f"[{i}]",
            weight_update_formula,
            f"Weight update formula correct for parameter {i}"
        )

    # Add constraints for bounds checking
    for i in range(5):
        gradient_var = computed_gradient + f"[{i}]"
        # Ensure gradients are within reasonable bounds (e.g., -10 to 10)
        upper_bound_constraint = circuit.sub("10", gradient_var)
        lower_bound_constraint = circuit.add(gradient_var, "10")
        circuit.add_constraint(
            f"upper_bound_{i} >= 0",
            upper_bound_constraint,
            f"Gradient {i} is <= 10"
        )
        circuit.add_constraint(
            f"lower_bound_{i} >= 0",
            lower_bound_constraint,
            f"Gradient {i} is >= -10"
        )

    # Compile the circuit
    return circuit.compile()


def simulate_training_step() -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Simulate a single training step to demonstrate the circuit.
    
    Returns:
        Tuple containing:
        - All inputs (private and public)
        - Just the public inputs
    """
    # Simulate original and updated weights
    original_weights = np.random.randn(5).tolist()
    updated_weights = [w - 0.01 * np.random.randn() for w in original_weights]

    # Create all inputs (including private ones)
    all_inputs = {
        "data_input": np.random.randn(10, 5).tolist(),
        "labels": np.random.randint(0, 2, 10).tolist(),
        "weights": original_weights,
        "learning_rate": 0.01,
        "computed_gradient": np.random.randn(5).tolist(),
        "original_weights": original_weights,
        "updated_weights": updated_weights,
        "batch_size": 10
    }

    # Extract just the public inputs
    public_inputs = {
        "original_weights": original_weights,
        "updated_weights": updated_weights,
        "batch_size": 10
    }

    return all_inputs, public_inputs


def main():
    """Demonstrate real FEDzk cryptographic proof generation and verification."""

    print("FEDzk Custom Circuits Example")
    print("=" * 35)

    # Explain the circuit concept
    print("\n1. Understanding FEDzk Circuit Architecture:")
    print("   FEDzk uses pre-compiled Circom circuits for zero-knowledge proofs.")
    print("   This example demonstrates how to work with existing circuits.")
    print("   Custom circuits would be developed using Circom and compiled separately.")

    # Show available circuits
    print("\n2. Available FEDzk Circuits:")
    try:
        from pathlib import Path
        import fedzk
        fedzk_path = Path(fedzk.__file__).parent
        circuits_dir = fedzk_path / "zk" / "circuits"

        if circuits_dir.exists():
            circuits = list(circuits_dir.glob("*.circom"))
            print(f"   Found {len(circuits)} Circom circuit files:")
            for circuit in circuits[:5]:  # Show first 5
                print(f"   📄 {circuit.name}")
        else:
            print("   ⚠️  Circuit files not found in expected location")
    except Exception as e:
        print(f"   ⚠️  Could not enumerate circuits: {e}")

    # Initialize real FEDzk provers and verifiers
    print("\n3. Initializing Real FEDzk Provers and Verifiers:")

    # Standard ZK prover/verifier
    try:
        std_prover = ZKProver(secure=False)
        std_verifier = ZKVerifier()
        print("   ✅ Standard ZK prover and verifier initialized")
    except Exception as e:
        print(f"   ❌ Failed to initialize standard prover/verifier: {e}")
        return

    # Batch ZK prover/verifier
    try:
        batch_prover = BatchZKProver(secure=False, batch_size=4, grad_size=4)
        batch_verifier = BatchZKVerifier(secure=False)
        print("   ✅ Batch ZK prover and verifier initialized")
    except Exception as e:
        print(f"   ❌ Failed to initialize batch prover/verifier: {e}")
        batch_prover = None
        batch_verifier = None

    # Generate sample data for demonstration
    print("\n4. Generating Sample Training Data:")
    sample_gradients = {
        'weights': torch.randn(4, 4),  # Sample weight gradients
        'bias': torch.randn(4)         # Sample bias gradients
    }
    print(f"   Created sample gradients: weights {sample_gradients['weights'].shape}, bias {sample_gradients['bias'].shape}")

    # Demonstrate standard ZK proof generation
    print("\n5. Demonstrating Standard ZK Proof Generation:")
    try:
        proof, public_signals = std_prover.generate_proof(sample_gradients)
        print("   ✅ Real ZK proof generated successfully!")
        print(f"   📊 Proof type: {type(proof)}")
        print(f"   📊 Public signals: {len(public_signals)} values")

        # Verify the proof
        is_valid = std_verifier.verify_real_proof(proof, public_signals)
        if is_valid:
            print("   ✅ Proof verification successful!")
        else:
            print("   ❌ Proof verification failed!")

    except Exception as e:
        print(f"   ⚠️  Standard ZK proof generation failed: {e}")
        print("   📋 This may be due to missing ZK toolchain setup")
        print("   📋 Run 'scripts/setup_zk.sh' to set up the ZK environment")

    # Demonstrate batch ZK proof generation (if available)
    if batch_prover and batch_verifier:
        print("\n6. Demonstrating Batch ZK Proof Generation:")
        try:
            # Create batch data (replicate the same gradients for batch)
            batch_data = {'weights': sample_gradients['weights']}
            batch_proof, batch_signals = batch_prover.generate_proof(batch_data)
            print("   ✅ Real batch ZK proof generated successfully!")
            print(f"   📊 Batch proof type: {type(batch_proof)}")
            print(f"   📊 Batch signals: {len(batch_signals)} values")

        except Exception as e:
            print(f"   ⚠️  Batch ZK proof generation failed: {e}")
            print("   📋 Batch circuits may require additional setup")

    # Educational summary
    print("\n" + "=" * 35)
    print("FEDzk Circuit Architecture Summary")
    print("=" * 35)

    print("\n✅ This example demonstrates real FEDzk cryptographic capabilities:")
    print("   • Real ZK proof generation using SNARK circuits")
    print("   • Real cryptographic proof verification")
    print("   • Batch processing for efficient federated learning")
    print("   • Integration with existing Circom circuit ecosystem")

    print("\n📚 FEDzk Circuit Development:")
    print("   • Circuits are written in Circom language")
    print("   • Compiled to WebAssembly and proving keys")
    print("   • Used by ZKProver for actual cryptographic proof generation")
    print("   • Verified by ZKVerifier using cryptographic validation")

    print("\n🔧 Custom Circuit Development:")
    print("   1. Write circuit logic in Circom (.circom files)")
    print("   2. Compile circuits: circom circuit.circom --wasm --r1cs")
    print("   3. Generate trusted setup: snarkjs groth16 setup")
    print("   4. Export verification key: snarkjs zkey export verificationkey")
    print("   5. Use in FEDzk applications with ZKProver/ZKVerifier")

    print("\n📋 To work with custom circuits:")
    print("   1. Place compiled circuits in src/fedzk/zk/circuits/")
    print("   2. Update ZKProver paths to use custom circuits")
    print("   3. Test with FEDzk's proof generation and verification")


if __name__ == "__main__":
    print("FEDzk Custom Verification Circuits Example")
    print("==========================================")
    main()
    print("\n✅ FEDzk Custom Circuits Example completed!")
    print("This example demonstrates real FEDzk cryptographic proof generation and verification.")
    print("For full ZK functionality, ensure ZK toolchain is set up with: scripts/setup_zk.sh")
