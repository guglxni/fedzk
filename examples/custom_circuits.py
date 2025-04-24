#!/usr/bin/env python3
# Copyright (c) 2025 Aaryan Guglani and FedZK Contributors
# SPDX-License-Identifier: MIT

"""
Custom Circuits Example for FedZK

This example demonstrates how to create custom verification circuits
for zero-knowledge proofs in the FedZK framework.
"""

import numpy as np
from typing import Dict, List, Tuple, Any

# Simulating FedZK imports
try:
    from fedzk.prover import CircuitBuilder, ZKProver
    from fedzk.prover.verifier import ZKVerifier
except ImportError:
    print("FedZK not installed. This is just an example file.")
    
    # Mock classes for example purposes
    class CircuitBuilder:
        def __init__(self):
            self.constraints = []
            
        def input(self, name, is_public=False):
            print(f"Adding {'public' if is_public else 'private'} input: {name}")
            return f"var_{name}"
            
        def add_constraint(self, lhs, rhs, description=""):
            self.constraints.append((lhs, rhs, description))
            print(f"Added constraint: {description or lhs} == {rhs}")
            
        def mul(self, a, b):
            return f"({a} * {b})"
            
        def add(self, a, b):
            return f"({a} + {b})"
            
        def sub(self, a, b):
            return f"({a} - {b})"
            
        def compile(self):
            print(f"Compiling circuit with {len(self.constraints)} constraints")
            return {"constraints": self.constraints}
    
    class ZKProver:
        def __init__(self, circuit=None):
            self.circuit = circuit
            
        def generate_proof(self, inputs):
            print(f"Generating proof with inputs: {inputs}")
            return {"proof": "mock_proof", "public_inputs": {k: v for k, v in inputs.items() if k.startswith("public_")}}
    
    class ZKVerifier:
        def __init__(self, circuit=None):
            self.circuit = circuit
            
        def verify_proof(self, proof, public_inputs):
            print(f"Verifying proof with public inputs: {public_inputs}")
            return True


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
    # Build the verification circuit
    print("\n1. Building custom verification circuit...")
    circuit = build_gradient_verification_circuit()
    
    # Initialize the prover and verifier with our circuit
    prover = ZKProver(circuit)
    verifier = ZKVerifier(circuit)
    
    # Simulate a training step
    print("\n2. Simulating a training step...")
    all_inputs, public_inputs = simulate_training_step()
    
    # Generate a proof
    print("\n3. Generating zero-knowledge proof...")
    proof = prover.generate_proof(all_inputs)
    
    # Verify the proof
    print("\n4. Verifying the proof...")
    is_valid = verifier.verify_proof(proof, public_inputs)
    
    # Show the result
    print("\nResult:", "✅ Valid proof!" if is_valid else "❌ Invalid proof!")
    print("\nThis demonstrates how custom verification circuits can be created to")
    print("ensure the integrity of specific computations in federated learning,")
    print("without revealing sensitive training data.")


if __name__ == "__main__":
    print("FedZK Custom Verification Circuits Example")
    print("==========================================")
    main()
    print("\nThis is a demonstration file. In a real deployment, you would use actual FedZK components.") 