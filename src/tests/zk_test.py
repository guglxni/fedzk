#!/usr/bin/env python3
# Functional Source License 1.1 with Apache-2.0 Future Grant (FSL-1.1-Apache-2.0)
# Copyright (c) 2025 Aaryan Guglani and FEDzk Contributors
# Licensed under FSL-1.1-Apache-2.0. See LICENSE for details.


"""
Direct test of ZK proof generation and verification.
"""

import torch
import sys
import time
import os
from pathlib import Path

# Add the current directory to the Python path
sys.path.insert(0, os.getcwd())

from fedzk.prover.zkgenerator import ZKProver
from fedzk.prover.verifier import ZKVerifier

# Define paths to local ZK assets
ZK_DIR = Path("src/fedzk/zk")
WASM_PATH = str(ZK_DIR / "model_update.wasm")
ZKEY_PATH = str(ZK_DIR / "proving_key.zkey")
VK_PATH = str(ZK_DIR / "verification_key.json")
SECURE_WASM_PATH = str(ZK_DIR / "model_update_secure.wasm")
SECURE_ZKEY_PATH = str(ZK_DIR / "proving_key_secure.zkey")
SECURE_VK_PATH = str(ZK_DIR / "verification_key_secure.json")

def test_standard_proof():
    """Test the standard ZK proof (no constraints)."""
    print("\n=== Testing Standard ZK Proof ===")
    
    # Create sample gradient dictionary with exactly 4 values to match the circuit
    gradient_dict = {
        "layer1.weight": torch.tensor([1.0, 2.0, 3.0, 4.0])
    }
    
    # Create ZKProver instance and override the paths
    prover = ZKProver(secure=False)
    prover.wasm_path = WASM_PATH
    prover.zkey_path = ZKEY_PATH
    
    print(f"Using WASM file: {WASM_PATH}")
    print(f"Using ZKEY file: {ZKEY_PATH}")
    print(f"Using VK file: {VK_PATH}")
    
    # Time the proof generation
    start_time = time.time()
    try:
        # Use max_inputs=4 to match the circuit definition
        proof, public_inputs = prover.generate_real_proof_standard(gradient_dict, max_inputs=4)
        gen_time = time.time() - start_time
        
        print(f"Proof generation time: {gen_time:.4f} seconds")
        
        # Create ZKVerifier instance with the verification key path
        verifier = ZKVerifier(verification_key_path=VK_PATH)
        
        # Time the verification
        start_time = time.time()
        result = verifier.verify_real_proof(proof, public_inputs)
        verify_time = time.time() - start_time
        
        print(f"Proof verification time: {verify_time:.4f} seconds")
        print(f"Verification result: {result}")
        
        return result
    except Exception as e:
        print(f"Standard proof generation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_secure_proof():
    """Test the secure ZK proof (with constraints)."""
    print("\n=== Testing Secure ZK Proof ===")
    
    # Create sample gradient dictionary with exactly 4 values to match the circuit
    gradient_dict = {
        "layer1.weight": torch.tensor([1.0, 2.0, 3.0, 4.0])
    }
    
    # Create ZKProver instance and override the paths
    prover = ZKProver(secure=True)
    prover.secure_wasm_path = SECURE_WASM_PATH
    prover.secure_zkey_path = SECURE_ZKEY_PATH
    
    print(f"Using secure WASM file: {SECURE_WASM_PATH}")
    print(f"Using secure ZKEY file: {SECURE_ZKEY_PATH}")
    print(f"Using secure VK file: {SECURE_VK_PATH}")
    
    # Time the proof generation
    start_time = time.time()
    # Sum of squares is 1^2 + 2^2 + 3^2 + 4^2 = 30
    max_norm_sq = 35.0  # Set high enough to succeed
    min_active_elements = 3
    try:
        # Use max_inputs=4 to match the circuit definition
        proof, public_inputs = prover.generate_real_proof_secure(
            gradient_dict, 
            max_norm_sq=max_norm_sq, 
            min_active_elements=min_active_elements,
            max_inputs=4
        )
        gen_time = time.time() - start_time
        
        print(f"Proof generation time: {gen_time:.4f} seconds")
        
        # Create ZKVerifier instance with the verification key path
        verifier = ZKVerifier(verification_key_path=SECURE_VK_PATH)
        
        # Time the verification
        start_time = time.time()
        # Use verify_real_proof_secure for the secure circuit
        result = verifier.verify_real_proof_secure(proof, public_inputs)
        verify_time = time.time() - start_time
        
        print(f"Proof verification time: {verify_time:.4f} seconds")
        print(f"Verification result: {result}")
        
        return result
    except Exception as e:
        print(f"Secure proof generation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("Testing ZK Proof System with real WASM/ZKey files")
    
    # Check if files exist
    for file_path in [WASM_PATH, ZKEY_PATH, VK_PATH, SECURE_WASM_PATH, SECURE_ZKEY_PATH, SECURE_VK_PATH]:
        if not os.path.exists(file_path):
            print(f"❌ Error: File not found: {file_path}")
            sys.exit(1)
    
    standard_result = test_standard_proof()
    secure_result = test_secure_proof()
    
    if standard_result and secure_result:
        print("\n✅ All ZK proofs generated and verified successfully!")
        sys.exit(0)
    else:
        print("\n❌ ZK proof testing failed")
        sys.exit(1) 