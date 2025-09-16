#!/usr/bin/env python3
# Functional Source License 1.1 with Apache-2.0 Future Grant (FSL-1.1-Apache-2.0)
# Copyright (c) 2025 Aaryan Guglani and FEDzk Contributors
# Licensed under FSL-1.1-Apache-2.0. See LICENSE for details.


"""
Test the secure ZK circuit by directly creating a valid input JSON.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

# Define paths to ZK assets
ZK_DIR = Path("src/fedzk/zk")
SECURE_WASM_PATH = str(ZK_DIR / "model_update_secure.wasm")
SECURE_ZKEY_PATH = str(ZK_DIR / "proving_key_secure.zkey")
SECURE_VK_PATH = str(ZK_DIR / "verification_key_secure.json")

# Sample gradient values (integers)
gradients = [3, 5, 0, 7]

# Calculate sum of squares for norm
norm_sq = sum(g * g for g in gradients)  # 3^2 + 5^2 + 0^2 + 7^2 = 9 + 25 + 0 + 49 = 83

# Count non-zero elements
non_zero_count = sum(1 for g in gradients if g != 0)  # 3, 5, 7 are non-zero, so 3 elements

# Create input JSON with all required parameters (integers only)
input_data = {
    "gradients": gradients,
    "maxNorm": 100,  # Integer value higher than norm_sq (83)
    "minNonZero": 2  # Integer value lower than non_zero_count (3)
}

print(f"Generated secure circuit input:")
print(f"  Gradients: {gradients}")
print(f"  Sum of squares: {norm_sq}")
print(f"  Non-zero count: {non_zero_count}")
print(f"  Max norm (integer): {input_data['maxNorm']}")
print(f"  Min non-zero (integer): {input_data['minNonZero']}")

# Save input to file
with open("secure_input.json", "w") as f:
    json.dump(input_data, f)
print(f"Input saved to secure_input.json")

# Test command to run manually
print(f"\nManual test command:")
print(f"snarkjs wtns calculate {SECURE_WASM_PATH} secure_input.json secure_witness.wtns")

# Try to execute the command automatically
try:
    print("\nCalculating witness...")
    result = subprocess.run(
        ["snarkjs", "wtns", "calculate", SECURE_WASM_PATH, "secure_input.json", "secure_witness.wtns"],
        check=True,
        capture_output=True,
        text=True
    )
    print("Success! Witness calculation completed.")
    
    print("\nGenerating proof...")
    subprocess.run(
        ["snarkjs", "groth16", "prove", SECURE_ZKEY_PATH, "secure_witness.wtns", "secure_proof.json", "secure_public.json"],
        check=True
    )
    print("Success! Proof generation completed.")
    
    print("\nVerifying proof...")
    verify_result = subprocess.run(
        ["snarkjs", "groth16", "verify", SECURE_VK_PATH, "secure_public.json", "secure_proof.json"],
        check=True,
        capture_output=True,
        text=True
    )
    print(f"Verification result: {'OK' in verify_result.stdout}")
    
    if 'OK' in verify_result.stdout:
        print("\n✅ Secure circuit test PASSED!")
        sys.exit(0)
    else:
        print("\n❌ Secure circuit test FAILED: Verification failed")
        sys.exit(1)
except subprocess.CalledProcessError as e:
    print(f"\n❌ Secure circuit test FAILED: {e}")
    if hasattr(e, 'stderr') and e.stderr:
        print(f"Error output: {e.stderr}")
    sys.exit(1) 