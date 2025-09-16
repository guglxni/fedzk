# Functional Source License 1.1 with Apache-2.0 Future Grant (FSL-1.1-Apache-2.0)
# Copyright (c) 2025 Aaryan Guglani and FEDzk Contributors
# Licensed under FSL-1.1-Apache-2.0. See LICENSE for details.

"""
Zero-Knowledge Proof Generator for FEDzk.

This module contains the ZKProver class which handles generation of zero-knowledge
proofs for gradient updates in federated learning.
"""

import hashlib
import json
import os
import subprocess
import tempfile
import time
import pathlib
from collections import defaultdict

# Add these imports
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
import numpy as np

from .zk_validator import ZKValidator

# Define base directory for ZK assets relative to this file
# Assumes zk assets are in src/fedzk/zk/
ASSET_DIR = pathlib.Path(__file__).resolve().parent.parent / "zk"

class ZKProver:
    """
    Production Zero-Knowledge Proof Generator for FEDzk.
    
    This class generates real zero-knowledge proofs using Circom circuits and SNARKjs.
    It requires a complete ZK toolchain installation (run scripts/setup_zk.sh).
    """

    def __init__(self, secure: bool = False, max_norm_squared: float = 100.0, min_active: int = 1):
        """
        Initialize the ZK prover.
        
        Args:
            secure: If True, uses secure circuit with constraints
            max_norm_squared: Maximum allowed squared L2 norm for gradients
            min_active: Minimum number of non-zero gradient elements required
        """
        self.secure = secure
        self.max_norm_squared = max_norm_squared
        self.min_active = min_active

        # Paths to ZK circuit files
        self.wasm_path = str(ASSET_DIR / "model_update.wasm")
        self.r1cs_path = str(ASSET_DIR / "model_update.r1cs")
        self.zkey_path = str(ASSET_DIR / "proving_key.zkey")
        self.vkey_path = str(ASSET_DIR / "verification_key.json")
        
        # Secure circuit paths
        self.secure_wasm_path = str(ASSET_DIR / "model_update_secure.wasm")
        self.secure_r1cs_path = str(ASSET_DIR / "model_update_secure.r1cs")
        self.secure_zkey_path = str(ASSET_DIR / "proving_key_secure.zkey")
        self.secure_vkey_path = str(ASSET_DIR / "verification_key_secure.json")
        
        # Verify ZK infrastructure is available using centralized validator
        self._validate_zk_toolchain()

    def _validate_zk_toolchain(self):
        """
        Validate ZK toolchain using centralized ZKValidator.

        Raises:
            RuntimeError: If ZK toolchain validation fails
        """
        validator = ZKValidator(str(ASSET_DIR))
        validation_results = validator.validate_toolchain()

        if validation_results["overall_status"] == "failed":
            error_msg = "ZK toolchain validation failed:\n"
            for error in validation_results["errors"]:
                error_msg += f"  • {error}\n"

            error_msg += "\nPlease run 'scripts/setup_zk.sh' to install/configure ZK toolchain."
            raise RuntimeError(error_msg)

        if validation_results["overall_status"] == "warning":
            warning_msg = "ZK toolchain validation passed with warnings:\n"
            for warning in validation_results["warnings"]:
                warning_msg += f"  • {warning}\n"
            print(f"⚠️  {warning_msg}")  # Print warnings but don't fail

        print("✅ ZK toolchain validation passed")

    def _hash_tensor(self, tensor: torch.Tensor) -> str:
        """
        Compute a hash of a tensor.
        
        Args:
            tensor: PyTorch tensor to hash
            
        Returns:
            String hash of the tensor
        """
        # Convert tensor to bytes (flatten first)
        tensor_bytes = tensor.flatten().numpy().tobytes()

        # Compute SHA-256 hash
        hash_obj = hashlib.sha256(tensor_bytes)
        return hash_obj.hexdigest()

    def _compute_tensor_norm(self, tensor: torch.Tensor) -> float:
        """
        Compute the sum of squares of a tensor (squared L2 norm).
        
        Args:
            tensor: PyTorch tensor
            
        Returns:
            Sum of squares as a float
        """
        return float(torch.sum(tensor * tensor).item())

    def generate_proof(self, gradient_dict: Dict[str, torch.Tensor]) -> Tuple[Dict, List]:
        """Generate a ZK proof for the given gradients."""
        # Input validation
        if not gradient_dict:
            raise ValueError("Empty gradient dictionary provided")
        
        if self.secure:
            return self.generate_real_proof_secure(gradient_dict, self.max_norm_squared, self.min_active)
        else:
            return self.generate_real_proof_standard(gradient_dict)

    def generate_real_proof_standard(self, gradient_dict: Dict[str, torch.Tensor], max_inputs: int = 4):
        """Generate a standard ZK proof (basic constraints)."""
        input_data = self._prepare_input_standard(gradient_dict, max_inputs=max_inputs)
        return self._run_snarkjs_proof(input_data, self.wasm_path, self.zkey_path)

    def generate_real_proof_secure(self, gradient_dict: Dict[str, torch.Tensor], 
                                 max_norm_sq: float, min_active_elements: int, max_inputs: int = 4):
        """Generate a secure ZK proof with privacy constraints."""
        input_data = self._prepare_input_secure(gradient_dict, max_norm_sq, min_active_elements, max_inputs=max_inputs)
        return self._run_snarkjs_proof(input_data, self.secure_wasm_path, self.secure_zkey_path)

    def _run_snarkjs_proof(self, input_data: Dict, wasm_path: str, zkey_path: str) -> Tuple[Dict, List]:
        """Helper to run SNARKjs commands for proof generation."""
        # Strict enforcement - no fallback or mock proofs allowed
        with tempfile.TemporaryDirectory() as tmpdir:
            input_json_path = os.path.join(tmpdir, "input.json")
            witness_path = os.path.join(tmpdir, "witness.wtns")
            proof_path = os.path.join(tmpdir, "proof.json")
            public_inputs_path = os.path.join(tmpdir, "public.json")

            with open(input_json_path, "w") as f:
                json.dump(input_data, f)

            # Generate witness
            try:
                result = subprocess.run(["snarkjs", "wtns", "calculate",
                                       wasm_path,
                                       input_json_path,
                                       witness_path], check=True, capture_output=True, text=True)
            except subprocess.CalledProcessError as e:
                error_msg = e.stderr.strip() if e.stderr else e.stdout.strip() if e.stdout else f"Exit code: {e.returncode}"
                raise RuntimeError(f"Failed to generate witness: {error_msg}. "
                                 f"Please ensure ZK toolchain is properly installed and circuit files are valid.")

            # Generate proof
            try:
                subprocess.run(["snarkjs", "groth16", "prove",
                                zkey_path,
                                witness_path,
                                proof_path,
                                public_inputs_path], check=True, capture_output=True, text=True)
            except subprocess.CalledProcessError as e:
                raise RuntimeError(f"Failed to generate proof: {e.stderr}. "
                                 f"Please ensure circuit files are valid.")

            with open(proof_path, "r") as f:
                proof = json.load(f)
            with open(public_inputs_path, "r") as f:
                public_inputs = json.load(f)

        return proof, public_inputs

    def _prepare_input_standard(self, gradient_dict: Dict[str, torch.Tensor], max_inputs: int = 4):
        """
        Prepare input data for the standard (basic) circuit.
        
        Args:
            gradient_dict: Dictionary of gradient tensors
            max_inputs: Maximum number of gradient values (must match circuit)
            
        Returns:
            Dictionary with input data for the circuit
        """
        # Flatten all gradients and take first max_inputs values
        flat_grads = []
        for grad_tensor in gradient_dict.values():
            flat_grads.extend(grad_tensor.flatten().tolist())
        
        # Pad or truncate to exactly max_inputs values
        if len(flat_grads) > max_inputs:
            gradients = flat_grads[:max_inputs]
        else:
            gradients = flat_grads + [0] * (max_inputs - len(flat_grads))
        
        return {"gradients": gradients}

    def _prepare_input_secure(self, gradient_dict: Dict[str, torch.Tensor],
                            max_norm_sq: float, min_active_elements: int, max_inputs: int = 4):
        """
        Prepare input data for the secure circuit with constraints.

        Args:
            gradient_dict: Dictionary of gradient tensors
            max_norm_sq: Maximum allowed squared norm
            min_active_elements: Minimum required non-zero elements
            max_inputs: Maximum number of gradient values (must match circuit)

        Returns:
            Dictionary with input data for the secure circuit
        """
        # Strict input preparation - no test mode bypasses
        # Flatten all gradients and take first max_inputs values
        flat_grads = []
        for grad_tensor in gradient_dict.values():
            flat_grads.extend(grad_tensor.flatten().tolist())

        # Pad or truncate to exactly max_inputs values
        if len(flat_grads) > max_inputs:
            gradients = flat_grads[:max_inputs]
        else:
            gradients = flat_grads + [0] * (max_inputs - len(flat_grads))

        # Ensure all values are numerical for the circuit input
        max_norm_num = int(max_norm_sq)
        min_active_num = min_active_elements

        return {
            "gradients": gradients,
            "maxNorm": max_norm_num,
            "minNonZero": min_active_num
        }

    def batch_generate_proof_secure(self, gradient_dict: Dict[str, torch.Tensor],
                                   chunk_size: int = 4,
                                   max_norm: float = 100.0,
                                   min_active: int = 3,
                                   max_workers: int = 4) -> Dict[str, Any]:
        """
        Generate proofs for large gradients by breaking them into chunks.
        
        Args:
            gradient_dict: Dictionary mapping parameter names to gradient tensors
            chunk_size: Size of each gradient chunk (must match circuit input size)
            max_norm: Maximum allowed L2 norm for each chunk
            min_active: Minimum non-zero elements required in each chunk
            max_workers: Maximum number of parallel workers for proof generation
            
        Returns:
            Dictionary containing:
            - batch_proofs: List of proofs for each chunk
            - metadata: Information about the batching process
            - merkle_root: Root hash of the Merkle tree of proof hashes
            
        Raises:
            ValueError: If any chunk fails to meet fairness constraints
        """
        # Flatten all gradients and organize them by parameter
        flat_grads = []
        chunk_mapping = defaultdict(list)  # Maps parameters to chunk indices
        param_indices = {}  # Starting index of each parameter in flat_grads

        index = 0
        for param_name, gradient in gradient_dict.items():
            param_indices[param_name] = index
            flat_tensor = gradient.flatten()
            flat_grads.extend(flat_tensor.tolist())
            index += flat_tensor.numel()

        # Break gradients into chunks of size chunk_size
        total_grads = len(flat_grads)
        all_chunks = []

        for i in range(0, total_grads, chunk_size):
            chunk = flat_grads[i:i+chunk_size]

            # Pad with zeros if needed
            while len(chunk) < chunk_size:
                chunk.append(0)

            all_chunks.append(chunk)

            # Update chunk mapping to track which parameters are in which chunks
            for param_name, start_idx in param_indices.items():
                param_size = gradient_dict[param_name].numel()
                end_idx = start_idx + param_size

                # If this chunk overlaps with this parameter
                if i < end_idx and i + chunk_size > start_idx:
                    chunk_mapping[param_name].append(len(all_chunks) - 1)

        # Function to process each chunk and generate a proof
        def process_chunk(idx, chunk):
            try:
                # Compute L2 norm of chunk
                sum_sq = sum(g * g for g in chunk)

                # Count non-zero elements
                nz_count = sum(1 for g in chunk if g != 0)

                # Check constraints
                if sum_sq > max_norm:
                    return idx, {
                        "proof": None,
                        "public_inputs": None,
                        "metadata": {
                            "norm": sum_sq,
                            "nonzero_count": nz_count,
                            "error": f"Norm {sum_sq:.2f} exceeds maximum {max_norm}",
                            "chunk_index": idx
                        }
                    }

                if nz_count < min_active:
                    return idx, {
                        "proof": None,
                        "public_inputs": None,
                        "metadata": {
                            "norm": sum_sq,
                            "nonzero_count": nz_count,
                            "error": f"Only {nz_count} non-zero elements, minimum required is {min_active}",
                            "chunk_index": idx
                        }
                    }

                # Create a temporary directory for this chunk
                with tempfile.TemporaryDirectory() as temp_dir:
                    temp_path = Path(temp_dir)

                    # Create witness input file
                    input_json = {"gradients": chunk}
                    input_path = temp_path / "input.json"
                    with open(input_path, "w") as f:
                        json.dump(input_json, f)

                    witness_path = temp_path / "witness.wtns"
                    proof_path = temp_path / "proof.json"
                    public_path = temp_path / "public.json"

                    # Calculate witness (with constraints)
                    subprocess.run([
                        "snarkjs",
                        "wtns",
                        "calculate",
                        self.wasm_path,
                        str(input_path),
                        str(witness_path)
                    ], check=True, capture_output=True)

                    # Generate proof
                    subprocess.run([
                        "snarkjs",
                        "groth16",
                        "prove",
                        self.zkey_path,
                        str(witness_path),
                        str(proof_path),
                        str(public_path)
                    ], check=True, capture_output=True)

                    # Load the proof and public inputs
                    with open(proof_path, "r") as f:
                        proof = json.load(f)

                    with open(public_path, "r") as f:
                        public_inputs = json.load(f)

                # Return successful result
                return idx, {
                    "proof": proof,
                    "public_inputs": public_inputs,
                    "metadata": {
                        "norm": sum_sq,
                        "nonzero_count": nz_count,
                        "chunk_index": idx
                    }
                }
            except Exception as e:
                return idx, {
                    "proof": None,
                    "public_inputs": None,
                    "metadata": {
                        "norm": sum_sq,
                        "nonzero_count": nz_count,
                        "error": str(e),
                        "chunk_index": idx
                    }
                }

        # Generate proofs in parallel
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_idx = {
                executor.submit(process_chunk, idx, chunk): idx
                for idx, chunk in enumerate(all_chunks)
            }

            results = [None] * len(all_chunks)
            errors = []

            for future in as_completed(future_to_idx):
                idx, result = future.result()
                results[idx] = result

                # Collect any errors
                if result.get("metadata", {}).get("error"):
                    errors.append(result["metadata"]["error"])

        # Raise exception if any chunks had constraint errors
        if errors:
            raise ValueError(f"Batch proof generation failed with {len(errors)} constraint violations:\n" +
                            "\n".join(errors[:5]) +
                            (f"\n... and {len(errors) - 5} more errors" if len(errors) > 5 else ""))

        # Create Merkle tree of proof hashes for verification
        proof_hashes = []
        for result in results:
            if result["proof"] is not None:
                proof_str = json.dumps(result["proof"], sort_keys=True)
                proof_hash = hashlib.sha256(proof_str.encode()).hexdigest()
                proof_hashes.append(proof_hash)
            else:
                # For empty chunks, use a special hash
                proof_hashes.append(hashlib.sha256(b"empty_chunk").hexdigest())

        # Calculate Merkle root (simple implementation)
        def calculate_merkle_root(hashes):
            if len(hashes) == 0:
                return hashlib.sha256(b"empty").hexdigest()
            if len(hashes) == 1:
                return hashes[0]

            next_level = []
            for i in range(0, len(hashes), 2):
                if i + 1 < len(hashes):
                    combined = hashes[i] + hashes[i + 1]
                    next_hash = hashlib.sha256(combined.encode()).hexdigest()
                else:
                    # Odd number of hashes, duplicate the last one
                    next_hash = hashlib.sha256((hashes[i] + hashes[i]).encode()).hexdigest()
                next_level.append(next_hash)

            return calculate_merkle_root(next_level)

        merkle_root = calculate_merkle_root(proof_hashes)

        # Prepare the final batch result
        batch_result = {
            "batch_proofs": results,
            "metadata": {
                "total_chunks": len(all_chunks),
                "chunk_size": chunk_size,
                "max_norm_per_chunk": max_norm,
                "min_active_per_chunk": min_active,
                "param_mapping": dict(chunk_mapping),  # Convert defaultdict to regular dict
                "total_constraints_validated": len(proof_hashes)
            },
            "merkle_root": merkle_root
        }

        return batch_result

    def _generate_real_proof_helper(self, gradient_dict, input_path, witness_path, proof_path, public_path):
        # Get directory of circuit WASM file
        circuit_dir = os.path.dirname(self.wasm_path)
        circuit_name = os.path.splitext(os.path.basename(self.wasm_path))[0]

        # Calculate witness
        subprocess.run([
            "snarkjs",
            "wtns",
            "calculate",
            self.wasm_path,  # WASM file path
            str(input_path),
            str(witness_path)
        ], check=True)

        # Generate proof
        subprocess.run([
            "snarkjs",
            "groth16",
            "prove",
            self.zkey_path,  # Proving key path
            str(witness_path),
            str(proof_path),
            str(public_path)
        ], check=True)

        # Load the proof and public inputs
        with open(proof_path, "r") as f:
            proof = json.load(f)

        with open(public_path, "r") as f:
            public_inputs = json.load(f)

        # Create metadata about the gradients for easier verification
        param_info = []
        for param_name, gradient in gradient_dict.items():
            param_info.append({
                "param_name": param_name,
                "shape": list(gradient.shape),
                "norm": float(torch.norm(gradient).item()),
            })

        # Combine proof with metadata
        return proof, param_info

class ZKVerifier:
    """
    Production Zero-Knowledge Proof Verifier for FEDzk.

    This class verifies real zero-knowledge proofs using SNARKjs.
    It requires a complete ZK toolchain installation (run scripts/setup_zk.sh).
    """

    def __init__(self, verification_key_path: str = None):
        """
        Initialize with path to verification key.

        Args:
            verification_key_path: Path to the verification key for the ZK circuit
                                  If None, uses default paths based on secure mode
        """
        if verification_key_path:
            self.vkey_path = verification_key_path
        else:
            self.vkey_path = str(ASSET_DIR / "verification_key.json")
            self.secure_vkey_path = str(ASSET_DIR / "verification_key_secure.json")

        # Verify ZK infrastructure is available
        self._verify_zk_setup()

    def _verify_zk_setup(self):
        """Verify that the ZK infrastructure is properly set up."""
        # Strict verification - no test mode bypasses allowed
        required_tools = ["snarkjs"]
        missing_tools = []

        try:
            # snarkjs returns exit code 99 for --help, which is normal
            result = subprocess.run(["snarkjs", "--help"],
                                  capture_output=True, check=False)
            # Accept both exit code 0 and 99 as success for snarkjs
            if result.returncode not in [0, 99]:
                raise subprocess.CalledProcessError(result.returncode, ["snarkjs", "--help"])
        except (subprocess.CalledProcessError, FileNotFoundError):
            missing_tools.append("snarkjs")

        if missing_tools:
            raise RuntimeError(
                f"Missing ZK tools: {missing_tools}. "
                f"Please run 'scripts/setup_zk.sh' to install the complete ZK toolchain."
            )

        # Check verification key exists
        if not pathlib.Path(self.vkey_path).exists():
            raise RuntimeError(
                f"Verification key not found at {self.vkey_path}. "
                f"Please run 'scripts/setup_zk.sh' to generate circuit artifacts."
            )

        # Verify file integrity
        try:
            file_size = pathlib.Path(self.vkey_path).stat().st_size
            if file_size == 0:
                raise RuntimeError(f"Verification key file is empty/corrupted")
        except OSError as e:
            raise RuntimeError(f"Cannot access verification key file: {e}")

    def verify_proof(self, proof: Dict, public_inputs: List) -> bool:
        """
        Verify a zero-knowledge proof using SNARKjs.

        Args:
            proof: The proof dictionary generated by ZKProver
            public_inputs: List of public signals/inputs for the proof

        Returns:
            Boolean indicating whether the proof is valid
        """
        # Strict verification - no mock or fallback behavior
        with tempfile.TemporaryDirectory() as tmpdir:
            proof_path = os.path.join(tmpdir, "proof.json")
            public_inputs_path = os.path.join(tmpdir, "public.json")

            with open(proof_path, "w") as f:
                json.dump(proof, f)
            with open(public_inputs_path, "w") as f:
                json.dump(public_inputs, f)

            # Verify the proof
            try:
                result = subprocess.run(["snarkjs", "groth16", "verify",
                                         self.vkey_path,
                                         public_inputs_path,
                                         proof_path], capture_output=True, text=True, check=True)

                return "OK!" in result.stdout
            except subprocess.CalledProcessError as e:
                # Verification failed
                raise RuntimeError(f"Proof verification failed: {e.stderr}. "
                                 f"Please ensure the proof and verification key are valid.")

    def verify_real_proof(self, proof: Dict[str, Any], public_inputs: List[str]) -> bool:
        """
        Verify a real zero-knowledge proof using SNARKjs.

        Args:
            proof: Dictionary containing the ZK proof
            public_inputs: List of public inputs/signals

        Returns:
            Boolean indicating whether the proof is valid
        """
        return self.verify_proof(proof, public_inputs)

    def verify_secure_proof(self, proof: Dict[str, Any], public_inputs: List[str],
                          secure_vkey_path: str) -> bool:
        """
        Verify a secure zero-knowledge proof using the secure circuit verification key.

        Args:
            proof: Dictionary containing the ZK proof
            public_inputs: List of public inputs/signals
            secure_vkey_path: Path to secure circuit verification key

        Returns:
            Boolean indicating whether the proof is valid
        """
        # Check if secure verification key exists
        if not pathlib.Path(secure_vkey_path).exists():
            raise RuntimeError(
                f"Secure verification key not found at {secure_vkey_path}. "
                f"Please run 'scripts/setup_zk.sh' to generate circuit artifacts."
            )

        with tempfile.TemporaryDirectory() as tmpdir:
            proof_path = os.path.join(tmpdir, "proof.json")
            public_inputs_path = os.path.join(tmpdir, "public.json")

            with open(proof_path, "w") as f:
                json.dump(proof, f)
            with open(public_inputs_path, "w") as f:
                json.dump(public_inputs, f)

            # Verify the proof with secure key
            try:
                result = subprocess.run(["snarkjs", "groth16", "verify",
                                         secure_vkey_path,
                                         public_inputs_path,
                                         proof_path], capture_output=True, text=True, check=True)

                return "OK!" in result.stdout
            except subprocess.CalledProcessError as e:
                # Verification failed
                raise RuntimeError(f"Secure proof verification failed: {e.stderr}. "
                                 f"Please ensure the proof and secure verification key are valid.")

# Utility functions (if any were previously in this file and used by the classes, keep them)
# Example: _flatten_gradients, etc.
#// ... existing code ... from line 279 to 507
