# Copyright (c) 2025 Aaryan Guglani and FedZK Contributors
# SPDX-License-Identifier: MIT

"""
Zero-Knowledge Proof Generator for FedZK.

This module contains the ZKProver class which handles generation of zero-knowledge
proofs for gradient updates in federated learning.
"""

import hashlib
import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional, Union

import numpy as np
import torch
import math

# Add these imports
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict


class ZKProver:
    """
    Generates zero-knowledge proofs for model updates in federated learning.
    
    This class can generate either dummy proofs (for testing) or real zero-knowledge
    proofs using snarkjs and Circom circuits.
    """
    
    def __init__(self, circuit_path: str, proving_key_path: str):
        """
        Initialize a ZKProver with paths to ZK circuit and proving key.
        
        Args:
            circuit_path: Path to the ZK circuit definition or WASM file
            proving_key_path: Path to the proving key for the circuit
        """
        self.circuit_path = circuit_path
        self.proving_key_path = proving_key_path
        
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
    
    def generate_proof(self, gradient_dict: Dict[str, torch.Tensor]) -> Tuple[str, List[Any]]:
        """
        Simulates generating a zero-knowledge proof from a model update.
        
        This method:
        1. Flattens and hashes gradient tensors
        2. Computes norms for public signals
        3. Generates a dummy proof (to be replaced with real ZK proof later)
        
        Args:
            gradient_dict: Dictionary mapping parameter names to gradient tensors
            
        Returns:
            Tuple of (proof_string, public_signals_list) where:
            - proof_string is a placeholder hash representing the proof
            - public_signals_list contains metadata about the gradients
        """
        # Generate parameter hashes and norms
        param_hashes = {}
        param_norms = {}
        
        for param_name, gradient in gradient_dict.items():
            param_hashes[param_name] = self._hash_tensor(gradient)
            param_norms[param_name] = self._compute_tensor_norm(gradient)
        
        # Create public signals (this would be actual public inputs to the ZK proof)
        public_signals = []
        for param_name in sorted(gradient_dict.keys()):
            public_signals.append({
                "param_name": param_name,
                "norm": param_norms[param_name],
                "hash_prefix": param_hashes[param_name][:8]  # First 8 chars of hash
            })
        
        # Dummy proof generation (in a real implementation, this would call snarkjs or similar)
        # Concatenate all hashes and hash them to simulate a proof
        all_hashes = "".join(param_hashes.values())
        proof_hash = hashlib.sha256(all_hashes.encode()).hexdigest()
        
        # Mock proof object (would be replaced with actual ZK proof)
        proof = f"dummy_proof_{proof_hash[:16]}"
        
        return proof, public_signals
    
    def generate_real_proof(self, gradient_dict: Dict[str, torch.Tensor], 
                           max_inputs: int = 4) -> Tuple[Dict[str, Any], List[Any]]:
        """
        Generate a real zero-knowledge proof using snarkjs and Circom.
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            # Flatten and pad gradients
            all_grads = []
            for gradient in gradient_dict.values():
                all_grads.extend(gradient.flatten().tolist())
            selected_grads = all_grads[:max_inputs]
            selected_grads += [0] * (max_inputs - len(selected_grads))
            # Create witness input
            input_json = {"gradients": selected_grads}
            input_path = temp_path / "input.json"
            with open(input_path, 'w') as f:
                json.dump(input_json, f)
            witness_path = temp_path / "witness.wtns"
            proof_path = temp_path / "proof.json"
            public_path = temp_path / "public.json"
            # Generate witness and proof
            subprocess.run(["snarkjs", "wtns", "calculate", self.circuit_path, str(input_path), str(witness_path)], check=True)
            subprocess.run(["snarkjs", "groth16", "prove", self.proving_key_path, str(witness_path), str(proof_path), str(public_path)], check=True)
            proof = json.load(open(proof_path))
            public_inputs = json.load(open(public_path))
            # Return proof and public inputs
            return proof, public_inputs
    
    def generate_real_proof_secure(self, gradient_dict: Dict[str, torch.Tensor], 
                                  max_inputs: int = 4,
                                  max_norm: float = 100.0,
                                  min_active: int = 3) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Generate a real zero-knowledge proof for secure gradients using snarkjs and Circom.
        
        Enforces the constraints:
        - L2 norm of gradients must be <= max_norm
        - At least min_active elements must be non-zero
        
        Args:
            gradient_dict: Dictionary mapping parameter names to gradient tensors
            max_inputs: Maximum number of gradient elements to include
            max_norm: Maximum allowed L2 norm for the gradients
            min_active: Minimum number of non-zero gradient elements required
            
        Returns:
            Tuple of (proof, public_inputs) where:
            - proof is the zero-knowledge proof from snarkjs
            - public_inputs contains the public inputs and metadata
            
        Raises:
            ValueError: If the gradients do not satisfy the constraints
        """
        # Flatten and pad gradients
        flat = []
        for gradient in gradient_dict.values():
            flat.extend(gradient.flatten().tolist())
        selected = (flat[:max_inputs] + [0]*max_inputs)[:max_inputs]
        
        # Compute norm and non-zero count
        sum_sq = sum(x*x for x in selected)
        norm_val = sum_sq  # Use sum of squares directly, not square root
        nz_count = sum(1 for x in selected if x != 0)
        
        # Enforce constraints
        if norm_val > max_norm:
            raise ValueError(f"Gradient norm {norm_val:.2f} exceeds maximum allowed value {max_norm}")
        if nz_count < min_active:
            raise ValueError(f"Only {nz_count} non-zero gradients, minimum required {min_active}")
        
        # Prepare input for the secure circuit
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create input JSON with secure circuit parameters
            input_json = {
                "gradients": selected,
                "maxNorm": float(max_norm),
                "minNonZero": int(min_active)
            }
            
            input_path = temp_path / "input.json"
            with open(input_path, 'w') as f:
                json.dump(input_json, f)
                
            witness_path = temp_path / "witness.wtns"
            proof_path = temp_path / "proof.json"
            public_path = temp_path / "public.json"
            
            # Generate witness and proof
            subprocess.run(["snarkjs", "wtns", "calculate", 
                           self.circuit_path, str(input_path), str(witness_path)], 
                           check=True)
            
            subprocess.run(["snarkjs", "groth16", "prove", 
                           self.proving_key_path, str(witness_path), 
                           str(proof_path), str(public_path)], 
                           check=True)
            
            # Load proof and public inputs
            with open(proof_path, 'r') as f:
                proof = json.load(f)
            with open(public_path, 'r') as f:
                public_inputs = json.load(f)
            
            # Return proof and public inputs with metadata
            return proof, {
                "public_inputs": public_inputs,
                "metadata": {
                    "norm": norm_val,
                    "nonzero_count": nz_count,
                    "max_norm": max_norm,
                    "min_active": min_active
                }
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
                    with open(input_path, 'w') as f:
                        json.dump(input_json, f)
                    
                    witness_path = temp_path / "witness.wtns"
                    proof_path = temp_path / "proof.json"
                    public_path = temp_path / "public.json"
                    
                    # Calculate witness (with constraints)
                    subprocess.run([
                        "snarkjs", 
                        "wtns", 
                        "calculate", 
                        self.circuit_path,
                        str(input_path), 
                        str(witness_path)
                    ], check=True, capture_output=True)
                    
                    # Generate proof
                    subprocess.run([
                        "snarkjs", 
                        "groth16", 
                        "prove", 
                        self.proving_key_path,
                        str(witness_path), 
                        str(proof_path), 
                        str(public_path)
                    ], check=True, capture_output=True)
                    
                    # Load the proof and public inputs
                    with open(proof_path, 'r') as f:
                        proof = json.load(f)
                    
                    with open(public_path, 'r') as f:
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
        circuit_dir = os.path.dirname(self.circuit_path)
        circuit_name = os.path.splitext(os.path.basename(self.circuit_path))[0]
        
        # Calculate witness
        subprocess.run([
            "snarkjs", 
            "wtns", 
            "calculate", 
            self.circuit_path,  # WASM file path
            str(input_path), 
            str(witness_path)
        ], check=True)
        
        # Generate proof
        subprocess.run([
            "snarkjs", 
            "groth16", 
            "prove", 
            self.proving_key_path,  # Proving key path
            str(witness_path), 
            str(proof_path), 
            str(public_path)
        ], check=True)
        
        # Load the proof and public inputs
        with open(proof_path, 'r') as f:
            proof = json.load(f)
        
        with open(public_path, 'r') as f:
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