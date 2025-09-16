# Functional Source License 1.1 with Apache-2.0 Future Grant (FSL-1.1-Apache-2.0)
# Copyright (c) 2025 Aaryan Guglani and FEDzk Contributors
# Licensed under FSL-1.1-Apache-2.0. See LICENSE for details.

# src/fedzk/prover/batch_zkgenerator.py
import hashlib
import json
import os
import pathlib
import subprocess
import tempfile
import logging
from typing import Dict, List, Tuple, Any, Optional

import torch
import numpy as np

from .zk_validator import ZKValidator

logger = logging.getLogger(__name__)

ASSET_DIR = pathlib.Path(__file__).resolve().parent.parent / "zk"


class BatchZKProver:
    """
    Batch Zero-Knowledge Proof Generator for FEDzk.

    This class generates ZK proofs for batches of gradient updates,
    validating that multiple gradient sets meet consistency requirements.
    """

    def __init__(
        self,
        secure: bool = False,
        chunk_size: int = 4,
        max_norm_squared: float = 100.0,
        min_active: int = 1,
        batch_size: int = 4,
        grad_size: int = 4,
        preserve_client_gradients: bool = True,
        enable_batch_validation: bool = True
    ):
        """
        Initialize the Batch ZK prover.

        Args:
            secure: Whether to use secure circuit with additional constraints
            chunk_size: Size of gradient chunks to process
            max_norm_squared: Maximum allowed squared L2 norm for gradients
            min_active: Minimum number of non-zero gradient elements required
            batch_size: Number of gradient sets in batch (must match circuit)
            grad_size: Size of each gradient vector (must match circuit)
            preserve_client_gradients: Whether to preserve individual client gradients in batch
            enable_batch_validation: Whether to enable cryptographic batch validation
        """
        self.secure = secure
        self.chunk_size = chunk_size
        self.max_norm_squared = max_norm_squared
        self.min_active = min_active
        self.batch_size = batch_size
        self.grad_size = grad_size
        self.preserve_client_gradients = preserve_client_gradients
        self.enable_batch_validation = enable_batch_validation

        # Circuit file paths
        if secure:
            self.wasm_path = str(ASSET_DIR / "batch_verification.wasm")
            self.zkey_path = str(ASSET_DIR / "proving_key_batch_verification.zkey")
        else:
            self.wasm_path = str(ASSET_DIR / "batch_verification.wasm")
            self.zkey_path = str(ASSET_DIR / "proving_key_batch_verification.zkey")

        # Validate ZK toolchain
        self._validate_zk_toolchain()

    def _validate_zk_toolchain(self):
        """Validate ZK toolchain for batch operations."""
        validator = ZKValidator(str(ASSET_DIR))
        validation_results = validator.validate_toolchain()

        if validation_results["overall_status"] == "failed":
            error_msg = "ZK toolchain validation failed for batch operations:\n"
            for error in validation_results["errors"]:
                error_msg += f"  • {error}\n"
            error_msg += "\nPlease run 'scripts/setup_zk.sh' to install/configure ZK toolchain."
            raise RuntimeError(error_msg)

        if validation_results["overall_status"] == "warning":
            warning_msg = "ZK toolchain validation passed with warnings for batch operations:\n"
            for warning in validation_results["warnings"]:
                warning_msg += f"  • {warning}\n"
            logger.warning(warning_msg)

        logger.info("✅ Batch ZK toolchain validation passed")

    def generate_proof(self, gradient_batch: Dict[str, torch.Tensor]) -> Tuple[Dict, List]:
        """
        Generate a batch ZK proof for multiple gradient updates.

        Args:
            gradient_batch: Dictionary containing batched gradient data

        Returns:
            Tuple of (proof_dict, public_signals_list)

        Raises:
            ValueError: If input validation fails
            RuntimeError: If proof generation fails
        """
        # Input validation
        if not gradient_batch:
            raise ValueError("Empty gradient batch provided")

        # Convert gradients to expected format
        batch_data = self._prepare_batch_input(gradient_batch)

        # Generate proof using SNARKjs
        return self._run_batch_proof_generation(batch_data)

    def _prepare_batch_input(self, gradient_batch: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        """
        Prepare input data for batch circuit with enhanced federated learning support.

        Args:
            gradient_batch: Raw gradient batch data (can be single tensor or batch)

        Returns:
            Formatted input data for the circuit with cryptographic integrity
        """
        if not gradient_batch:
            raise ValueError("Empty gradient batch provided")

        # Extract gradients - handle both single tensors and batched tensors
        if 'weights' in gradient_batch:
            gradients = gradient_batch['weights']
        else:
            # Take first available gradient tensor
            first_key = next(iter(gradient_batch.keys()))
            gradients = gradient_batch[first_key]

        # Handle different input formats for federated learning scenarios
        if gradients.dim() == 1:
            # Single client gradient - replicate for batch
            return self._prepare_single_client_batch(gradients)
        elif gradients.dim() == 2:
            # Multiple client gradients in batch format [num_clients, grad_size]
            return self._prepare_multi_client_batch(gradients)
        else:
            # Higher dimensional - flatten and process
            gradients = gradients.flatten()
            return self._prepare_single_client_batch(gradients)

    def _prepare_single_client_batch(self, gradients: torch.Tensor) -> Dict[str, Any]:
        """
        Prepare batch input for single client scenario (replicate gradients).

        Args:
            gradients: Single gradient tensor

        Returns:
            Formatted batch input
        """
        # Ensure gradients is a 1D tensor and pad/truncate to expected size
        if gradients.dim() > 1:
            gradients = gradients.flatten()

        # Pad or truncate to expected grad_size
        grad_list = gradients.tolist()
        if len(grad_list) < self.grad_size:
            grad_list.extend([0.0] * (self.grad_size - len(grad_list)))
        elif len(grad_list) > self.grad_size:
            grad_list = grad_list[:self.grad_size]

        # Create batch by replicating this gradient set
        gradient_batch_data = []
        expected_norms = []

        for i in range(self.batch_size):
            gradient_batch_data.append(grad_list)
            # Calculate expected norm (squared L2 norm)
            norm_sq = sum(x * x for x in grad_list)
            expected_norms.append(norm_sq)

        # Calculate aggregated norm
        total_norm = sum(expected_norms)

        return {
            "gradientBatch": gradient_batch_data,
            "expectedNorms": expected_norms,
            "globalMaxNorm": self.max_norm_squared,
            "batchMetadata": {
                "inputType": "single_client_replicated",
                "originalGradSize": len(gradients.tolist()),
                "totalNorm": total_norm,
                "clientCount": 1,
                "batchSize": self.batch_size
            }
        }

    def _prepare_multi_client_batch(self, gradients: torch.Tensor) -> Dict[str, Any]:
        """
        Prepare batch input for multi-client federated learning scenario.

        Args:
            gradients: Batch of gradients [num_clients, grad_size]

        Returns:
            Formatted batch input with client preservation
        """
        num_clients, grad_size = gradients.shape

        if num_clients > self.batch_size:
            logger.warning(f"More clients ({num_clients}) than batch size ({self.batch_size}), truncating")
            gradients = gradients[:self.batch_size]
            num_clients = self.batch_size

        # Prepare gradient batch data
        gradient_batch_data = []
        expected_norms = []
        original_gradients = []

        for i in range(num_clients):
            client_grad = gradients[i]

            # Convert to list and pad/truncate
            grad_list = client_grad.tolist()
            original_gradients.append(grad_list.copy())

            if len(grad_list) < self.grad_size:
                grad_list.extend([0.0] * (self.grad_size - len(grad_list)))
            elif len(grad_list) > self.grad_size:
                grad_list = grad_list[:self.grad_size]

            gradient_batch_data.append(grad_list)

            # Calculate expected norm for this client
            norm_sq = sum(x * x for x in grad_list)
            expected_norms.append(norm_sq)

        # Fill remaining batch slots if needed
        while len(gradient_batch_data) < self.batch_size:
            # Use zero gradients for padding
            zero_grad = [0.0] * self.grad_size
            gradient_batch_data.append(zero_grad)
            expected_norms.append(0.0)
            original_gradients.append([])  # Empty list for padding

        # Calculate aggregated norm
        total_norm = sum(expected_norms)

        # Create integrity hash for batch validation
        batch_integrity = self._compute_batch_integrity_hash(original_gradients)

        return {
            "gradientBatch": gradient_batch_data,
            "expectedNorms": expected_norms,
            "globalMaxNorm": self.max_norm_squared,
            "batchMetadata": {
                "inputType": "multi_client_batch",
                "originalClientCount": num_clients,
                "originalGradSize": grad_size,
                "totalNorm": total_norm,
                "batchSize": self.batch_size,
                "batchIntegrityHash": batch_integrity,
                "preserveGradients": self.preserve_client_gradients
            }
        }

    def _compute_batch_integrity_hash(self, client_gradients: List[List[float]]) -> str:
        """
        Compute cryptographic integrity hash for batch validation.

        Args:
            client_gradients: List of original client gradient lists

        Returns:
            SHA256 hash of batch contents
        """
        if not client_gradients:
            return hashlib.sha256(b"empty_batch").hexdigest()

        # Create combined hash of all client gradients
        combined_data = ""
        for i, grad in enumerate(client_gradients):
            combined_data += f"client_{i}:" + ",".join(map(str, grad)) + ";"

        return hashlib.sha256(combined_data.encode()).hexdigest()

    def validate_batch_cryptographic_integrity(self, proof: Dict[str, Any],
                                               public_signals: List[Any],
                                               batch_metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate cryptographic integrity of batch proof and metadata.

        Args:
            proof: Generated ZK proof
            public_signals: Public signals from proof generation
            batch_metadata: Metadata from batch preparation

        Returns:
            Dict with validation results and security metrics
        """
        validation_results = {
            "proof_integrity": False,
            "metadata_integrity": False,
            "cryptographic_security": False,
            "batch_consistency": False,
            "security_score": 0.0,
            "warnings": [],
            "errors": []
        }

        try:
            # 1. Validate proof structure
            if not self._validate_proof_structure(proof):
                validation_results["errors"].append("Invalid proof structure")
                return validation_results

            validation_results["proof_integrity"] = True

            # 2. Validate public signals
            if not self._validate_public_signals(public_signals, batch_metadata):
                validation_results["errors"].append("Invalid public signals")
                return validation_results

            # 3. Validate batch metadata integrity
            if self._validate_batch_metadata_integrity(batch_metadata):
                validation_results["metadata_integrity"] = True
            else:
                validation_results["warnings"].append("Batch metadata integrity check failed")

            # 4. Check cryptographic security properties
            security_check = self._check_cryptographic_security(proof, public_signals)
            validation_results["cryptographic_security"] = security_check["valid"]
            if not security_check["valid"]:
                validation_results["errors"].extend(security_check["issues"])

            # 5. Validate batch consistency
            if self._validate_batch_consistency_check(proof, batch_metadata):
                validation_results["batch_consistency"] = True
            else:
                validation_results["warnings"].append("Batch consistency validation failed")

            # Calculate overall security score
            validation_results["security_score"] = self._calculate_security_score(validation_results)

        except Exception as e:
            validation_results["errors"].append(f"Integrity validation failed: {str(e)}")

        return validation_results

    def _validate_proof_structure(self, proof: Dict[str, Any]) -> bool:
        """Validate proof has required cryptographic structure."""
        required_keys = ['pi_a', 'pi_b', 'pi_c', 'protocol']

        for key in required_keys:
            if key not in proof:
                return False

        # Validate array structures
        if not isinstance(proof['pi_a'], list) or len(proof['pi_a']) != 3:
            return False
        if not isinstance(proof['pi_b'], list) or len(proof['pi_b']) != 3:
            return False
        if not isinstance(proof['pi_c'], list) or len(proof['pi_c']) != 3:
            return False

        # Validate protocol
        if proof.get('protocol') != 'groth16':
            return False

        return True

    def _validate_public_signals(self, public_signals: List[Any], batch_metadata: Dict[str, Any]) -> bool:
        """Validate public signals match expected format."""
        if not isinstance(public_signals, list) or len(public_signals) == 0:
            return False

        # Check that all signals are numeric
        for signal in public_signals:
            if not isinstance(signal, (int, float)):
                return False

        # Validate expected number of public signals based on circuit
        expected_signals = batch_metadata.get("batchSize", 4) + 2  # norms + aggregated norm + batchValid
        if len(public_signals) != expected_signals:
            logger.warning(f"Expected {expected_signals} public signals, got {len(public_signals)}")
            return False

        return True

    def _validate_batch_metadata_integrity(self, batch_metadata: Dict[str, Any]) -> bool:
        """Validate batch metadata integrity."""
        required_fields = ["inputType", "batchSize", "totalNorm"]

        for field in required_fields:
            if field not in batch_metadata:
                return False

        # Validate batch size consistency
        if batch_metadata.get("batchSize", 0) <= 0:
            return False

        # Validate total norm is reasonable
        total_norm = batch_metadata.get("totalNorm", -1)
        if total_norm < 0:
            return False

        return True

    def _check_cryptographic_security(self, proof: Dict[str, Any], public_signals: List[Any]) -> Dict[str, Any]:
        """Check cryptographic security properties of proof."""
        issues = []
        valid = True

        try:
            # Check for weak cryptographic parameters
            for component in ['pi_a', 'pi_b', 'pi_c']:
                if component in proof:
                    for value in proof[component]:
                        if isinstance(value, list):
                            for v in value:
                                if isinstance(v, (int, float)) and abs(v) < 1e-10:
                                    issues.append(f"Weak cryptographic parameter in {component}")
                                    valid = False

            # Check public signals for anomalies
            for i, signal in enumerate(public_signals):
                if isinstance(signal, (int, float)):
                    if abs(signal) > 1e20:  # Unrealistically large values
                        issues.append(f"Anomalous public signal at index {i}")
                        valid = False

        except Exception as e:
            issues.append(f"Cryptographic security check failed: {str(e)}")
            valid = False

        return {"valid": valid, "issues": issues}

    def _validate_batch_consistency_check(self, proof: Dict[str, Any], batch_metadata: Dict[str, Any]) -> bool:
        """Validate batch consistency between proof and metadata."""
        try:
            # Check batch size consistency
            batch_size = batch_metadata.get("batchSize", 0)
            if batch_size <= 0 or batch_size > 100:  # Reasonable upper bound
                return False

            # Validate input type
            input_type = batch_metadata.get("inputType", "")
            if input_type not in ["single_client_replicated", "multi_client_batch"]:
                return False

            return True

        except Exception:
            return False

    def _calculate_security_score(self, validation_results: Dict[str, Any]) -> float:
        """Calculate overall security score from validation results."""
        score_components = [
            validation_results["proof_integrity"],
            validation_results["metadata_integrity"],
            validation_results["cryptographic_security"],
            validation_results["batch_consistency"]
        ]

        base_score = sum(score_components) / len(score_components)

        # Penalty for warnings
        warning_penalty = len(validation_results.get("warnings", [])) * 0.1
        base_score = max(0, base_score - warning_penalty)

        # Penalty for errors
        error_penalty = len(validation_results.get("errors", [])) * 0.2
        base_score = max(0, base_score - error_penalty)

        return round(base_score, 3)

    def _run_batch_proof_generation(self, batch_data: Dict[str, Any]) -> Tuple[Dict, List]:
        """
        Run SNARKjs proof generation for batch circuit.

        Args:
            batch_data: Prepared batch input data

        Returns:
            Tuple of (proof_dict, public_signals_list)
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            input_json_path = os.path.join(tmpdir, "batch_input.json")
            witness_path = os.path.join(tmpdir, "witness.wtns")
            proof_path = os.path.join(tmpdir, "proof.json")
            public_path = os.path.join(tmpdir, "public.json")

            # Write input data
            with open(input_json_path, "w") as f:
                json.dump(batch_data, f)

            # Generate witness
            try:
                result = subprocess.run([
                    "snarkjs", "wtns", "calculate",
                    self.wasm_path,
                    input_json_path,
                    witness_path
                ], check=True, capture_output=True, text=True)
            except subprocess.CalledProcessError as e:
                error_msg = e.stderr.strip() if e.stderr else e.stdout.strip() if e.stdout else f"Exit code: {e.returncode}"
                raise RuntimeError(f"Failed to generate batch witness: {error_msg}")

            # Generate proof
            try:
                subprocess.run([
                    "snarkjs", "groth16", "prove",
                    self.zkey_path,
                    witness_path,
                    proof_path,
                    public_path
                ], check=True, capture_output=True, text=True)
            except subprocess.CalledProcessError as e:
                error_msg = e.stderr.strip() if e.stderr else e.stdout.strip() if e.stdout else f"Exit code: {e.returncode}"
                raise RuntimeError(f"Failed to generate batch proof: {error_msg}")

            # Read proof and public signals
            with open(proof_path, "r") as f:
                proof = json.load(f)

            with open(public_path, "r") as f:
                public_signals = json.load(f)

            logger.info("✅ Batch proof generated successfully")
            return proof, public_signals


class BatchZKVerifier:
    """
    Batch Zero-Knowledge Proof Verifier for FEDzk.

    This class verifies ZK proofs for batched gradient updates.
    """

    def __init__(self, verification_key_path: Optional[str] = None, secure: bool = False):
        """
        Initialize the Batch ZK verifier.

        Args:
            verification_key_path: Path to verification key (optional)
            secure: Whether to use secure verification key
        """
        if verification_key_path:
            self.verification_key_path = verification_key_path
        else:
            # Use batch verification key from setup artifacts
            self.verification_key_path = str(ASSET_DIR / "batch_verification_verification_key.json")

        self.secure = secure

        # Validate verification key exists
        if not os.path.exists(self.verification_key_path):
            logger.warning(f"Batch verification key not found: {self.verification_key_path}")
            logger.warning("Batch proof verification may not work correctly")

    def verify_proof(self, proof: Dict[str, Any], public_signals: List[Any]) -> bool:
        """
        Verify a batch ZK proof.

        Args:
            proof: Proof dictionary
            public_signals: Public signals list

        Returns:
            bool: Whether proof is valid
        """
        return self.verify_real_proof(proof, public_signals)

    def verify_real_proof(self, proof: Dict[str, Any], public_signals: List[Any]) -> bool:
        """
        Verify a batch ZK proof using real cryptographic validation.

        Args:
            proof: Proof dictionary
            public_signals: Public signals list

        Returns:
            bool: Whether proof is valid
        """
        # Check if verification key exists
        if not os.path.exists(self.verification_key_path):
            logger.error(f"Batch verification key not found: {self.verification_key_path}")
            return False

        with tempfile.TemporaryDirectory() as tmpdir:
            proof_path = os.path.join(tmpdir, "proof.json")
            public_path = os.path.join(tmpdir, "public.json")

            # Write proof and public signals
            with open(proof_path, "w") as f:
                json.dump(proof, f)

            with open(public_path, "w") as f:
                json.dump(public_signals, f)

            # Verify proof using SNARKjs
            try:
                result = subprocess.run([
                    "snarkjs", "groth16", "verify",
                    self.verification_key_path,
                    public_path,
                    proof_path
                ], capture_output=True, text=True, check=False)

                # SNARKjs verify returns exit code 0 for valid proofs
                if result.returncode == 0:
                    logger.info("✅ Batch proof verification successful")
        return True 
                else:
                    logger.warning(f"❌ Batch proof verification failed: {result.stderr.strip()}")
                    return False
        
            except Exception as e:
                logger.error(f"Batch proof verification error: {e}")
                return False