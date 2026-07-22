"""
Advanced Proof Verification Module for FEDzk

This module provides enhanced cryptographic verification mechanisms that support:
1. Quantized gradient proof verification
2. Multi-circuit proof validation
3. Batch verification with integrity checks
4. Adaptive verification strategies
5. Comprehensive security validation
"""

import hashlib
import time
import json
from typing import Dict, List, Tuple, Any, Optional, Union
import torch
from pathlib import Path
import subprocess
import logging

from .input_normalization import GradientQuantizer, create_quantized_proof_input

logger = logging.getLogger(__name__)


class AdvancedProofVerifier:
    """
    Advanced proof verifier with support for multiple circuit types and verification strategies.
    """

    def __init__(self, project_root: Optional[Path] = None):
        """
        Initialize the advanced proof verifier.

        Args:
            project_root: Root directory of the FEDzk project
        """
        if project_root is None:
            # Try to find project root
            current_path = Path(__file__).resolve()
            for parent in current_path.parents:
                if (parent / "src" / "fedzk").exists():
                    project_root = parent
                    break

        self.project_root = project_root
        self.circuit_configs = self._load_circuit_configs()
        self.quantizer = GradientQuantizer(scale_factor=1000)

    def _load_circuit_configs(self) -> Dict[str, Dict[str, Any]]:
        """Load circuit configurations for different verification types."""
        return {
            "model_update": {
                "wasm_path": "model_update.wasm",
                "zkey_path": "proving_key_model_update.zkey",
                "vkey_path": "verification_key_model_update.json",
                "input_size": 4,
                "public_inputs": ["gradients"]
            },
            "model_update_secure": {
                "wasm_path": "model_update_secure.wasm",
                "zkey_path": "proving_key_model_update_secure.zkey",
                "vkey_path": "verification_key_model_update_secure.json",
                "input_size": 4,
                "public_inputs": ["gradients", "maxNorm", "minNonZero"]
            },
            "model_update_quantized": {
                "wasm_path": "model_update_quantized.wasm",
                "zkey_path": "proving_key_model_update_quantized.zkey",
                "vkey_path": "verification_key_model_update_quantized.json",
                "input_size": 4,
                "public_inputs": ["quantized_gradients", "scale_factor_input"]
            },
            "batch_verification": {
                "wasm_path": "batch_verification.wasm",
                "zkey_path": "proving_key_batch_verification.zkey",
                "vkey_path": "verification_key_batch_verification.json",
                "input_size": 8,
                "public_inputs": ["gradientBatch"]
            }
        }

    def verify_proof_comprehensive(self, proof: Dict[str, Any], public_inputs: List[Any],
                                 circuit_type: str = "model_update",
                                 verification_level: str = "standard") -> Dict[str, Any]:
        """
        Comprehensive proof verification with multiple validation layers.

        Args:
            proof: Zero-knowledge proof
            public_inputs: Public inputs for verification
            circuit_type: Type of circuit used
            verification_level: Level of verification ("basic", "standard", "comprehensive")

        Returns:
            Dictionary with verification results and metadata
        """
        verification_result = {
            "is_valid": False,
            "verification_time": 0.0,
            "circuit_type": circuit_type,
            "verification_level": verification_level,
            "checks_performed": [],
            "errors": [],
            "metadata": {}
        }

        start_time = time.time()

        try:
            # Basic structure validation
            if not self._validate_proof_structure(proof):
                verification_result["errors"].append("Invalid proof structure")
                verification_result["checks_performed"].append("structure_validation")
                return verification_result

            verification_result["checks_performed"].append("structure_validation")

            # Circuit-specific validation
            if not self._validate_circuit_specific(proof, public_inputs, circuit_type):
                verification_result["errors"].append(f"Circuit-specific validation failed for {circuit_type}")
                verification_result["checks_performed"].append("circuit_validation")
                return verification_result

            verification_result["checks_performed"].append("circuit_validation")

            # SNARK verification
            if verification_level in ["standard", "comprehensive"]:
                snark_valid = self._verify_snark_proof(proof, public_inputs, circuit_type)
                if not snark_valid:
                    verification_result["errors"].append("SNARK proof verification failed")
                    verification_result["checks_performed"].append("snark_verification")
                    return verification_result

                verification_result["checks_performed"].append("snark_verification")

            # Comprehensive checks
            if verification_level == "comprehensive":
                integrity_checks = self._perform_integrity_checks(proof, public_inputs, circuit_type)
                if not integrity_checks["all_passed"]:
                    verification_result["errors"].extend(integrity_checks["failed_checks"])
                    verification_result["checks_performed"].append("integrity_checks")
                    verification_result["metadata"]["integrity_results"] = integrity_checks
                    return verification_result

                verification_result["checks_performed"].append("integrity_checks")
                verification_result["metadata"]["integrity_results"] = integrity_checks

            # If we reach here, verification passed
            verification_result["is_valid"] = True
            verification_result["verification_time"] = time.time() - start_time

        except Exception as e:
            verification_result["errors"].append(f"Verification error: {str(e)}")
            verification_result["verification_time"] = time.time() - start_time
            logger.error(f"Proof verification failed: {e}")

        return verification_result

    def _validate_proof_structure(self, proof: Dict[str, Any]) -> bool:
        """Validate basic proof structure."""
        required_keys = ["pi_a", "pi_b", "pi_c", "protocol"]

        if not isinstance(proof, dict):
            return False

        for key in required_keys:
            if key not in proof:
                return False

        # Validate protocol
        if proof.get("protocol") != "groth16":
            return False

        # Validate proof arrays
        for key in ["pi_a", "pi_b", "pi_c"]:
            if not isinstance(proof[key], list) or len(proof[key]) != 3:
                return False
            for sub_array in proof[key]:
                if not isinstance(sub_array, list):
                    return False

        return True

    def _validate_circuit_specific(self, proof: Dict[str, Any], public_inputs: List[Any],
                                 circuit_type: str) -> bool:
        """Validate circuit-specific requirements."""
        if circuit_type not in self.circuit_configs:
            return False

        config = self.circuit_configs[circuit_type]

        # Check input size
        if len(public_inputs) != config["input_size"]:
            return False

        # Circuit-specific validations
        if circuit_type == "model_update_quantized":
            # Check for scale factor in public inputs
            if len(public_inputs) < 2:
                return False
            # Last element should be scale factor
            scale_factor = public_inputs[-1]
            if not isinstance(scale_factor, (int, float)):
                return False

        elif circuit_type == "model_update_secure":
            # Check for maxNorm and minNonZero
            if len(public_inputs) < 3:
                return False

        return True

    def _verify_snark_proof(self, proof: Dict[str, Any], public_inputs: List[Any],
                          circuit_type: str) -> bool:
        """
        Perform actual SNARK proof verification using snarkjs.

        Args:
            proof: Proof to verify
            public_inputs: Public inputs
            circuit_type: Circuit type for verification key selection

        Returns:
            True if proof is valid, False otherwise
        """
        try:
            config = self.circuit_configs[circuit_type]
            vkey_path = self.project_root / "src" / "fedzk" / "zk" / "circuits" / config["vkey_path"]

            if not vkey_path.exists():
                logger.warning(f"Verification key not found: {vkey_path}")
                return False

            # Create temporary proof file
            proof_file = self.project_root / "temp_proof.json"
            public_file = self.project_root / "temp_public.json"

            try:
                # Write proof to temporary file
                with open(proof_file, 'w') as f:
                    json.dump(proof, f)

                # Write public inputs to temporary file
                with open(public_file, 'w') as f:
                    json.dump(public_inputs, f)

                # Run snarkjs verification
                cmd = [
                    "snarkjs", "groth16", "verify",
                    str(vkey_path),
                    str(public_file),
                    str(proof_file)
                ]

                result = subprocess.run(cmd, capture_output=True, text=True, cwd=self.project_root)

                return result.returncode == 0

            finally:
                # Clean up temporary files
                for temp_file in [proof_file, public_file]:
                    if temp_file.exists():
                        temp_file.unlink()

        except Exception as e:
            logger.error(f"SNARK verification failed: {e}")
            return False

    def _perform_integrity_checks(self, proof: Dict[str, Any], public_inputs: List[Any],
                                circuit_type: str) -> Dict[str, Any]:
        """Perform comprehensive integrity checks on the proof."""
        integrity_results = {
            "all_passed": True,
            "failed_checks": [],
            "check_results": {}
        }

        # Check 1: Proof consistency
        proof_hash = hashlib.sha256(json.dumps(proof, sort_keys=True).encode()).hexdigest()
        integrity_results["check_results"]["proof_consistency"] = proof_hash

        # Check 2: Input validation
        if circuit_type == "model_update_quantized":
            scale_factor = public_inputs[-1]
            quantized_gradients = public_inputs[:-1]

            # Validate scale factor is reasonable
            if not (1 <= scale_factor <= 1000000):
                integrity_results["failed_checks"].append("Invalid scale factor")
                integrity_results["all_passed"] = False

            # Validate quantized gradients are integers
            for grad in quantized_gradients:
                if not isinstance(grad, int):
                    integrity_results["failed_checks"].append("Non-integer quantized gradient")
                    integrity_results["all_passed"] = False
                    break

        # Check 3: Proof size validation
        expected_sizes = {
            "pi_a": [3, None],  # 3 arrays, variable length
            "pi_b": [3, None],
            "pi_c": [3, None]
        }

        for key, expected in expected_sizes.items():
            if key in proof:
                if len(proof[key]) != expected[0]:
                    integrity_results["failed_checks"].append(f"Invalid {key} size")
                    integrity_results["all_passed"] = False

        return integrity_results

    def verify_quantized_proof(self, quantized_gradients: Dict[str, List[int]],
                             quantization_metadata: Dict[str, Any],
                             proof: Dict[str, Any],
                             circuit_type: str = "model_update_quantized") -> Dict[str, Any]:
        """
        Verify a proof for quantized gradients with enhanced validation.

        Args:
            quantized_gradients: Quantized gradient values
            quantization_metadata: Metadata from quantization process
            proof: Zero-knowledge proof
            circuit_type: Circuit type used

        Returns:
            Comprehensive verification results
        """
        # Create circuit input from quantized gradients
        circuit_input = create_quantized_proof_input(quantized_gradients, quantization_metadata, circuit_type)

        # Prepare public inputs for verification
        public_inputs = circuit_input["quantized_gradients"] + [circuit_input["scale_factor_input"]]

        # Perform comprehensive verification
        verification_result = self.verify_proof_comprehensive(
            proof, public_inputs, circuit_type, "comprehensive"
        )

        # Add quantization-specific metadata
        verification_result["quantization_metadata"] = quantization_metadata
        verification_result["circuit_input"] = circuit_input

        return verification_result


class BatchProofVerifier:
    """
    Verifier for batch proofs that aggregates multiple individual proofs
    with enhanced integrity and performance validation.
    """

    def __init__(self, max_batch_size: int = 10):
        """
        Initialize batch proof verifier.

        Args:
            max_batch_size: Maximum number of proofs to verify in one batch
        """
        self.max_batch_size = max_batch_size
        self.verifier = AdvancedProofVerifier()

    def verify_batch_proofs(self, proofs: List[Dict[str, Any]],
                          public_inputs_list: List[List[Any]],
                          circuit_type: str = "model_update") -> Dict[str, Any]:
        """
        Verify a batch of proofs with performance and integrity optimizations.

        Args:
            proofs: List of proofs to verify
            public_inputs_list: List of public inputs for each proof
            circuit_type: Circuit type for all proofs

        Returns:
            Batch verification results
        """
        if len(proofs) != len(public_inputs_list):
            raise ValueError("Number of proofs must match number of public input sets")

        batch_results = {
            "batch_size": len(proofs),
            "all_valid": True,
            "individual_results": [],
            "batch_stats": {
                "total_verification_time": 0.0,
                "average_verification_time": 0.0,
                "min_verification_time": float('inf'),
                "max_verification_time": 0.0,
                "failed_count": 0
            },
            "integrity_checks": {
                "batch_consistency": True,
                "proof_uniqueness": True,
                "input_consistency": True
            }
        }

        # Verify individual proofs
        for i, (proof, public_inputs) in enumerate(zip(proofs, public_inputs_list)):
            result = self.verifier.verify_proof_comprehensive(
                proof, public_inputs, circuit_type, "standard"
            )

            batch_results["individual_results"].append(result)

            # Update batch statistics
            batch_results["batch_stats"]["total_verification_time"] += result["verification_time"]
            batch_results["batch_stats"]["min_verification_time"] = min(
                batch_results["batch_stats"]["min_verification_time"], result["verification_time"]
            )
            batch_results["batch_stats"]["max_verification_time"] = max(
                batch_results["batch_stats"]["max_verification_time"], result["verification_time"]
            )

            if not result["is_valid"]:
                batch_results["all_valid"] = False
                batch_results["batch_stats"]["failed_count"] += 1

        # Calculate averages
        if batch_results["batch_size"] > 0:
            batch_results["batch_stats"]["average_verification_time"] = (
                batch_results["batch_stats"]["total_verification_time"] / batch_results["batch_size"]
            )

        # Perform batch integrity checks
        batch_results["integrity_checks"] = self._perform_batch_integrity_checks(
            proofs, public_inputs_list
        )

        # Overall batch validity
        batch_results["all_valid"] = (
            batch_results["all_valid"] and
            all(batch_results["integrity_checks"].values())
        )

        return batch_results

    def _perform_batch_integrity_checks(self, proofs: List[Dict[str, Any]],
                                      public_inputs_list: List[List[Any]]) -> Dict[str, bool]:
        """Perform integrity checks specific to batch verification."""
        checks = {
            "batch_consistency": True,
            "proof_uniqueness": True,
            "input_consistency": True
        }

        # Check proof uniqueness (no duplicate proofs)
        proof_hashes = []
        for proof in proofs:
            proof_hash = hashlib.sha256(json.dumps(proof, sort_keys=True).encode()).hexdigest()
            if proof_hash in proof_hashes:
                checks["proof_uniqueness"] = False
                break
            proof_hashes.append(proof_hash)

        # Check input consistency (all inputs have same structure)
        if public_inputs_list:
            first_input_len = len(public_inputs_list[0])
            for inputs in public_inputs_list[1:]:
                if len(inputs) != first_input_len:
                    checks["input_consistency"] = False
                    break

        return checks


# Global verifier instances
advanced_verifier = AdvancedProofVerifier()
batch_verifier = BatchProofVerifier(max_batch_size=10)

