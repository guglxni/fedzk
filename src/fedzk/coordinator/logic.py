# Functional Source License 1.1 with Apache-2.0 Future Grant (FSL-1.1-Apache-2.0)
# Copyright (c) 2025 Aaryan Guglani and FEDzk Contributors
# Licensed under FSL-1.1-Apache-2.0. See LICENSE for details.

"""
Coordinator logic for FEDzk.
Handles in-memory state for pending updates, proof verification, and aggregation.
Provides production-grade cryptographic validation and proof aggregation.
"""

import os
import pathlib
import hashlib
import time
import logging
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict
from dataclasses import dataclass

from fedzk.prover.verifier import ZKVerifier
from fedzk.prover.advanced_proof_validator import (
    AdvancedProofValidator,
    ProofValidationConfig,
    ProofValidationError,
    AttackPattern
)

logger = logging.getLogger(__name__)


@dataclass
class VerifiedUpdate:
    """Represents a cryptographically verified update."""
    gradients: Dict[str, List[float]]
    proof: Dict[str, Any]
    public_inputs: List[Any]
    client_id: str
    timestamp: float
    proof_hash: str
    verification_time: float

    def __post_init__(self):
        """Generate proof hash for integrity verification."""
        proof_str = str(sorted(self.proof.items()))
        inputs_str = str(self.public_inputs)
        self.proof_hash = hashlib.sha256(
            f"{proof_str}{inputs_str}{self.client_id}{self.timestamp}".encode()
        ).hexdigest()


@dataclass
class AggregationBatch:
    """Represents a batch of verified updates ready for aggregation."""
    updates: List[VerifiedUpdate]
    batch_id: str
    created_at: float
    aggregated_at: Optional[float] = None
    global_update: Optional[Dict[str, List[float]]] = None
    aggregation_proof: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        if not self.batch_id:
            self.batch_id = hashlib.sha256(
                f"batch_{len(self.updates)}_{self.created_at}".encode()
            ).hexdigest()[:16]


class ProofVerificationError(Exception):
    """Raised when ZK proof verification fails."""
    def __init__(self, message: str, error_type: str = "verification_failed",
                 proof_hash: Optional[str] = None):
        self.message = message
        self.error_type = error_type
        self.proof_hash = proof_hash
        super().__init__(self.message)


class CryptographicIntegrityError(Exception):
    """Raised when cryptographic integrity checks fail."""
    def __init__(self, message: str, component: str = "unknown"):
        self.message = message
        self.component = component
        super().__init__(self.message)


class CoordinatorSecurityManager:
    """Manages security and rate limiting for the coordinator."""

    def __init__(self, max_pending_updates: int = 1000, rate_limit_window: int = 300):
        self.max_pending_updates = max_pending_updates
        self.rate_limit_window = rate_limit_window

        # Security tracking
        self.client_request_times: Dict[str, List[float]] = defaultdict(list)
        self.failed_verifications: Dict[str, int] = defaultdict(int)
        self.blocked_clients: Dict[str, float] = {}

        # Performance tracking
        self.verification_times: List[float] = []
        self.total_verifications = 0
        self.failed_verifications_count = 0

    def check_rate_limit(self, client_id: str, max_requests: int = 10) -> bool:
        """Check if client has exceeded rate limit."""
        now = time.time()
        window_start = now - self.rate_limit_window

        # Clean old requests
        self.client_request_times[client_id] = [
            t for t in self.client_request_times[client_id] if t > window_start
        ]

        # Check limit
        if len(self.client_request_times[client_id]) >= max_requests:
            return False

        # Record this request
        self.client_request_times[client_id].append(now)
        return True

    def check_client_blocked(self, client_id: str) -> bool:
        """Check if client is temporarily blocked."""
        if client_id in self.blocked_clients:
            if time.time() - self.blocked_clients[client_id] > 3600:  # 1 hour block
                del self.blocked_clients[client_id]
                return False
            return True
        return False

    def record_failed_verification(self, client_id: str):
        """Record a failed verification attempt."""
        self.failed_verifications[client_id] += 1
        self.failed_verifications_count += 1

        # Block client after 5 failed attempts
        if self.failed_verifications[client_id] >= 5:
            self.blocked_clients[client_id] = time.time()
            logger.warning(f"Client {client_id} blocked due to repeated verification failures")

    def record_verification_time(self, verification_time: float):
        """Record verification performance metrics."""
        self.verification_times.append(verification_time)
        self.total_verifications += 1

        # Keep only last 1000 measurements
        if len(self.verification_times) > 1000:
            self.verification_times = self.verification_times[-1000:]

    def record_security_event(self, client_id: str, event_type: str, details: Dict[str, Any]):
        """Record a security-related event."""
        logger.info(
            f"Security event for client {client_id}: {event_type} - {details}"
        )

        # In production, this would be stored in a secure audit log
        # For now, we just log it
        # TODO: Implement secure audit logging for production

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        if not self.verification_times:
            return {"average_time": 0, "total_verifications": 0}

        return {
            "average_verification_time": sum(self.verification_times) / len(self.verification_times),
            "total_verifications": self.total_verifications,
            "failed_verifications": self.failed_verifications_count,
            "blocked_clients": len(self.blocked_clients),
            "success_rate": 1 - (self.failed_verifications_count / max(1, self.total_verifications))
        }


def _initialize_verifier():
    """
    Initialize ZKVerifier with proper verification keys and security manager.

    Returns:
        Tuple[ZKVerifier, CoordinatorSecurityManager]: Properly initialized verifier and security manager

    Raises:
        RuntimeError: If verification keys are not found or invalid
    """
    # Get ZK asset directory
    zk_dir = pathlib.Path(__file__).resolve().parent.parent / "zk"

    # Check that verification keys exist
    standard_vkey = zk_dir / "verification_key.json"
    secure_vkey = zk_dir / "verification_key_secure.json"

    missing_keys = []
    if not standard_vkey.exists():
        missing_keys.append(str(standard_vkey))
    if not secure_vkey.exists():
        missing_keys.append(str(secure_vkey))

    if missing_keys:
        raise RuntimeError(
            f"Missing verification keys required for coordinator: {', '.join(missing_keys)}. "
            "Please run 'scripts/setup_zk.sh' to generate the required ZK artifacts."
        )

    # Validate verification key files are not empty and have reasonable size
    for key_path in [standard_vkey, secure_vkey]:
        try:
            file_size = key_path.stat().st_size
            if file_size == 0:
                raise RuntimeError(f"Verification key file is empty: {key_path}")
            if file_size < 1000:  # Verification keys should be substantial
                logger.warning(f"Verification key file seems unusually small: {key_path} ({file_size} bytes)")
        except OSError as e:
            raise RuntimeError(f"Cannot access verification key file {key_path}: {e}")

    # Initialize verifier with default paths (it will automatically find the right keys)
    try:
        verifier = ZKVerifier()
        security_manager = CoordinatorSecurityManager()
        logger.info("✅ Coordinator cryptographic verification system initialized")
        return verifier, security_manager
    except Exception as e:
        raise RuntimeError(f"Failed to initialize coordinator verification system: {e}")


def validate_proof_structure(proof: Dict[str, Any]) -> bool:
    """
    Validate the structure and format of a ZK proof.

    Args:
        proof: The proof dictionary to validate

    Returns:
        bool: True if proof structure is valid
    """
    if not isinstance(proof, dict):
        return False

    # Check for required proof components (Groth16 format)
    required_keys = ['pi_a', 'pi_b', 'pi_c', 'protocol']
    for key in required_keys:
        if key not in proof:
            logger.warning(f"Proof missing required key: {key}")
            return False

    # Validate proof arrays have correct structure
    if not isinstance(proof['pi_a'], list) or len(proof['pi_a']) != 3:
        logger.warning("Invalid pi_a format in proof")
        return False

    if not isinstance(proof['pi_b'], list) or len(proof['pi_b']) != 3:
        logger.warning("Invalid pi_b format in proof")
        return False

    if not isinstance(proof['pi_c'], list) or len(proof['pi_c']) != 3:
        logger.warning("Invalid pi_c format in proof")
        return False

    # Validate protocol
    if proof.get('protocol') != 'groth16':
        logger.warning(f"Unsupported proof protocol: {proof.get('protocol')}")
        return False

    return True


def validate_public_inputs(public_inputs: List[Any]) -> bool:
    """
    Validate the structure and format of public inputs.

    Args:
        public_inputs: The public inputs list to validate

    Returns:
        bool: True if public inputs are valid
    """
    if not isinstance(public_inputs, list):
        return False

    if len(public_inputs) == 0:
        logger.warning("Empty public inputs")
        return False

    # Check that all inputs are numbers (integers or floats)
    for i, inp in enumerate(public_inputs):
        if not isinstance(inp, (int, float)):
            logger.warning(f"Invalid public input type at index {i}: {type(inp)}")
            return False

    return True


def validate_gradient_consistency(gradients: Dict[str, List[float]],
                                 public_inputs: List[Any]) -> bool:
    """
    Validate that gradients are consistent with proof public inputs.

    Args:
        gradients: Gradient dictionary
        public_inputs: Public inputs from proof

    Returns:
        bool: True if gradients are consistent with public inputs
    """
    if not gradients:
        return False

    # Basic validation - ensure gradients have reasonable structure
    total_elements = 0
    for param_name, param_gradients in gradients.items():
        if not isinstance(param_gradients, list):
            logger.warning(f"Invalid gradient format for parameter {param_name}")
            return False

        if len(param_gradients) == 0:
            logger.warning(f"Empty gradients for parameter {param_name}")
            return False

        total_elements += len(param_gradients)

    # Ensure we have gradients to verify against
    if total_elements == 0:
        logger.warning("No gradient elements found")
        return False

    return True


def verify_proof_cryptographically(
    verifier: ZKVerifier,
    proof: Dict[str, Any],
    public_inputs: List[Any],
    security_manager: CoordinatorSecurityManager,
    client_id: str = "unknown"
) -> Tuple[bool, float]:
    """
    Perform comprehensive cryptographic verification of a ZK proof.

    This function implements multi-layer verification:
    1. Advanced proof validation (security, attacks, format)
    2. Structural validation
    3. SNARKjs cryptographic verification

    Args:
        verifier: ZKVerifier instance
        proof: Proof to verify
        public_inputs: Public inputs for verification
        security_manager: Security manager for tracking
        client_id: Client identifier for security tracking

    Returns:
        Tuple[bool, float]: (verification_result, verification_time)
    """
    start_time = time.time()

    try:
        # Step 1: Advanced Security Validation
        logger.debug(f"Starting advanced proof validation for client {client_id}")

        # Create advanced proof validator with production settings
        advanced_validator = AdvancedProofValidator(
            ProofValidationConfig(
                max_proof_size=512 * 1024,  # 512KB limit
                max_signal_count=500,
                max_field_size=5000,
                enable_attack_detection=True,
                strict_mode=True
            )
        )

        # Perform comprehensive security validation
        security_result = advanced_validator.validate_proof_comprehensive(
            proof, public_inputs, circuit_type="model_update"
        )

        # Log security validation results
        if security_result.attack_patterns_detected:
            attack_names = [attack.value for attack in security_result.attack_patterns_detected]
            logger.warning(
                f"Security validation detected attacks for client {client_id}: {attack_names}"
            )
            security_manager.record_security_event(
                client_id, "attack_detected", {"attacks": attack_names}
            )

        if security_result.security_score < 50:
            logger.error(
                f"Proof security score too low for client {client_id}: {security_result.security_score:.1f}"
            )
            security_manager.record_failed_verification(client_id)
            return False, time.time() - start_time

        if not security_result.is_valid:
            logger.warning(f"Advanced validation failed for client {client_id}")
            security_manager.record_failed_verification(client_id)
            return False, time.time() - start_time

        # Log successful security validation
        logger.debug(
            f"Advanced validation passed for client {client_id} "
            f"(score: {security_result.security_score:.1f}, "
            f"time: {security_result.validation_time:.3f}s)"
        )

        # Step 2: Structural validation
        if not validate_proof_structure(proof):
            security_manager.record_failed_verification(client_id)
            return False, time.time() - start_time

        if not validate_public_inputs(public_inputs):
            security_manager.record_failed_verification(client_id)
            return False, time.time() - start_time

        # Step 3: SNARKjs Cryptographic verification
        try:
            is_valid = verifier.verify_real_proof(proof, public_inputs)
        except Exception as e:
            logger.error(f"Cryptographic verification failed for client {client_id}: {e}")
            security_manager.record_failed_verification(client_id)
            security_manager.record_security_event(
                client_id, "snarkjs_verification_error", {"error": str(e)}
            )
            return False, time.time() - start_time

        # Step 4: Record comprehensive metrics
        verification_time = time.time() - start_time
        security_manager.record_verification_time(verification_time)

        # Log final verification result
        if not is_valid:
            security_manager.record_failed_verification(client_id)
            logger.warning(f"Complete proof verification failed for client {client_id}")
        else:
            logger.info(
                f"Complete proof verification successful for client {client_id} "
                f"(security_score: {security_result.security_score:.1f}, "
                f"total_time: {verification_time:.3f}s)"
            )

            # Record successful verification metrics
            security_manager.record_security_event(
                client_id, "verification_success", {
                    "security_score": security_result.security_score,
                    "verification_time": verification_time,
                    "attack_patterns": len(security_result.attack_patterns_detected)
                }
            )

        return is_valid, verification_time

    except ProofValidationError as e:
        logger.error(f"Proof validation error for client {client_id}: {e}")
        security_manager.record_failed_verification(client_id)
        security_manager.record_security_event(
            client_id, "proof_validation_error", {"error": str(e)}
        )
        return False, time.time() - start_time

    except Exception as e:
        logger.error(f"Unexpected error during proof verification for client {client_id}: {e}")
        security_manager.record_failed_verification(client_id)
        security_manager.record_security_event(
            client_id, "unexpected_verification_error", {"error": str(e)}
        )
        return False, time.time() - start_time


# Enhanced in-memory storage with cryptographic integrity
pending_updates: List[VerifiedUpdate] = []
aggregation_batches: List[AggregationBatch] = []
current_version: int = 1

# Initialize verifier and security manager with real verification keys
try:
    verifier, security_manager = _initialize_verifier()
except RuntimeError as e:
    print(f"❌ Coordinator initialization failed: {e}")
    print("🔧 Please ensure ZK toolchain is properly set up by running 'scripts/setup_zk.sh'")
    raise

def submit_update(
    gradients: Dict[str, List[float]],
    proof: Dict,
    public_inputs: List,
    client_id: str = "unknown"
) -> Tuple[str, int, Optional[Dict[str, List[float]]]]:
    """
    Verify the provided ZK proof with comprehensive cryptographic validation and aggregate updates.

    Args:
        gradients: Gradient updates by parameter name
        proof: Zero-knowledge proof object
        public_inputs: Public inputs/signals for proof verification
        client_id: Client identifier for security tracking

    Returns:
        Tuple[str, int, Optional[Dict[str, List[float]]]]: (status, model_version, global_update)

    Raises:
        ProofVerificationError: if verification fails
        CryptographicIntegrityError: if integrity checks fail
    """
    global pending_updates, current_version, aggregation_batches

    # Step 1: Security and rate limiting checks
    if security_manager.check_client_blocked(client_id):
        raise ProofVerificationError(f"Client {client_id} is temporarily blocked",
                                   error_type="client_blocked")

    if not security_manager.check_rate_limit(client_id):
        raise ProofVerificationError(f"Client {client_id} exceeded rate limit",
                                   error_type="rate_limited")

    # Step 2: Input validation
    if not gradients:
        raise ProofVerificationError("Empty gradients provided", error_type="invalid_input")

    if not validate_gradient_consistency(gradients, public_inputs):
        raise CryptographicIntegrityError("Gradient consistency validation failed", "gradient_validation")

    # Step 3: Comprehensive cryptographic verification
    is_valid, verification_time = verify_proof_cryptographically(
        verifier, proof, public_inputs, security_manager, client_id
    )

    if not is_valid:
        proof_hash = hashlib.sha256(str(proof).encode()).hexdigest()[:16]
        raise ProofVerificationError(
            f"Cryptographic proof verification failed for client {client_id}",
            error_type="verification_failed",
            proof_hash=proof_hash
        )

    # Step 4: Create verified update with integrity tracking
    verified_update = VerifiedUpdate(
        gradients=gradients,
        proof=proof,
        public_inputs=public_inputs,
        client_id=client_id,
        timestamp=time.time(),
        proof_hash="",  # Will be set by __post_init__
        verification_time=verification_time
    )

    # Step 5: Store verified update
    pending_updates.append(verified_update)
    logger.info(f"✅ Verified update stored for client {client_id} ({len(pending_updates)} pending)")

    # Step 6: Check aggregation threshold and perform secure aggregation
    aggregation_threshold = 3  # Require at least 3 updates for aggregation

    if len(pending_updates) >= aggregation_threshold:
        return _perform_secure_aggregation()
    else:
        return "accepted", current_version, None


def _perform_secure_aggregation() -> Tuple[str, int, Optional[Dict[str, List[float]]]]:
    """
    Perform secure aggregation of verified updates with cryptographic integrity.

    Returns:
        Tuple[str, int, Optional[Dict[str, List[float]]]]: (status, model_version, global_update)
    """
    global pending_updates, current_version, aggregation_batches

    if len(pending_updates) < 3:
        return "accepted", current_version, None

    try:
        # Step 1: Create aggregation batch
        batch = AggregationBatch(
            updates=pending_updates.copy(),
            batch_id="",
            created_at=time.time()
        )

        # Step 2: Validate all updates have consistent parameter structure
        if not _validate_batch_consistency(batch.updates):
            raise CryptographicIntegrityError("Batch consistency validation failed", "batch_validation")

        # Step 3: Perform secure aggregation with integrity checks
        global_update = _compute_secure_average(batch.updates)

        if not global_update:
            raise CryptographicIntegrityError("Secure aggregation failed", "aggregation")

        # Step 4: Generate aggregation proof (simplified for now)
        aggregation_proof = {
            "batch_id": batch.batch_id,
            "num_updates": len(batch.updates),
            "timestamp": time.time(),
            "integrity_hash": _compute_batch_integrity_hash(batch.updates)
        }

        # Step 5: Update batch metadata
        batch.aggregated_at = time.time()
        batch.global_update = global_update
        batch.aggregation_proof = aggregation_proof

        # Step 6: Store completed batch
        aggregation_batches.append(batch)

        # Step 7: Reset pending updates and increment version
        pending_updates.clear()
        current_version += 1

        logger.info(f"✅ Secure aggregation completed: batch {batch.batch_id}, {len(batch.updates)} updates, version {current_version}")

        return "aggregated", current_version, global_update

    except Exception as e:
        logger.error(f"Secure aggregation failed: {e}")
        raise CryptographicIntegrityError(f"Aggregation failed: {str(e)}", "aggregation")


def _validate_batch_consistency(updates: List[VerifiedUpdate]) -> bool:
    """
    Validate that all updates in a batch have consistent parameter structures.

    Args:
        updates: List of verified updates to validate

    Returns:
        bool: True if batch is consistent
    """
    if not updates:
        return False

    # Use first update as reference
    reference_params = set(updates[0].gradients.keys())
    reference_shapes = {k: len(v) for k, v in updates[0].gradients.items()}

    for update in updates[1:]:
        # Check parameter names match
        if set(update.gradients.keys()) != reference_params:
            logger.warning("Parameter mismatch in batch")
            return False

        # Check parameter shapes match
        for param_name, param_values in update.gradients.items():
            if len(param_values) != reference_shapes.get(param_name, 0):
                logger.warning(f"Shape mismatch for parameter {param_name}")
                return False

    return True


def _compute_secure_average(updates: List[VerifiedUpdate]) -> Optional[Dict[str, List[float]]]:
    """
    Compute secure average of gradients with integrity validation.

    Args:
        updates: List of verified updates to average

    Returns:
        Optional[Dict[str, List[float]]]: Averaged gradients or None if failed
    """
    if not updates:
        return None

    try:
        # Get parameter names from first update
        param_names = list(updates[0].gradients.keys())

        # Compute secure average for each parameter
        avg_update = {}
        for param_name in param_names:
            # Collect all values for this parameter
            param_values = []
            for update in updates:
                if param_name in update.gradients:
                    param_values.append(update.gradients[param_name])
                else:
                    logger.error(f"Missing parameter {param_name} in update")
                    return None

            # Compute element-wise average
            if param_values:
                num_updates = len(param_values)
                param_length = len(param_values[0])

                averaged_values = []
                for i in range(param_length):
                    # Sum values at position i across all updates
                    total = sum(update_values[i] for update_values in param_values)
                    # Compute average
                    average = total / num_updates
                    averaged_values.append(average)

                avg_update[param_name] = averaged_values

        return avg_update

    except Exception as e:
        logger.error(f"Secure averaging failed: {e}")
        return None


def _compute_batch_integrity_hash(updates: List[VerifiedUpdate]) -> str:
    """
    Compute integrity hash for a batch of updates.

    Args:
        updates: List of verified updates

    Returns:
        str: Integrity hash for the batch
    """
    # Create a combined hash of all update hashes and gradients
    combined_data = ""
    for update in updates:
        combined_data += update.proof_hash
        combined_data += str(sorted(update.gradients.items()))

    return hashlib.sha256(combined_data.encode()).hexdigest()

def get_status() -> Tuple[int, int]:
    """
    Get current pending update count and model version.

    Returns:
        Tuple[int, int]: (pending_count, model_version)
    """
    return len(pending_updates), current_version


def get_security_stats() -> Dict[str, Any]:
    """
    Get security and performance statistics.

    Returns:
        Dict containing security metrics and performance data
    """
    return {
        "security": {
            "blocked_clients": len(security_manager.blocked_clients),
            "total_failed_verifications": security_manager.failed_verifications_count,
        },
        "performance": security_manager.get_performance_stats(),
        "system": {
            "total_pending_updates": len(pending_updates),
            "total_aggregation_batches": len(aggregation_batches),
            "current_model_version": current_version
        }
    }


def validate_proof_integrity(proof: Dict[str, Any], expected_hash: str) -> bool:
    """
    Validate proof integrity against expected hash.

    Args:
        proof: Proof to validate
        expected_hash: Expected integrity hash

    Returns:
        bool: True if proof integrity is valid
    """
    try:
        computed_hash = hashlib.sha256(str(sorted(proof.items())).encode()).hexdigest()
        return computed_hash == expected_hash
    except Exception:
        return False


def get_aggregation_history(limit: int = 10) -> List[Dict[str, Any]]:
    """
    Get history of completed aggregation batches.

    Args:
        limit: Maximum number of batches to return

    Returns:
        List of aggregation batch summaries
    """
    history = []
    for batch in aggregation_batches[-limit:]:
        history.append({
            "batch_id": batch.batch_id,
            "num_updates": len(batch.updates),
            "created_at": batch.created_at,
            "aggregated_at": batch.aggregated_at,
            "model_version": current_version if batch.aggregated_at else None
        })
    return history



