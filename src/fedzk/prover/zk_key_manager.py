#!/usr/bin/env python3
"""
ZK Key Manager Integration
==========================

Integration layer between FEDzk key management system and ZK proof operations.
Provides secure key handling for zero-knowledge proof generation and verification.
"""

import logging
import secrets
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
from datetime import datetime
import json

from fedzk.security.key_manager import KeyManager, KeyType, KeyStorageType
from fedzk.prover.verifier import ZKVerifier
from fedzk.prover.zkgenerator import ZKGenerator

logger = logging.getLogger(__name__)

class ZKKeyManager:
    """
    Specialized key manager for ZK proof operations.

    Handles keys specific to zero-knowledge proof generation and verification,
    including proving keys, verification keys, and cryptographic parameters.
    """

    def __init__(self, key_manager: KeyManager, circuit_name: str = "model_update"):
        """
        Initialize ZK key manager.

        Args:
            key_manager: Underlying key management system
            circuit_name: Name of the ZK circuit
        """
        self.key_manager = key_manager
        self.circuit_name = circuit_name
        self.circuit_keys: Dict[str, str] = {}

        # Initialize circuit-specific keys
        self._initialize_circuit_keys()

    def _initialize_circuit_keys(self):
        """Initialize keys needed for ZK circuit operations."""
        # Define key patterns for this circuit
        key_patterns = {
            "proving_key": f"{self.circuit_name}_proving_key",
            "verification_key": f"{self.circuit_name}_verification_key",
            "toxic_waste": f"{self.circuit_name}_toxic_waste",
            "witness_calculator": f"{self.circuit_name}_witness_calculator"
        }

        self.circuit_keys = key_patterns

        # Ensure keys exist or create them
        self._ensure_circuit_keys_exist()

    def _ensure_circuit_keys_exist(self):
        """Ensure all required circuit keys exist."""
        for key_name, key_id in self.circuit_keys.items():
            if not self._key_exists(key_id):
                self._create_circuit_key(key_name, key_id)

    def _key_exists(self, key_id: str) -> bool:
        """Check if a key exists."""
        try:
            status = self.key_manager.get_key_status(key_id)
            return status.get("exists", False)
        except Exception:
            return False

    def _create_circuit_key(self, key_name: str, key_id: str):
        """Create a circuit-specific key."""
        try:
            # Generate appropriate key based on type
            if key_name == "proving_key":
                # Proving keys are typically large binary files
                key_data = self._load_or_generate_proving_key()
            elif key_name == "verification_key":
                # Verification keys are JSON structures
                key_data = self._load_or_generate_verification_key()
            elif key_name == "toxic_waste":
                # Toxic waste (ceremony contributions) - sensitive
                key_data = self._generate_toxic_waste()
            elif key_name == "witness_calculator":
                # Witness calculator code/binary
                key_data = self._load_witness_calculator()
            else:
                # Generic key
                key_data = b"default_circuit_key_data"

            # Store the key with appropriate metadata
            key_type = self._get_key_type_for_name(key_name)
            tags = {
                "circuit": self.circuit_name,
                "component": key_name,
                "zk_related": "true"
            }

            success = self.key_manager.store_key(
                key_id, key_data, key_type,
                tags=tags,
                algorithm=self._get_algorithm_for_key(key_name)
            )

            if success:
                logger.info(f"Created ZK key: {key_id} for circuit {self.circuit_name}")
            else:
                logger.error(f"Failed to create ZK key: {key_id}")

        except Exception as e:
            logger.error(f"Error creating circuit key {key_id}: {e}")

    def _load_or_generate_proving_key(self) -> bytes:
        """Load existing proving key or generate new one."""
        # In production, this would load from trusted setup ceremony
        # For now, return placeholder data
        return b"proving_key_placeholder_data_" + secrets.token_bytes(64)

    def _load_or_generate_verification_key(self) -> bytes:
        """Load existing verification key or generate new one."""
        # Try to load from standard location first
        vk_path = Path(f"./zk_setup/{self.circuit_name}_verification_key.json")

        if vk_path.exists():
            try:
                with open(vk_path, 'r') as f:
                    vk_data = json.load(f)
                return json.dumps(vk_data).encode('utf-8')
            except Exception as e:
                logger.warning(f"Failed to load verification key from {vk_path}: {e}")

        # Generate placeholder verification key structure
        placeholder_vk = {
            "protocol": "groth16",
            "curve": "bn128",
            "nPublic": 4,
            "vk_alpha_1": ["placeholder_alpha_1"],
            "vk_beta_2": [["placeholder_beta_2"]],
            "vk_gamma_2": [["placeholder_gamma_2"]],
            "vk_delta_2": [["placeholder_delta_2"]],
            "vk_alphabeta_12": ["placeholder_alphabeta"],
            "IC": ["placeholder_ic_0", "placeholder_ic_1"]
        }

        return json.dumps(placeholder_vk).encode('utf-8')

    def _generate_toxic_waste(self) -> bytes:
        """Generate toxic waste for trusted setup ceremony."""
        # Toxic waste should be securely generated and never reused
        return secrets.token_bytes(128)  # Large random data

    def _load_witness_calculator(self) -> bytes:
        """Load witness calculator binary/code."""
        # In production, this would be the compiled witness calculator
        return b"witness_calculator_placeholder_binary_data"

    def _get_key_type_for_name(self, key_name: str) -> KeyType:
        """Get appropriate key type for a key name."""
        type_mapping = {
            "proving_key": KeyType.PROVING_KEY,
            "verification_key": KeyType.VERIFICATION_KEY,
            "toxic_waste": KeyType.SYMMETRIC,  # Sensitive symmetric key
            "witness_calculator": KeyType.SYMMETRIC
        }
        return type_mapping.get(key_name, KeyType.SYMMETRIC)

    def _get_algorithm_for_key(self, key_name: str) -> str:
        """Get algorithm name for a key."""
        algorithm_mapping = {
            "proving_key": "groth16_proving_key",
            "verification_key": "groth16_verification_key",
            "toxic_waste": "trusted_setup_contribution",
            "witness_calculator": "witness_calculator_binary"
        }
        return algorithm_mapping.get(key_name, "unknown")

    def get_proving_key(self) -> Tuple[bytes, Dict[str, Any]]:
        """
        Get the proving key for this circuit.

        Returns:
            Tuple[bytes, Dict]: Key data and metadata
        """
        key_id = self.circuit_keys["proving_key"]
        return self._get_key_with_metadata(key_id)

    def get_verification_key(self) -> Tuple[bytes, Dict[str, Any]]:
        """
        Get the verification key for this circuit.

        Returns:
            Tuple[bytes, Dict]: Key data and metadata
        """
        key_id = self.circuit_keys["verification_key"]
        return self._get_key_with_metadata(key_id)

    def get_verification_key_json(self) -> Dict[str, Any]:
        """
        Get the verification key as parsed JSON.

        Returns:
            Dict: Parsed verification key
        """
        key_data, _ = self.get_verification_key()
        return json.loads(key_data.decode('utf-8'))

    def get_toxic_waste(self) -> Tuple[bytes, Dict[str, Any]]:
        """
        Get the toxic waste for this circuit.
        WARNING: Toxic waste should never be exposed in production!

        Returns:
            Tuple[bytes, Dict]: Toxic waste data and metadata
        """
        key_id = self.circuit_keys["toxic_waste"]
        return self._get_key_with_metadata(key_id)

    def get_witness_calculator(self) -> Tuple[bytes, Dict[str, Any]]:
        """
        Get the witness calculator for this circuit.

        Returns:
            Tuple[bytes, Dict]: Calculator data and metadata
        """
        key_id = self.circuit_keys["witness_calculator"]
        return self._get_key_with_metadata(key_id)

    def _get_key_with_metadata(self, key_id: str) -> Tuple[bytes, Dict[str, Any]]:
        """Get a key with its metadata."""
        try:
            key_data, metadata = self.key_manager.retrieve_key(key_id)

            # Convert metadata to dict
            metadata_dict = {
                "key_id": metadata.key_id,
                "key_type": metadata.key_type.value,
                "storage_type": metadata.storage_type.value,
                "created_at": metadata.created_at.isoformat() if metadata.created_at else None,
                "expires_at": metadata.expires_at.isoformat() if metadata.expires_at else None,
                "last_rotated_at": metadata.last_rotated_at.isoformat() if metadata.last_rotated_at else None,
                "rotation_policy": metadata.rotation_policy.value,
                "usage_count": metadata.usage_count,
                "algorithm": metadata.algorithm,
                "key_size": metadata.key_size,
                "fingerprint": metadata.fingerprint,
                "tags": metadata.tags
            }

            return key_data, metadata_dict

        except Exception as e:
            logger.error(f"Failed to retrieve key {key_id}: {e}")
            raise

    def rotate_circuit_keys(self) -> Dict[str, bool]:
        """
        Rotate all circuit keys.

        Returns:
            Dict[str, bool]: Rotation success status for each key
        """
        rotation_results = {}

        for key_name, key_id in self.circuit_keys.items():
            try:
                # Only rotate certain types of keys
                if key_name in ["proving_key", "verification_key"]:
                    success = self.key_manager.rotate_key(key_id)
                    rotation_results[key_name] = success

                    if success:
                        logger.info(f"Rotated ZK key: {key_name}")
                    else:
                        logger.error(f"Failed to rotate ZK key: {key_name}")
                else:
                    # Some keys (like toxic waste) should not be rotated
                    rotation_results[key_name] = True  # Consider as successful (no-op)

            except Exception as e:
                logger.error(f"Error rotating key {key_name}: {e}")
                rotation_results[key_name] = False

        return rotation_results

    def get_circuit_key_status(self) -> Dict[str, Dict[str, Any]]:
        """
        Get status of all circuit keys.

        Returns:
            Dict: Status information for each circuit key
        """
        status = {}

        for key_name, key_id in self.circuit_keys.items():
            try:
                key_status = self.key_manager.get_key_status(key_id)
                status[key_name] = key_status
            except Exception as e:
                status[key_name] = {
                    "key_id": key_id,
                    "exists": False,
                    "error": str(e)
                }

        return status

    def backup_circuit_keys(self, backup_path: Optional[str] = None) -> bool:
        """
        Backup all circuit keys.

        Args:
            backup_path: Path to backup location

        Returns:
            bool: Backup success status
        """
        if backup_path is None:
            backup_path = f"./backups/{self.circuit_name}_keys_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        try:
            backup_dir = Path(backup_path)
            backup_dir.mkdir(parents=True, exist_ok=True)

            success_count = 0

            for key_name, key_id in self.circuit_keys.items():
                try:
                    key_data, metadata = self.key_manager.retrieve_key(key_id)

                    # Save key data
                    key_file = backup_dir / f"{key_name}.key"
                    with open(key_file, 'wb') as f:
                        f.write(key_data)

                    # Save metadata
                    meta_file = backup_dir / f"{key_name}.meta"
                    with open(meta_file, 'w') as f:
                        json.dump({
                            "key_id": metadata.key_id,
                            "key_type": metadata.key_type.value,
                            "algorithm": metadata.algorithm,
                            "created_at": metadata.created_at.isoformat() if metadata.created_at else None,
                            "tags": metadata.tags
                        }, f, indent=2)

                    success_count += 1
                    logger.info(f"Backed up key: {key_name}")

                except Exception as e:
                    logger.error(f"Failed to backup key {key_name}: {e}")

            total_keys = len(self.circuit_keys)
            success = success_count == total_keys

            if success:
                logger.info(f"Successfully backed up all {total_keys} circuit keys to {backup_path}")
            else:
                logger.warning(f"Backed up {success_count}/{total_keys} circuit keys to {backup_path}")

            return success

        except Exception as e:
            logger.error(f"Failed to create circuit key backup: {e}")
            return False

    def restore_circuit_keys(self, backup_path: str) -> bool:
        """
        Restore circuit keys from backup.

        Args:
            backup_path: Path to backup location

        Returns:
            bool: Restore success status
        """
        try:
            backup_dir = Path(backup_path)
            if not backup_dir.exists():
                raise FileNotFoundError(f"Backup path does not exist: {backup_path}")

            success_count = 0

            for key_name, key_id in self.circuit_keys.items():
                try:
                    # Load key data
                    key_file = backup_dir / f"{key_name}.key"
                    if not key_file.exists():
                        logger.warning(f"Key file not found in backup: {key_name}")
                        continue

                    with open(key_file, 'rb') as f:
                        key_data = f.read()

                    # Load metadata
                    meta_file = backup_dir / f"{key_name}.meta"
                    if meta_file.exists():
                        with open(meta_file, 'r') as f:
                            meta_data = json.load(f)
                        algorithm = meta_data.get("algorithm", "unknown")
                        tags = meta_data.get("tags", {})
                    else:
                        algorithm = "unknown"
                        tags = {}

                    # Store the key
                    key_type = self._get_key_type_for_name(key_name)
                    success = self.key_manager.store_key(
                        key_id, key_data, key_type,
                        tags={**tags, "restored_from_backup": backup_path},
                        algorithm=algorithm
                    )

                    if success:
                        success_count += 1
                        logger.info(f"Restored key: {key_name}")

                except Exception as e:
                    logger.error(f"Failed to restore key {key_name}: {e}")

            total_keys = len(self.circuit_keys)
            success = success_count > 0  # At least some keys restored

            if success:
                logger.info(f"Successfully restored {success_count}/{total_keys} circuit keys from {backup_path}")
            else:
                logger.error(f"Failed to restore any circuit keys from {backup_path}")

            return success

        except Exception as e:
            logger.error(f"Failed to restore circuit keys from backup: {e}")
            return False

    def get_circuit_security_metrics(self) -> Dict[str, Any]:
        """
        Get security metrics specific to this circuit's keys.

        Returns:
            Dict: Circuit-specific security metrics
        """
        metrics = {
            "circuit_name": self.circuit_name,
            "key_status": {},
            "security_score": 100,
            "recommendations": []
        }

        # Get status of each key
        key_status = self.get_circuit_key_status()
        metrics["key_status"] = key_status

        # Calculate security score
        security_score = 100
        recommendations = []

        for key_name, status in key_status.items():
            if not status.get("exists", False):
                security_score -= 25
                recommendations.append(f"Missing key: {key_name}")

            if status.get("is_expired", False):
                security_score -= 20
                recommendations.append(f"Expired key: {key_name}")

            if status.get("needs_rotation", True):  # Most ZK keys benefit from rotation
                security_score -= 10
                recommendations.append(f"Rotation recommended: {key_name}")

            # Check usage patterns
            usage_count = status.get("usage_count", 0)
            if usage_count > 10000:  # High usage
                security_score -= 5
                recommendations.append(f"High usage on key: {key_name}")

        # Check for toxic waste exposure (security risk)
        if "toxic_waste" in key_status:
            toxic_status = key_status["toxic_waste"]
            if toxic_status.get("usage_count", 0) > 0:
                security_score -= 50  # Major security risk
                recommendations.append("CRITICAL: Toxic waste has been accessed")

        metrics["security_score"] = max(0, security_score)
        metrics["recommendations"] = recommendations

        return metrics

# Convenience functions for ZK integration
def create_zk_key_manager(key_manager: KeyManager, circuit_name: str = "model_update") -> ZKKeyManager:
    """
    Create a ZK key manager for a specific circuit.

    Args:
        key_manager: Base key management system
        circuit_name: Name of the ZK circuit

    Returns:
        ZKKeyManager: Configured ZK key manager
    """
    return ZKKeyManager(key_manager, circuit_name)

def setup_federated_learning_keys(key_manager: KeyManager) -> Dict[str, str]:
    """
    Setup standard keys needed for federated learning.

    Args:
        key_manager: Key management system

    Returns:
        Dict[str, str]: Key IDs for FL operations
    """
    from fedzk.security.key_manager import generate_federated_learning_keys
    return generate_federated_learning_keys(key_manager)

# Integration with ZKVerifier
def get_verifier_with_secure_keys(zk_key_manager: ZKKeyManager) -> ZKVerifier:
    """
    Create a ZKVerifier using secure key management.

    Args:
        zk_key_manager: ZK key manager instance

    Returns:
        ZKVerifier: Configured verifier
    """
    try:
        # Get verification key
        vk_data, _ = zk_key_manager.get_verification_key()
        vk_json = json.loads(vk_data.decode('utf-8'))

        # Create temporary file for verifier
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(vk_json, f)
            vk_path = f.name

        verifier = ZKVerifier(vk_path)

        # Clean up temporary file (in production, use secure temp location)
        import os
        os.unlink(vk_path)

        return verifier

    except Exception as e:
        logger.error(f"Failed to create secure ZK verifier: {e}")
        raise
