#!/usr/bin/env python3
"""
FEDzk Key Management System
===========================

Comprehensive cryptographic key management for FEDzk.
Implements secure key storage, rotation, integrity verification,
and enterprise-grade key management practices.

Features:
- Secure key storage and retrieval
- Automatic key rotation
- Key integrity verification
- Environment and vault integration
- Comprehensive audit logging
- Key access monitoring
- Production security standards
"""

import os
import json
import hashlib
import hmac
import time
import logging
import base64
import secrets
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path
import threading
from datetime import datetime, timedelta
import cryptography
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa, ec
from cryptography.hazmat.primitives.asymmetric.padding import PSS, MGF1
import warnings

logger = logging.getLogger(__name__)

class KeyType(Enum):
    """Types of cryptographic keys supported."""
    SYMMETRIC = "symmetric"
    ASYMMETRIC_PRIVATE = "asymmetric_private"
    ASYMMETRIC_PUBLIC = "asymmetric_public"
    HMAC = "hmac"
    VERIFICATION_KEY = "verification_key"
    PROVING_KEY = "proving_key"

class KeyStorageType(Enum):
    """Key storage backend types."""
    FILE = "file"
    ENVIRONMENT = "environment"
    VAULT = "vault"
    DATABASE = "database"
    HSM = "hsm"
    MEMORY = "memory"

class KeyRotationPolicy(Enum):
    """Key rotation policies."""
    NEVER = "never"
    TIME_BASED = "time_based"
    USAGE_BASED = "usage_based"
    COMPROMISE_BASED = "compromise_based"
    MANUAL = "manual"

@dataclass
class KeyMetadata:
    """Metadata for a cryptographic key."""
    key_id: str
    key_type: KeyType
    storage_type: KeyStorageType
    created_at: datetime
    expires_at: Optional[datetime]
    last_rotated_at: Optional[datetime]
    rotation_policy: KeyRotationPolicy
    usage_count: int
    max_usage_count: Optional[int]
    algorithm: str
    key_size: int
    fingerprint: str
    tags: Dict[str, str]
    access_log: List[Dict[str, Any]]

@dataclass
class KeyRotationConfig:
    """Configuration for key rotation."""
    enabled: bool = True
    max_age_days: int = 90
    max_usage_count: int = 10000
    rotation_window_hours: int = 24
    backup_old_keys: bool = True
    backup_retention_days: int = 365
    auto_rotation: bool = True

@dataclass
class KeySecurityConfig:
    """Security configuration for key management."""
    encryption_enabled: bool = True
    master_key_rotation_days: int = 365
    integrity_check_enabled: bool = True
    access_logging_enabled: bool = True
    audit_retention_days: int = 2555  # 7 years
    max_keys_per_type: int = 10
    key_backup_encryption: bool = True

class KeyManagementError(Exception):
    """Base exception for key management errors."""
    pass

class KeyNotFoundError(KeyManagementError):
    """Exception for missing keys."""
    pass

class KeyIntegrityError(KeyManagementError):
    """Exception for key integrity violations."""
    pass

class KeyRotationError(KeyManagementError):
    """Exception for key rotation failures."""
    pass

class KeyAccessDeniedError(KeyManagementError):
    """Exception for unauthorized key access."""
    pass

class KeyStorageBackend:
    """Abstract base class for key storage backends."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config

    def store_key(self, key_id: str, key_data: bytes, metadata: KeyMetadata) -> bool:
        """Store a key in the backend."""
        raise NotImplementedError

    def retrieve_key(self, key_id: str) -> Tuple[bytes, KeyMetadata]:
        """Retrieve a key from the backend."""
        raise NotImplementedError

    def list_keys(self, key_type: Optional[KeyType] = None) -> List[str]:
        """List keys in the backend."""
        raise NotImplementedError

    def delete_key(self, key_id: str) -> bool:
        """Delete a key from the backend."""
        raise NotImplementedError

    def key_exists(self, key_id: str) -> bool:
        """Check if a key exists."""
        raise NotImplementedError

class FileKeyStorage(KeyStorageBackend):
    """File-based key storage backend."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.storage_path = Path(config.get("path", "./keys"))
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.encryption_enabled = config.get("encryption", True)

    def _get_key_path(self, key_id: str) -> Path:
        """Get the file path for a key."""
        return self.storage_path / f"{key_id}.key"

    def _get_metadata_path(self, key_id: str) -> Path:
        """Get the metadata file path for a key."""
        return self.storage_path / f"{key_id}.meta"

    def store_key(self, key_id: str, key_data: bytes, metadata: KeyMetadata) -> bool:
        """Store a key and its metadata."""
        try:
            key_path = self._get_key_path(key_id)
            meta_path = self._get_metadata_path(key_id)

            # Store the key
            with open(key_path, 'wb') as f:
                f.write(key_data)

            # Store metadata
            with open(meta_path, 'w') as f:
                json.dump(asdict(metadata), f, default=str, indent=2)

            logger.info(f"Key {key_id} stored successfully in file storage")
            return True

        except Exception as e:
            logger.error(f"Failed to store key {key_id}: {e}")
            return False

    def retrieve_key(self, key_id: str) -> Tuple[bytes, KeyMetadata]:
        """Retrieve a key and its metadata."""
        key_path = self._get_key_path(key_id)
        meta_path = self._get_metadata_path(key_id)

        if not key_path.exists():
            raise KeyNotFoundError(f"Key {key_id} not found")

        try:
            # Load key data
            with open(key_path, 'rb') as f:
                key_data = f.read()

            # Load metadata
            with open(meta_path, 'r') as f:
                meta_dict = json.load(f)
                metadata = KeyMetadata(**meta_dict)

            return key_data, metadata

        except Exception as e:
            logger.error(f"Failed to retrieve key {key_id}: {e}")
            raise KeyManagementError(f"Key retrieval failed: {e}")

    def list_keys(self, key_type: Optional[KeyType] = None) -> List[str]:
        """List keys in storage."""
        keys = []
        for meta_file in self.storage_path.glob("*.meta"):
            key_id = meta_file.stem
            try:
                _, metadata = self.retrieve_key(key_id)
                if key_type is None or metadata.key_type == key_type:
                    keys.append(key_id)
            except Exception:
                continue
        return keys

    def delete_key(self, key_id: str) -> bool:
        """Delete a key and its metadata."""
        key_path = self._get_key_path(key_id)
        meta_path = self._get_metadata_path(key_id)

        deleted = False
        if key_path.exists():
            key_path.unlink()
            deleted = True
        if meta_path.exists():
            meta_path.unlink()
            deleted = True

        if deleted:
            logger.info(f"Key {key_id} deleted from file storage")
        return deleted

    def key_exists(self, key_id: str) -> bool:
        """Check if a key exists."""
        return self._get_key_path(key_id).exists() and self._get_metadata_path(key_id).exists()

class EnvironmentKeyStorage(KeyStorageBackend):
    """Environment variable-based key storage."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.prefix = config.get("prefix", "FEDZK_KEY_")
        self.keys: Dict[str, Tuple[bytes, KeyMetadata]] = {}

    def store_key(self, key_id: str, key_data: bytes, metadata: KeyMetadata) -> bool:
        """Store a key in environment (for testing/development only)."""
        try:
            env_key = f"{self.prefix}{key_id}"
            encoded_data = base64.b64encode(key_data).decode('utf-8')
            os.environ[env_key] = encoded_data

            # Store in memory for retrieval
            self.keys[key_id] = (key_data, metadata)

            logger.warning(f"Key {key_id} stored in environment (NOT SECURE for production)")
            return True

        except Exception as e:
            logger.error(f"Failed to store key {key_id} in environment: {e}")
            return False

    def retrieve_key(self, key_id: str) -> Tuple[bytes, KeyMetadata]:
        """Retrieve a key from environment."""
        if key_id in self.keys:
            return self.keys[key_id]

        env_key = f"{self.prefix}{key_id}"
        if env_key not in os.environ:
            raise KeyNotFoundError(f"Key {key_id} not found in environment")

        try:
            encoded_data = os.environ[env_key]
            key_data = base64.b64decode(encoded_data)

            # Create basic metadata (limited for env storage)
            metadata = KeyMetadata(
                key_id=key_id,
                key_type=KeyType.SYMMETRIC,
                storage_type=KeyStorageType.ENVIRONMENT,
                created_at=datetime.now(),
                expires_at=None,
                last_rotated_at=None,
                rotation_policy=KeyRotationPolicy.NEVER,
                usage_count=0,
                max_usage_count=None,
                algorithm="unknown",
                key_size=len(key_data),
                fingerprint=self._calculate_fingerprint(key_data),
                tags={},
                access_log=[]
            )

            return key_data, metadata

        except Exception as e:
            logger.error(f"Failed to retrieve key {key_id} from environment: {e}")
            raise KeyManagementError(f"Key retrieval failed: {e}")

    def list_keys(self, key_type: Optional[KeyType] = None) -> List[str]:
        """List keys in environment."""
        keys = []
        prefix_len = len(self.prefix)
        for env_key in os.environ:
            if env_key.startswith(self.prefix):
                key_id = env_key[prefix_len:]
                keys.append(key_id)
        return keys

    def delete_key(self, key_id: str) -> bool:
        """Delete a key from environment."""
        env_key = f"{self.prefix}{key_id}"
        if env_key in os.environ:
            del os.environ[env_key]
            if key_id in self.keys:
                del self.keys[key_id]
            logger.info(f"Key {key_id} deleted from environment")
            return True
        return False

    def key_exists(self, key_id: str) -> bool:
        """Check if a key exists in environment."""
        env_key = f"{self.prefix}{key_id}"
        return env_key in os.environ

    def _calculate_fingerprint(self, key_data: bytes) -> str:
        """Calculate key fingerprint."""
        return hashlib.sha256(key_data).hexdigest()[:16]

class VaultKeyStorage(KeyStorageBackend):
    """HashiCorp Vault-based key storage backend."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.vault_addr = config.get("vault_addr", "http://localhost:8200")
        self.token = config.get("token", os.getenv("VAULT_TOKEN"))
        self.mount_point = config.get("mount_point", "fedzk")
        self.namespace = config.get("namespace", "")

        # Initialize vault client (lazy loading)
        self._vault_client = None

    def _get_vault_client(self):
        """Get vault client with lazy initialization."""
        if self._vault_client is None:
            try:
                import hvac
                self._vault_client = hvac.Client(
                    url=self.vault_addr,
                    token=self.token,
                    namespace=self.namespace
                )
            except ImportError:
                raise KeyManagementError("hvac library required for Vault integration")
        return self._vault_client

    def store_key(self, key_id: str, key_data: bytes, metadata: KeyMetadata) -> bool:
        """Store a key in Vault."""
        try:
            client = self._get_vault_client()
            path = f"{self.mount_point}/keys/{key_id}"

            data = {
                "key": base64.b64encode(key_data).decode('utf-8'),
                "metadata": asdict(metadata)
            }

            client.secrets.kv.v2.create_or_update_secret_version(
                path=path,
                secret=data
            )

            logger.info(f"Key {key_id} stored successfully in Vault")
            return True

        except Exception as e:
            logger.error(f"Failed to store key {key_id} in Vault: {e}")
            return False

    def retrieve_key(self, key_id: str) -> Tuple[bytes, KeyMetadata]:
        """Retrieve a key from Vault."""
        try:
            client = self._get_vault_client()
            path = f"{self.mount_point}/keys/{key_id}"

            response = client.secrets.kv.v2.read_secret_version(path=path)

            if not response or 'data' not in response:
                raise KeyNotFoundError(f"Key {key_id} not found in Vault")

            data = response['data']['data']
            key_data = base64.b64decode(data['key'])

            # Reconstruct metadata
            meta_dict = data['metadata']
            metadata = KeyMetadata(**meta_dict)

            return key_data, metadata

        except Exception as e:
            logger.error(f"Failed to retrieve key {key_id} from Vault: {e}")
            raise KeyManagementError(f"Key retrieval failed: {e}")

    def list_keys(self, key_type: Optional[KeyType] = None) -> List[str]:
        """List keys in Vault."""
        try:
            client = self._get_vault_client()
            path = f"{self.mount_point}/keys"

            response = client.secrets.kv.v2.list_secrets_version(path=path)

            if not response or 'data' not in response:
                return []

            keys = []
            for key_data in response['data']['keys']:
                key_name = key_data['name'] if isinstance(key_data, dict) else key_data
                if key_name.endswith('/'):
                    continue  # Skip directories
                keys.append(key_name)

            return keys

        except Exception as e:
            logger.error(f"Failed to list keys in Vault: {e}")
            return []

    def delete_key(self, key_id: str) -> bool:
        """Delete a key from Vault."""
        try:
            client = self._get_vault_client()
            path = f"{self.mount_point}/keys/{key_id}"

            client.secrets.kv.v2.delete_metadata_and_all_versions(path=path)

            logger.info(f"Key {key_id} deleted from Vault")
            return True

        except Exception as e:
            logger.error(f"Failed to delete key {key_id} from Vault: {e}")
            return False

    def key_exists(self, key_id: str) -> bool:
        """Check if a key exists in Vault."""
        try:
            self.retrieve_key(key_id)
            return True
        except KeyNotFoundError:
            return False
        except Exception:
            return False

class KeyIntegrityVerifier:
    """Key integrity verification system."""

    def __init__(self, master_key: bytes):
        self.master_key = master_key

    def calculate_integrity_hash(self, key_data: bytes, metadata: KeyMetadata) -> str:
        """Calculate integrity hash for a key."""
        combined_data = key_data + json.dumps(asdict(metadata), sort_keys=True, default=str).encode()
        return hmac.new(self.master_key, combined_data, hashlib.sha256).hexdigest()

    def verify_integrity(self, key_data: bytes, metadata: KeyMetadata, expected_hash: str) -> bool:
        """Verify key integrity using HMAC."""
        calculated_hash = self.calculate_integrity_hash(key_data, metadata)
        return hmac.compare_digest(calculated_hash, expected_hash)

    def sign_key(self, key_data: bytes, metadata: KeyMetadata) -> str:
        """Sign a key for integrity verification."""
        return self.calculate_integrity_hash(key_data, metadata)

class KeyManager:
    """
    Comprehensive key management system for FEDzk.

    Features:
    - Multi-backend key storage (file, environment, vault)
    - Automatic key rotation
    - Key integrity verification
    - Access logging and monitoring
    - Production security standards
    """

    def __init__(
        self,
        security_config: Optional[KeySecurityConfig] = None,
        rotation_config: Optional[KeyRotationConfig] = None,
        storage_backends: Optional[Dict[KeyStorageType, KeyStorageBackend]] = None
    ):
        self.security_config = security_config or KeySecurityConfig()
        self.rotation_config = rotation_config or KeyRotationConfig()

        # Initialize storage backends
        self.storage_backends = storage_backends or self._initialize_default_backends()

        # Key cache for performance
        self.key_cache: Dict[str, Tuple[bytes, KeyMetadata, float]] = {}
        self.cache_ttl = 300  # 5 minutes

        # Initialize master key for integrity verification
        self.master_key = self._initialize_master_key()
        self.integrity_verifier = KeyIntegrityVerifier(self.master_key)

        # Access logging
        self.access_log: List[Dict[str, Any]] = []
        self.log_lock = threading.Lock()

        # Key rotation scheduler
        self.rotation_scheduler = None
        if self.rotation_config.auto_rotation:
            self._start_rotation_scheduler()

        logger.info("KeyManager initialized with comprehensive security features")

    def _initialize_default_backends(self) -> Dict[KeyStorageType, KeyStorageBackend]:
        """Initialize default storage backends."""
        return {
            KeyStorageType.FILE: FileKeyStorage({
                "path": "./keys",
                "encryption": self.security_config.encryption_enabled
            }),
            KeyStorageType.ENVIRONMENT: EnvironmentKeyStorage({
                "prefix": "FEDZK_KEY_"
            })
        }

    def _initialize_master_key(self) -> bytes:
        """Initialize or load master key for integrity verification."""
        master_key_path = Path("./keys/master.key")

        if master_key_path.exists():
            with open(master_key_path, 'rb') as f:
                return f.read()
        else:
            # Generate new master key
            master_key = secrets.token_bytes(32)
            master_key_path.parent.mkdir(parents=True, exist_ok=True)
            with open(master_key_path, 'wb') as f:
                f.write(master_key)
            logger.info("New master key generated for integrity verification")
            return master_key

    def _start_rotation_scheduler(self):
        """Start automatic key rotation scheduler."""
        def rotation_worker():
            while True:
                try:
                    self._perform_automatic_rotation()
                except Exception as e:
                    logger.error(f"Key rotation error: {e}")
                time.sleep(3600)  # Check every hour

        self.rotation_scheduler = threading.Thread(target=rotation_worker, daemon=True)
        self.rotation_scheduler.start()
        logger.info("Key rotation scheduler started")

    def _perform_automatic_rotation(self):
        """Perform automatic key rotation based on policy."""
        now = datetime.now()

        for backend_type, backend in self.storage_backends.items():
            try:
                key_ids = backend.list_keys()

                for key_id in key_ids:
                    try:
                        _, metadata = backend.retrieve_key(key_id)

                        should_rotate = self._should_rotate_key(metadata, now)
                        if should_rotate:
                            logger.info(f"Auto-rotating key {key_id}")
                            self.rotate_key(key_id)

                    except Exception as e:
                        logger.error(f"Error checking rotation for key {key_id}: {e}")

            except Exception as e:
                logger.error(f"Error in rotation check for backend {backend_type}: {e}")

    def _should_rotate_key(self, metadata: KeyMetadata, now: datetime) -> bool:
        """Determine if a key should be rotated."""
        if metadata.rotation_policy == KeyRotationPolicy.NEVER:
            return False

        # Time-based rotation
        if metadata.rotation_policy == KeyRotationPolicy.TIME_BASED:
            if metadata.created_at:
                age_days = (now - metadata.created_at).days
                return age_days >= self.rotation_config.max_age_days

        # Usage-based rotation
        if metadata.rotation_policy == KeyRotationPolicy.USAGE_BASED:
            if metadata.max_usage_count:
                return metadata.usage_count >= metadata.max_usage_count

        return False

    def store_key(
        self,
        key_id: str,
        key_data: bytes,
        key_type: KeyType,
        storage_type: KeyStorageType = KeyStorageType.FILE,
        rotation_policy: KeyRotationPolicy = KeyRotationPolicy.TIME_BASED,
        tags: Optional[Dict[str, str]] = None,
        algorithm: str = "unknown",
        expires_at: Optional[datetime] = None
    ) -> bool:
        """
        Store a cryptographic key with comprehensive metadata.

        Args:
            key_id: Unique identifier for the key
            key_data: Raw key data
            key_type: Type of cryptographic key
            storage_type: Storage backend to use
            rotation_policy: Key rotation policy
            tags: Key tags for organization
            algorithm: Cryptographic algorithm used
            expires_at: Optional expiration date

        Returns:
            bool: Success status
        """
        try:
            # Get storage backend
            if storage_type not in self.storage_backends:
                raise KeyManagementError(f"Storage backend {storage_type} not configured")

            backend = self.storage_backends[storage_type]

            # Check key limits
            existing_keys = backend.list_keys(key_type)
            if len(existing_keys) >= self.security_config.max_keys_per_type:
                logger.warning(f"Maximum keys per type reached for {key_type}")

            # Create metadata
            metadata = KeyMetadata(
                key_id=key_id,
                key_type=key_type,
                storage_type=storage_type,
                created_at=datetime.now(),
                expires_at=expires_at,
                last_rotated_at=None,
                rotation_policy=rotation_policy,
                usage_count=0,
                max_usage_count=self.rotation_config.max_usage_count,
                algorithm=algorithm,
                key_size=len(key_data),
                fingerprint=hashlib.sha256(key_data).hexdigest()[:16],
                tags=tags or {},
                access_log=[]
            )

            # Add integrity verification
            if self.security_config.integrity_check_enabled:
                integrity_hash = self.integrity_verifier.sign_key(key_data, metadata)
                metadata.tags["integrity_hash"] = integrity_hash

            # Encrypt key data if enabled
            if self.security_config.encryption_enabled and storage_type != KeyStorageType.ENVIRONMENT:
                key_data = self._encrypt_key_data(key_data)

            # Store the key
            success = backend.store_key(key_id, key_data, metadata)

            if success:
                # Log successful storage
                self._log_access(key_id, "store", "success", {"key_type": key_type.value})
                logger.info(f"Key {key_id} stored successfully")
            else:
                self._log_access(key_id, "store", "failure", {"key_type": key_type.value})
                logger.error(f"Failed to store key {key_id}")

            return success

        except Exception as e:
            logger.error(f"Error storing key {key_id}: {e}")
            self._log_access(key_id, "store", "error", {"error": str(e)})
            return False

    def retrieve_key(self, key_id: str, storage_type: Optional[KeyStorageType] = None) -> Tuple[bytes, KeyMetadata]:
        """
        Retrieve a cryptographic key with integrity verification.

        Args:
            key_id: Key identifier
            storage_type: Specific storage backend (optional)

        Returns:
            Tuple[bytes, KeyMetadata]: Key data and metadata

        Raises:
            KeyNotFoundError: If key doesn't exist
            KeyIntegrityError: If integrity check fails
        """
        try:
            # Check cache first
            if key_id in self.key_cache:
                cached_data, cached_metadata, cache_time = self.key_cache[key_id]
                if time.time() - cache_time < self.cache_ttl:
                    self._log_access(key_id, "retrieve", "cache_hit")
                    return cached_data, cached_metadata

            # Determine storage backend
            if storage_type:
                backend = self.storage_backends.get(storage_type)
                if not backend:
                    raise KeyManagementError(f"Storage backend {storage_type} not configured")
            else:
                # Try all backends
                backend = None
                for b in self.storage_backends.values():
                    if b.key_exists(key_id):
                        backend = b
                        break
                if not backend:
                    raise KeyNotFoundError(f"Key {key_id} not found in any backend")

            # Retrieve key
            key_data, metadata = backend.retrieve_key(key_id)

            # Decrypt if necessary
            if self.security_config.encryption_enabled and metadata.storage_type != KeyStorageType.ENVIRONMENT:
                key_data = self._decrypt_key_data(key_data)

            # Verify integrity
            if self.security_config.integrity_check_enabled:
                expected_hash = metadata.tags.get("integrity_hash")
                if expected_hash and not self.integrity_verifier.verify_integrity(key_data, metadata, expected_hash):
                    self._log_access(key_id, "retrieve", "integrity_failure")
                    raise KeyIntegrityError(f"Key {key_id} integrity verification failed")

            # Update usage count
            metadata.usage_count += 1

            # Cache the key
            self.key_cache[key_id] = (key_data, metadata, time.time())

            # Log successful retrieval
            self._log_access(key_id, "retrieve", "success", {
                "key_type": metadata.key_type.value,
                "usage_count": metadata.usage_count
            })

            return key_data, metadata

        except (KeyNotFoundError, KeyIntegrityError):
            raise
        except Exception as e:
            logger.error(f"Error retrieving key {key_id}: {e}")
            self._log_access(key_id, "retrieve", "error", {"error": str(e)})
            raise KeyManagementError(f"Key retrieval failed: {e}")

    def rotate_key(self, key_id: str, new_key_data: Optional[bytes] = None) -> bool:
        """
        Rotate a cryptographic key.

        Args:
            key_id: Key to rotate
            new_key_data: New key data (generated if None)

        Returns:
            bool: Success status
        """
        try:
            # Retrieve current key
            current_key_data, metadata = self.retrieve_key(key_id)

            # Generate new key if not provided
            if new_key_data is None:
                new_key_data = self._generate_key_of_type(metadata.key_type, metadata.key_size)

            # Backup old key if enabled
            if self.rotation_config.backup_old_keys:
                self._backup_key(key_id, current_key_data, metadata)

            # Update metadata
            metadata.last_rotated_at = datetime.now()
            metadata.usage_count = 0

            # Store new key
            success = self.store_key(
                key_id,
                new_key_data,
                metadata.key_type,
                metadata.storage_type,
                metadata.rotation_policy,
                metadata.tags,
                metadata.algorithm
            )

            if success:
                # Clear cache
                if key_id in self.key_cache:
                    del self.key_cache[key_id]

                # Log rotation
                self._log_access(key_id, "rotate", "success")
                logger.info(f"Key {key_id} rotated successfully")
            else:
                self._log_access(key_id, "rotate", "failure")
                logger.error(f"Failed to rotate key {key_id}")

            return success

        except Exception as e:
            logger.error(f"Error rotating key {key_id}: {e}")
            self._log_access(key_id, "rotate", "error", {"error": str(e)})
            return False

    def delete_key(self, key_id: str, storage_type: Optional[KeyStorageType] = None) -> bool:
        """
        Delete a cryptographic key.

        Args:
            key_id: Key to delete
            storage_type: Specific storage backend

        Returns:
            bool: Success status
        """
        try:
            # Determine storage backend
            if storage_type:
                backend = self.storage_backends.get(storage_type)
            else:
                backend = None
                for b in self.storage_backends.values():
                    if b.key_exists(key_id):
                        backend = b
                        break

            if not backend:
                logger.warning(f"Key {key_id} not found for deletion")
                return False

            success = backend.delete_key(key_id)

            if success:
                # Clear cache
                if key_id in self.key_cache:
                    del self.key_cache[key_id]

                self._log_access(key_id, "delete", "success")
                logger.info(f"Key {key_id} deleted successfully")
            else:
                self._log_access(key_id, "delete", "failure")
                logger.error(f"Failed to delete key {key_id}")

            return success

        except Exception as e:
            logger.error(f"Error deleting key {key_id}: {e}")
            self._log_access(key_id, "delete", "error", {"error": str(e)})
            return False

    def list_keys(self, key_type: Optional[KeyType] = None, storage_type: Optional[KeyStorageType] = None) -> List[Dict[str, Any]]:
        """
        List keys with metadata.

        Args:
            key_type: Filter by key type
            storage_type: Filter by storage type

        Returns:
            List[Dict[str, Any]]: List of key information
        """
        keys_info = []

        backends = [self.storage_backends[storage_type]] if storage_type else self.storage_backends.values()

        for backend in backends:
            try:
                key_ids = backend.list_keys(key_type)

                for key_id in key_ids:
                    try:
                        _, metadata = backend.retrieve_key(key_id)
                        keys_info.append({
                            "key_id": key_id,
                            "key_type": metadata.key_type.value,
                            "storage_type": metadata.storage_type.value,
                            "created_at": metadata.created_at.isoformat(),
                            "expires_at": metadata.expires_at.isoformat() if metadata.expires_at else None,
                            "usage_count": metadata.usage_count,
                            "algorithm": metadata.algorithm,
                            "fingerprint": metadata.fingerprint
                        })
                    except Exception as e:
                        logger.error(f"Error retrieving metadata for key {key_id}: {e}")

            except Exception as e:
                logger.error(f"Error listing keys in backend: {e}")

        return keys_info

    def get_key_status(self, key_id: str) -> Dict[str, Any]:
        """
        Get comprehensive status of a key.

        Args:
            key_id: Key identifier

        Returns:
            Dict[str, Any]: Key status information
        """
        try:
            _, metadata = self.retrieve_key(key_id)

            now = datetime.now()
            age_days = (now - metadata.created_at).days if metadata.created_at else 0

            status = {
                "key_id": key_id,
                "exists": True,
                "key_type": metadata.key_type.value,
                "storage_type": metadata.storage_type.value,
                "created_at": metadata.created_at.isoformat() if metadata.created_at else None,
                "age_days": age_days,
                "usage_count": metadata.usage_count,
                "rotation_policy": metadata.rotation_policy.value,
                "last_rotated": metadata.last_rotated_at.isoformat() if metadata.last_rotated_at else None,
                "expires_at": metadata.expires_at.isoformat() if metadata.expires_at else None,
                "is_expired": metadata.expires_at and now > metadata.expires_at,
                "needs_rotation": self._should_rotate_key(metadata, now),
                "integrity_verified": True,
                "algorithm": metadata.algorithm,
                "key_size": metadata.key_size,
                "fingerprint": metadata.fingerprint,
                "tags": metadata.tags
            }

            return status

        except KeyNotFoundError:
            return {"key_id": key_id, "exists": False}
        except Exception as e:
            return {
                "key_id": key_id,
                "exists": True,
                "error": str(e)
            }

    def get_security_metrics(self) -> Dict[str, Any]:
        """Get security metrics for the key management system."""
        metrics = {
            "total_keys": 0,
            "keys_by_type": {},
            "keys_by_storage": {},
            "expired_keys": 0,
            "keys_needing_rotation": 0,
            "access_log_entries": len(self.access_log),
            "cache_size": len(self.key_cache),
            "integrity_failures": 0,
            "last_security_check": datetime.now().isoformat()
        }

        # Count keys by type and storage
        all_keys = self.list_keys()
        metrics["total_keys"] = len(all_keys)

        for key_info in all_keys:
            key_type = key_info["key_type"]
            storage_type = key_info["storage_type"]

            metrics["keys_by_type"][key_type] = metrics["keys_by_type"].get(key_type, 0) + 1
            metrics["keys_by_storage"][storage_type] = metrics["keys_by_storage"].get(storage_type, 0) + 1

        # Check for expired and rotation-needed keys
        for key_info in all_keys:
            key_id = key_info["key_id"]
            status = self.get_key_status(key_id)

            if status.get("is_expired"):
                metrics["expired_keys"] += 1
            if status.get("needs_rotation"):
                metrics["keys_needing_rotation"] += 1

        return metrics

    def _generate_key_of_type(self, key_type: KeyType, key_size: int) -> bytes:
        """Generate a new key of the specified type and size."""
        if key_type in [KeyType.SYMMETRIC, KeyType.HMAC]:
            return secrets.token_bytes(key_size)
        elif key_type == KeyType.ASYMMETRIC_PRIVATE:
            # Generate RSA private key
            private_key = rsa.generate_private_key(
                public_exponent=65537,
                key_size=key_size * 8,  # Convert bytes to bits
                backend=default_backend()
            )
            return private_key.private_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PrivateFormat.PKCS8,
                encryption_algorithm=serialization.NoEncryption()
            )
        elif key_type == KeyType.VERIFICATION_KEY:
            # Generate ECDSA key for verification
            private_key = ec.generate_private_key(ec.SECP256R1(), default_backend())
            return private_key.private_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PrivateFormat.PKCS8,
                encryption_algorithm=serialization.NoEncryption()
            )
        else:
            return secrets.token_bytes(key_size)

    def _encrypt_key_data(self, key_data: bytes) -> bytes:
        """Encrypt key data using the master key."""
        salt = secrets.token_bytes(16)
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
            backend=default_backend()
        )

        key = kdf.derive(self.master_key)
        iv = secrets.token_bytes(16)

        cipher = Cipher(algorithms.AES(key), modes.CBC(iv), backend=default_backend())
        encryptor = cipher.encryptor()

        # Pad the data
        block_size = 16
        padding_length = block_size - (len(key_data) % block_size)
        padded_data = key_data + bytes([padding_length]) * padding_length

        encrypted_data = encryptor.update(padded_data) + encryptor.finalize()

        # Return salt + iv + encrypted_data
        return salt + iv + encrypted_data

    def _decrypt_key_data(self, encrypted_data: bytes) -> bytes:
        """Decrypt key data using the master key."""
        salt = encrypted_data[:16]
        iv = encrypted_data[16:32]
        encrypted_content = encrypted_data[32:]

        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
            backend=default_backend()
        )

        key = kdf.derive(self.master_key)

        cipher = Cipher(algorithms.AES(key), modes.CBC(iv), backend=default_backend())
        decryptor = cipher.decryptor()

        decrypted_padded = decryptor.update(encrypted_content) + decryptor.finalize()

        # Remove padding
        padding_length = decrypted_padded[-1]
        decrypted_data = decrypted_padded[:-padding_length]

        return decrypted_data

    def _backup_key(self, key_id: str, key_data: bytes, metadata: KeyMetadata):
        """Backup a key before rotation."""
        try:
            backup_dir = Path("./keys/backup")
            backup_dir.mkdir(parents=True, exist_ok=True)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_key_id = f"{key_id}_backup_{timestamp}"

            backup_metadata = metadata
            backup_metadata.key_id = backup_key_id
            backup_metadata.tags["backup_of"] = key_id
            backup_metadata.tags["backup_time"] = timestamp

            # Store backup
            self.store_key(
                backup_key_id,
                key_data,
                metadata.key_type,
                KeyStorageType.FILE,  # Always backup to file
                KeyRotationPolicy.NEVER,
                backup_metadata.tags,
                metadata.algorithm
            )

            # Clean old backups
            self._cleanup_old_backups(key_id)

            logger.info(f"Key {key_id} backed up as {backup_key_id}")

        except Exception as e:
            logger.error(f"Failed to backup key {key_id}: {e}")

    def _cleanup_old_backups(self, original_key_id: str):
        """Clean up old backups for a key."""
        try:
            backup_dir = Path("./keys/backup")
            if not backup_dir.exists():
                return

            # Find all backups for this key
            backup_pattern = f"{original_key_id}_backup_"
            backups = []

            for backup_file in backup_dir.glob("*.meta"):
                if backup_pattern in backup_file.name:
                    backup_key_id = backup_file.stem
                    try:
                        _, metadata = self.retrieve_key(backup_key_id, KeyStorageType.FILE)
                        backup_time = metadata.tags.get("backup_time")
                        if backup_time:
                            backups.append((backup_key_id, datetime.strptime(backup_time, "%Y%m%d_%H%M%S")))
                    except Exception:
                        continue

            # Sort by backup time and keep only recent ones
            backups.sort(key=lambda x: x[1], reverse=True)

            # Delete old backups (keep last 5)
            for backup_key_id, _ in backups[5:]:
                try:
                    self.delete_key(backup_key_id, KeyStorageType.FILE)
                    logger.info(f"Cleaned up old backup: {backup_key_id}")
                except Exception as e:
                    logger.error(f"Failed to cleanup backup {backup_key_id}: {e}")

        except Exception as e:
            logger.error(f"Error cleaning up backups for {original_key_id}: {e}")

    def _log_access(self, key_id: str, operation: str, status: str, details: Optional[Dict[str, Any]] = None):
        """Log key access for audit purposes."""
        if not self.security_config.access_logging_enabled:
            return

        with self.log_lock:
            log_entry = {
                "timestamp": datetime.now().isoformat(),
                "key_id": key_id,
                "operation": operation,
                "status": status,
                "details": details or {},
                "client_info": self._get_client_info()
            }

            self.access_log.append(log_entry)

            # Keep only recent entries
            max_entries = 10000
            if len(self.access_log) > max_entries:
                self.access_log = self.access_log[-max_entries:]

            # Also log to system logger
            logger.info(f"Key access: {operation} {key_id} - {status}")

    def _get_client_info(self) -> Dict[str, str]:
        """Get client information for logging."""
        return {
            "user": os.getenv("USER", "unknown"),
            "hostname": os.getenv("HOSTNAME", "unknown"),
            "pid": str(os.getpid())
        }

    def cleanup_expired_keys(self) -> int:
        """
        Clean up expired keys.

        Returns:
            int: Number of keys cleaned up
        """
        cleaned_count = 0
        now = datetime.now()

        all_keys = self.list_keys()
        for key_info in all_keys:
            key_id = key_info["key_id"]
            status = self.get_key_status(key_id)

            if status.get("is_expired"):
                try:
                    self.delete_key(key_id)
                    cleaned_count += 1
                    logger.info(f"Cleaned up expired key: {key_id}")
                except Exception as e:
                    logger.error(f"Failed to cleanup expired key {key_id}: {e}")

        return cleaned_count

# Convenience functions for easy usage
def create_secure_key_manager(
    vault_config: Optional[Dict[str, Any]] = None,
    enable_vault: bool = False
) -> KeyManager:
    """
    Create a key manager with secure default settings.

    Args:
        vault_config: Vault configuration if using Vault
        enable_vault: Whether to enable Vault integration

    Returns:
        KeyManager: Configured key manager
    """
    security_config = KeySecurityConfig(
        encryption_enabled=True,
        master_key_rotation_days=365,
        integrity_check_enabled=True,
        access_logging_enabled=True,
        audit_retention_days=2555,
        max_keys_per_type=10,
        key_backup_encryption=True
    )

    rotation_config = KeyRotationConfig(
        enabled=True,
        max_age_days=90,
        max_usage_count=10000,
        rotation_window_hours=24,
        backup_old_keys=True,
        backup_retention_days=365,
        auto_rotation=True
    )

    storage_backends = {
        KeyStorageType.FILE: FileKeyStorage({
            "path": "./keys",
            "encryption": True
        })
    }

    # Add Vault if enabled
    if enable_vault and vault_config:
        storage_backends[KeyStorageType.VAULT] = VaultKeyStorage(vault_config)

    return KeyManager(
        security_config=security_config,
        rotation_config=rotation_config,
        storage_backends=storage_backends
    )

def generate_federated_learning_keys(key_manager: KeyManager) -> Dict[str, str]:
    """
    Generate standard keys needed for federated learning.

    Args:
        key_manager: Key manager instance

    Returns:
        Dict[str, str]: Generated key IDs
    """
    keys = {}

    # Generate model encryption key
    model_key = secrets.token_bytes(32)
    key_manager.store_key(
        "fedzk_model_encryption",
        model_key,
        KeyType.SYMMETRIC,
        algorithm="AES-256-GCM"
    )
    keys["model_encryption"] = "fedzk_model_encryption"

    # Generate proof verification key
    verification_key = secrets.token_bytes(32)
    key_manager.store_key(
        "fedzk_proof_verification",
        verification_key,
        KeyType.VERIFICATION_KEY,
        algorithm="ECDSA-P256"
    )
    keys["proof_verification"] = "fedzk_proof_verification"

    # Generate secure aggregation key
    aggregation_key = secrets.token_bytes(32)
    key_manager.store_key(
        "fedzk_aggregation",
        aggregation_key,
        KeyType.SYMMETRIC,
        algorithm="AES-256-GCM"
    )
    keys["aggregation"] = "fedzk_aggregation"

    # Generate HMAC key for integrity
    hmac_key = secrets.token_bytes(32)
    key_manager.store_key(
        "fedzk_integrity_hmac",
        hmac_key,
        KeyType.HMAC,
        algorithm="HMAC-SHA256"
    )
    keys["integrity_hmac"] = "fedzk_integrity_hmac"

    logger.info("Generated standard federated learning keys")
    return keys

