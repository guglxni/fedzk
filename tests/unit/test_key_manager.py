#!/usr/bin/env python3
"""
Tests for Key Management System
===============================

Comprehensive test suite for FEDzk key management system.
Tests secure key storage, rotation, integrity verification,
and enterprise-grade key management features.
"""

import unittest
import tempfile
import shutil
import json
import time
from pathlib import Path
from unittest.mock import patch, MagicMock
from datetime import datetime, timedelta

from fedzk.security.key_manager import (
    KeyManager,
    KeyType,
    KeyStorageType,
    KeyRotationPolicy,
    KeyMetadata,
    KeySecurityConfig,
    KeyRotationConfig,
    KeyManagementError,
    KeyNotFoundError,
    KeyIntegrityError,
    FileKeyStorage,
    EnvironmentKeyStorage,
    VaultKeyStorage,
    create_secure_key_manager,
    generate_federated_learning_keys
)

class TestKeyManager(unittest.TestCase):
    """Test KeyManager class."""

    def setUp(self):
        """Setup test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.keys_dir = Path(self.temp_dir) / "keys"

        # Create key manager with file storage
        security_config = KeySecurityConfig(
            encryption_enabled=False,  # Disable for testing
            integrity_check_enabled=True,
            access_logging_enabled=True
        )

        storage_backends = {
            KeyStorageType.FILE: FileKeyStorage({
                "path": str(self.keys_dir),
                "encryption": False
            })
        }

        self.key_manager = KeyManager(
            security_config=security_config,
            rotation_config=KeyRotationConfig(auto_rotation=False),  # Disable auto-rotation for tests
            storage_backends=storage_backends
        )

    def tearDown(self):
        """Cleanup test environment."""
        shutil.rmtree(self.temp_dir)

    def test_key_storage_and_retrieval(self):
        """Test basic key storage and retrieval."""
        key_id = "test_key"
        key_data = b"test_key_data_12345"
        key_type = KeyType.SYMMETRIC

        # Store key
        success = self.key_manager.store_key(
            key_id, key_data, key_type,
            tags={"test": "true"}
        )
        self.assertTrue(success)

        # Retrieve key
        retrieved_data, metadata = self.key_manager.retrieve_key(key_id)

        self.assertEqual(retrieved_data, key_data)
        self.assertEqual(metadata.key_id, key_id)
        self.assertEqual(metadata.key_type, key_type)
        self.assertEqual(metadata.tags["test"], "true")
        self.assertEqual(metadata.usage_count, 1)

    def test_key_not_found(self):
        """Test handling of non-existent keys."""
        with self.assertRaises(KeyNotFoundError):
            self.key_manager.retrieve_key("nonexistent_key")

    def test_key_rotation(self):
        """Test key rotation functionality."""
        key_id = "rotation_test_key"
        original_data = b"original_key_data"

        # Store initial key
        success = self.key_manager.store_key(
            key_id, original_data, KeyType.SYMMETRIC
        )
        self.assertTrue(success)

        # Rotate key
        new_data = b"new_rotated_key_data"
        rotation_success = self.key_manager.rotate_key(key_id, new_data)
        self.assertTrue(rotation_success)

        # Verify rotated key
        retrieved_data, metadata = self.key_manager.retrieve_key(key_id)
        self.assertEqual(retrieved_data, new_data)
        self.assertIsNotNone(metadata.last_rotated_at)
        self.assertEqual(metadata.usage_count, 1)  # Reset after rotation

    def test_key_deletion(self):
        """Test key deletion."""
        key_id = "delete_test_key"
        key_data = b"test_data_for_deletion"

        # Store key
        self.key_manager.store_key(key_id, key_data, KeyType.SYMMETRIC)

        # Verify key exists
        self.assertIsNotNone(self.key_manager.get_key_status(key_id)["exists"])

        # Delete key
        success = self.key_manager.delete_key(key_id)
        self.assertTrue(success)

        # Verify key is gone
        status = self.key_manager.get_key_status(key_id)
        self.assertFalse(status["exists"])

    def test_key_integrity_verification(self):
        """Test key integrity verification."""
        key_id = "integrity_test_key"
        key_data = b"integrity_test_data"

        # Store key with integrity check
        success = self.key_manager.store_key(
            key_id, key_data, KeyType.SYMMETRIC
        )
        self.assertTrue(success)

        # Retrieve and verify integrity
        retrieved_data, metadata = self.key_manager.retrieve_key(key_id)
        self.assertEqual(retrieved_data, key_data)

        # Verify integrity hash exists
        self.assertIn("integrity_hash", metadata.tags)

    def test_key_listing(self):
        """Test key listing functionality."""
        # Store multiple keys
        test_keys = [
            ("key1", b"data1", KeyType.SYMMETRIC),
            ("key2", b"data2", KeyType.VERIFICATION_KEY),
            ("key3", b"data3", KeyType.SYMMETRIC)
        ]

        for key_id, data, key_type in test_keys:
            self.key_manager.store_key(key_id, data, key_type)

        # List all keys
        all_keys = self.key_manager.list_keys()
        self.assertEqual(len(all_keys), 3)

        # List symmetric keys only
        symmetric_keys = self.key_manager.list_keys(KeyType.SYMMETRIC)
        self.assertEqual(len(symmetric_keys), 2)

        # Verify key information
        key_info = next(k for k in all_keys if k["key_id"] == "key1")
        self.assertEqual(key_info["key_type"], "symmetric")

    def test_key_status(self):
        """Test key status retrieval."""
        key_id = "status_test_key"
        key_data = b"status_test_data"

        # Store key
        self.key_manager.store_key(
            key_id, key_data, KeyType.SYMMETRIC,
            tags={"environment": "test"}
        )

        # Get status
        status = self.key_manager.get_key_status(key_id)

        self.assertTrue(status["exists"])
        self.assertEqual(status["key_id"], key_id)
        self.assertEqual(status["key_type"], "symmetric")
        self.assertEqual(status["usage_count"], 0)  # Not retrieved yet
        self.assertEqual(status["tags"]["environment"], "test")
        self.assertFalse(status["is_expired"])
        self.assertFalse(status["needs_rotation"])

    def test_access_logging(self):
        """Test access logging functionality."""
        key_id = "logging_test_key"
        key_data = b"logging_test_data"

        # Store key
        self.key_manager.store_key(key_id, key_data, KeyType.SYMMETRIC)

        # Retrieve key multiple times
        self.key_manager.retrieve_key(key_id)
        self.key_manager.retrieve_key(key_id)

        # Check access log
        self.assertGreater(len(self.key_manager.access_log), 0)

        # Verify log entries
        store_entries = [e for e in self.key_manager.access_log if e["operation"] == "store"]
        retrieve_entries = [e for e in self.key_manager.access_log if e["operation"] == "retrieve"]

        self.assertEqual(len(store_entries), 1)
        self.assertEqual(len(retrieve_entries), 2)

    def test_security_metrics(self):
        """Test security metrics collection."""
        # Store test keys
        self.key_manager.store_key("metric_key1", b"data1", KeyType.SYMMETRIC)
        self.key_manager.store_key("metric_key2", b"data2", KeyType.VERIFICATION_KEY)

        # Get metrics
        metrics = self.key_manager.get_security_metrics()

        self.assertGreater(metrics["total_keys"], 0)
        self.assertIn("symmetric", metrics["keys_by_type"])
        self.assertIn("verification_key", metrics["keys_by_type"])
        self.assertGreater(metrics["access_log_entries"], 0)
        self.assertIn("last_security_check", metrics)

    def test_key_rotation_policy(self):
        """Test key rotation policies."""
        # Test time-based rotation
        key_id = "time_rotation_key"
        key_data = b"time_rotation_data"

        # Create key with time-based rotation
        self.key_manager.store_key(
            key_id, key_data, KeyType.SYMMETRIC,
            rotation_policy=KeyRotationPolicy.TIME_BASED
        )

        # Manually set old creation date to trigger rotation
        _, metadata = self.key_manager.retrieve_key(key_id)
        metadata.created_at = datetime.now() - timedelta(days=100)  # Older than 90 days

        # Check if rotation is needed
        status = self.key_manager.get_key_status(key_id)
        self.assertTrue(status["needs_rotation"])

    def test_key_backup_and_cleanup(self):
        """Test key backup and cleanup functionality."""
        key_id = "backup_test_key"
        original_data = b"original_backup_data"

        # Store initial key
        self.key_manager.store_key(key_id, original_data, KeyType.SYMMETRIC)

        # Rotate key (should create backup)
        new_data = b"new_backup_data"
        self.key_manager.rotate_key(key_id, new_data)

        # Verify new key
        retrieved_data, _ = self.key_manager.retrieve_key(key_id)
        self.assertEqual(retrieved_data, new_data)

        # Check that backup exists
        backup_keys = [k for k in self.key_manager.list_keys() if "backup" in k["key_id"]]
        self.assertGreater(len(backup_keys), 0)

    def test_cleanup_expired_keys(self):
        """Test cleanup of expired keys."""
        # Create expired key
        expired_key_id = "expired_test_key"
        key_data = b"expired_data"

        expired_time = datetime.now() - timedelta(days=1)
        self.key_manager.store_key(
            expired_key_id, key_data, KeyType.SYMMETRIC,
            expires_at=expired_time
        )

        # Create valid key
        valid_key_id = "valid_test_key"
        self.key_manager.store_key(
            valid_key_id, key_data, KeyType.SYMMETRIC,
            expires_at=datetime.now() + timedelta(days=1)
        )

        # Cleanup expired keys
        cleaned_count = self.key_manager.cleanup_expired_keys()

        # Verify expired key is gone
        expired_status = self.key_manager.get_key_status(expired_key_id)
        self.assertFalse(expired_status["exists"])

        # Verify valid key still exists
        valid_status = self.key_manager.get_key_status(valid_key_id)
        self.assertTrue(valid_status["exists"])

        self.assertEqual(cleaned_count, 1)

class TestFileKeyStorage(unittest.TestCase):
    """Test FileKeyStorage backend."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.storage = FileKeyStorage({
            "path": self.temp_dir,
            "encryption": False
        })

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def test_file_storage_operations(self):
        """Test file storage operations."""
        key_id = "file_test_key"
        key_data = b"file_storage_test_data"
        metadata = KeyMetadata(
            key_id=key_id,
            key_type=KeyType.SYMMETRIC,
            storage_type=KeyStorageType.FILE,
            created_at=datetime.now(),
            expires_at=None,
            last_rotated_at=None,
            rotation_policy=KeyRotationPolicy.NEVER,
            usage_count=0,
            max_usage_count=None,
            algorithm="AES-256",
            key_size=32,
            fingerprint="test_fingerprint",
            tags={},
            access_log=[]
        )

        # Store key
        success = self.storage.store_key(key_id, key_data, metadata)
        self.assertTrue(success)

        # Verify files exist
        key_path = Path(self.temp_dir) / f"{key_id}.key"
        meta_path = Path(self.temp_dir) / f"{key_id}.meta"
        self.assertTrue(key_path.exists())
        self.assertTrue(meta_path.exists())

        # Retrieve key
        retrieved_data, retrieved_metadata = self.storage.retrieve_key(key_id)
        self.assertEqual(retrieved_data, key_data)
        self.assertEqual(retrieved_metadata.key_id, key_id)

        # List keys
        keys = self.storage.list_keys()
        self.assertIn(key_id, keys)

        # Delete key
        success = self.storage.delete_key(key_id)
        self.assertTrue(success)
        self.assertFalse(key_path.exists())
        self.assertFalse(meta_path.exists())

class TestEnvironmentKeyStorage(unittest.TestCase):
    """Test EnvironmentKeyStorage backend."""

    def setUp(self):
        self.storage = EnvironmentKeyStorage({
            "prefix": "TEST_FEDZK_KEY_"
        })

    def tearDown(self):
        # Clean up test environment variables
        test_vars = [k for k in os.environ.keys() if k.startswith("TEST_FEDZK_KEY_")]
        for var in test_vars:
            del os.environ[var]

    def test_environment_storage_operations(self):
        """Test environment storage operations."""
        key_id = "env_test_key"
        key_data = b"environment_storage_test_data"

        # Store key
        success = self.storage.store_key(key_id, key_data, None)
        self.assertTrue(success)

        # Verify environment variable exists
        env_key = f"TEST_FEDZK_KEY_{key_id}"
        self.assertIn(env_key, os.environ)

        # Retrieve key
        retrieved_data, metadata = self.storage.retrieve_key(key_id)
        self.assertEqual(retrieved_data, key_data)

        # List keys
        keys = self.storage.list_keys()
        self.assertIn(key_id, keys)

        # Delete key
        success = self.storage.delete_key(key_id)
        self.assertTrue(success)
        self.assertNotIn(env_key, os.environ)

class TestVaultKeyStorage(unittest.TestCase):
    """Test VaultKeyStorage backend."""

    @patch('fedzk.security.key_manager.hvac')
    def test_vault_storage_operations(self, mock_hvac):
        """Test vault storage operations with mocked client."""
        # Mock vault client
        mock_client = MagicMock()
        mock_hvac.Client.return_value = mock_client

        # Mock vault responses
        mock_response = {
            'data': {
                'data': {
                    'key': 'dGVzdF9rZXlfZGF0YQ==',  # base64 encoded "test_key_data"
                    'metadata': {
                        'key_id': 'vault_test_key',
                        'key_type': 'symmetric',
                        'storage_type': 'vault',
                        'created_at': '2024-01-01T00:00:00',
                        'rotation_policy': 'time_based',
                        'usage_count': 0,
                        'algorithm': 'AES-256',
                        'key_size': 32,
                        'fingerprint': 'test_fingerprint',
                        'tags': '{}',
                        'access_log': '[]'
                    }
                }
            }
        }
        mock_client.secrets.kv.v2.read_secret_version.return_value = mock_response

        storage = VaultKeyStorage({
            "vault_addr": "http://localhost:8200",
            "token": "test_token",
            "mount_point": "fedzk"
        })

        # Test retrieval
        key_data, metadata = storage.retrieve_key("vault_test_key")
        self.assertEqual(key_data, b"test_key_data")
        self.assertEqual(metadata.key_id, "vault_test_key")

class TestConvenienceFunctions(unittest.TestCase):
    """Test convenience functions."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def test_create_secure_key_manager(self):
        """Test create_secure_key_manager function."""
        key_manager = create_secure_key_manager()

        self.assertIsInstance(key_manager, KeyManager)
        self.assertTrue(key_manager.security_config.encryption_enabled)
        self.assertTrue(key_manager.security_config.integrity_check_enabled)
        self.assertTrue(key_manager.security_config.access_logging_enabled)

    def test_generate_federated_learning_keys(self):
        """Test generate_federated_learning_keys function."""
        # Create key manager with temp directory
        keys_dir = Path(self.temp_dir) / "keys"
        storage_backends = {
            KeyStorageType.FILE: FileKeyStorage({
                "path": str(keys_dir),
                "encryption": False
            })
        }

        key_manager = KeyManager(
            security_config=KeySecurityConfig(integrity_check_enabled=False),
            storage_backends=storage_backends
        )

        # Generate FL keys
        key_ids = generate_federated_learning_keys(key_manager)

        expected_keys = [
            "model_encryption",
            "proof_verification",
            "aggregation",
            "integrity_hmac"
        ]

        for key_name in expected_keys:
            self.assertIn(key_name, key_ids)

            # Verify key exists
            key_id = key_ids[key_name]
            status = key_manager.get_key_status(key_id)
            self.assertTrue(status["exists"])

class TestKeyGeneration(unittest.TestCase):
    """Test key generation functionality."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        keys_dir = Path(self.temp_dir) / "keys"

        storage_backends = {
            KeyStorageType.FILE: FileKeyStorage({
                "path": str(keys_dir),
                "encryption": False
            })
        }

        self.key_manager = KeyManager(
            security_config=KeySecurityConfig(integrity_check_enabled=False),
            storage_backends=storage_backends
        )

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def test_symmetric_key_generation(self):
        """Test symmetric key generation."""
        key_data = self.key_manager._generate_key_of_type(KeyType.SYMMETRIC, 32)
        self.assertEqual(len(key_data), 32)
        self.assertIsInstance(key_data, bytes)

    def test_hmac_key_generation(self):
        """Test HMAC key generation."""
        key_data = self.key_manager._generate_key_of_type(KeyType.HMAC, 32)
        self.assertEqual(len(key_data), 32)
        self.assertIsInstance(key_data, bytes)

    def test_verification_key_generation(self):
        """Test verification key generation."""
        key_data = self.key_manager._generate_key_of_type(KeyType.VERIFICATION_KEY, 32)
        self.assertIsInstance(key_data, bytes)
        self.assertGreater(len(key_data), 0)

class TestErrorHandling(unittest.TestCase):
    """Test error handling in key management."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        keys_dir = Path(self.temp_dir) / "keys"

        storage_backends = {
            KeyStorageType.FILE: FileKeyStorage({
                "path": str(keys_dir),
                "encryption": False
            })
        }

        self.key_manager = KeyManager(
            security_config=KeySecurityConfig(integrity_check_enabled=False),
            storage_backends=storage_backends
        )

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def test_invalid_storage_backend(self):
        """Test handling of invalid storage backend."""
        with self.assertRaises(KeyManagementError):
            self.key_manager.store_key(
                "test_key", b"data", KeyType.SYMMETRIC,
                storage_type="invalid_backend"
            )

    def test_corrupted_metadata_file(self):
        """Test handling of corrupted metadata files."""
        key_id = "corrupted_test_key"

        # Create corrupted metadata file
        meta_path = Path(self.temp_dir) / "keys" / f"{key_id}.meta"
        meta_path.parent.mkdir(parents=True, exist_ok=True)

        with open(meta_path, 'w') as f:
            f.write("invalid json content {")

        # Attempt to list keys (should handle corruption gracefully)
        keys = self.key_manager.list_keys()
        # Should not crash, but may not include corrupted key

class TestKeyRotationScenarios(unittest.TestCase):
    """Test various key rotation scenarios."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        keys_dir = Path(self.temp_dir) / "keys"

        storage_backends = {
            KeyStorageType.FILE: FileKeyStorage({
                "path": str(keys_dir),
                "encryption": False
            })
        }

        self.key_manager = KeyManager(
            security_config=KeySecurityConfig(integrity_check_enabled=False),
            rotation_config=KeyRotationConfig(auto_rotation=False),
            storage_backends=storage_backends
        )

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def test_usage_based_rotation(self):
        """Test usage-based key rotation."""
        key_id = "usage_rotation_key"
        key_data = b"usage_test_data"

        # Store key with usage-based rotation
        self.key_manager.store_key(
            key_id, key_data, KeyType.SYMMETRIC,
            rotation_policy=KeyRotationPolicy.USAGE_BASED
        )

        # Set low max usage count
        _, metadata = self.key_manager.retrieve_key(key_id)
        metadata.max_usage_count = 2

        # Use key multiple times
        for _ in range(3):
            self.key_manager.retrieve_key(key_id)

        # Check if rotation is needed
        status = self.key_manager.get_key_status(key_id)
        self.assertTrue(status["needs_rotation"])

if __name__ == '__main__':
    unittest.main()

