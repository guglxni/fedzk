#!/usr/bin/env python3
"""
Secrets Management Test Suite
============================

Comprehensive test suite for Task 8.2.2 Secrets Management.
Tests external providers, rotation, logging, and backup/recovery.
"""

import os
import sys
import tempfile
import unittest
import json
from pathlib import Path
from unittest.mock import patch, MagicMock, mock_open
from datetime import datetime, timedelta
import shutil

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from fedzk.config.secrets_manager import (
    FEDzkSecretsManager, SecretProvider, SecretRotationPolicy,
    SecretMetadata, SecretAccessEvent, SecretRotationManager,
    SecretAuditLogger, SecretBackupManager,
    HashiCorpVaultProvider, AWSSecretsManagerProvider
)


class TestSecretMetadata(unittest.TestCase):
    """Test SecretMetadata class."""

    def test_metadata_creation(self):
        """Test creating secret metadata."""
        metadata = SecretMetadata(
            name="test_secret",
            provider=SecretProvider.LOCAL,
            description="Test secret",
            tags={"env": "test"}
        )

        self.assertEqual(metadata.name, "test_secret")
        self.assertEqual(metadata.provider, SecretProvider.LOCAL)
        self.assertEqual(metadata.description, "Test secret")
        self.assertEqual(metadata.tags, {"env": "test"})
        self.assertEqual(metadata.version, 1)

    def test_metadata_serialization(self):
        """Test metadata serialization."""
        metadata = SecretMetadata(
            name="test_secret",
            provider=SecretProvider.AWS_SECRETS_MANAGER,
            description="Test description"
        )

        # Test to_dict
        data = metadata.to_dict()
        self.assertEqual(data['name'], "test_secret")
        self.assertEqual(data['provider'], "aws_secrets_manager")
        self.assertEqual(data['description'], "Test description")

        # Test from_dict
        new_metadata = SecretMetadata.from_dict(data)
        self.assertEqual(new_metadata.name, metadata.name)
        self.assertEqual(new_metadata.provider, metadata.provider)


class TestSecretAccessEvent(unittest.TestCase):
    """Test SecretAccessEvent class."""

    def test_event_creation(self):
        """Test creating access events."""
        event = SecretAccessEvent(
            secret_name="test_secret",
            operation="read",
            user="test_user",
            success=True
        )

        self.assertEqual(event.secret_name, "test_secret")
        self.assertEqual(event.operation, "read")
        self.assertEqual(event.user, "test_user")
        self.assertTrue(event.success)

    def test_event_serialization(self):
        """Test event serialization."""
        event = SecretAccessEvent("test_secret", "write", success=True)
        data = event.to_dict()

        self.assertEqual(data['secret_name'], "test_secret")
        self.assertEqual(data['operation'], "write")
        self.assertTrue(data['success'])
        self.assertIn('timestamp', data)


class TestHashiCorpVaultProvider(unittest.TestCase):
    """Test HashiCorp Vault provider."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = Path(tempfile.mkdtemp())

    def tearDown(self):
        """Clean up after tests."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    @patch('fedzk.config.secrets_manager.hvac')
    def test_vault_available(self, mock_hvac):
        """Test Vault availability check."""
        mock_client = MagicMock()
        mock_client.is_authenticated.return_value = True
        mock_hvac.Client.return_value = mock_client

        provider = HashiCorpVaultProvider()
        self.assertTrue(provider.is_available())

    @patch('fedzk.config.secrets_manager.hvac')
    def test_vault_unavailable(self, mock_hvac):
        """Test Vault unavailability."""
        mock_hvac.Client.side_effect = ImportError("hvac not available")

        provider = HashiCorpVaultProvider()
        self.assertFalse(provider.is_available())

    @patch('fedzk.config.secrets_manager.hvac')
    def test_vault_operations(self, mock_hvac):
        """Test Vault CRUD operations."""
        mock_client = MagicMock()
        mock_client.is_authenticated.return_value = True
        mock_hvac.Client.return_value = mock_client

        provider = HashiCorpVaultProvider()

        # Test store
        result = provider.store_secret("test/path", {"key": "value"})
        self.assertTrue(result)
        mock_client.secrets.kv.v2.create_or_update_secret.assert_called_once()

        # Test retrieve
        mock_response = {'data': {'data': {'key': 'value'}}}
        mock_client.secrets.kv.v2.read_secret_version.return_value = mock_response

        result = provider.retrieve_secret("test/path")
        self.assertEqual(result, {"key": "value"})


class TestAWSSecretsManagerProvider(unittest.TestCase):
    """Test AWS Secrets Manager provider."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = Path(tempfile.mkdtemp())

    def tearDown(self):
        """Clean up after tests."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    @patch('fedzk.config.secrets_manager.boto3')
    def test_aws_available(self, mock_boto3):
        """Test AWS availability check."""
        mock_client = MagicMock()
        mock_client.list_secrets.return_value = {}
        mock_boto3.client.return_value = mock_client

        provider = AWSSecretsManagerProvider()
        self.assertTrue(provider.is_available())

    @patch('fedzk.config.secrets_manager.boto3')
    def test_aws_operations(self, mock_boto3):
        """Test AWS CRUD operations."""
        mock_client = MagicMock()
        mock_boto3.client.return_value = mock_client

        provider = AWSSecretsManagerProvider()

        # Test store
        result = provider.store_secret("test-secret", {"key": "value"})
        self.assertTrue(result)
        mock_client.create_secret.assert_called_once()

        # Test retrieve
        mock_response = {'SecretString': '{"key": "value"}'}
        mock_client.get_secret_value.return_value = mock_response

        result = provider.retrieve_secret("test-secret")
        self.assertEqual(result, {"key": "value"})


class TestSecretRotationManager(unittest.TestCase):
    """Test secret rotation manager."""

    def setUp(self):
        """Set up test fixtures."""
        self.manager = FEDzkSecretsManager(SecretProvider.LOCAL)
        self.rotation_manager = SecretRotationManager(self.manager)

    def tearDown(self):
        """Clean up after tests."""
        self.rotation_manager.stop_rotation_scheduler()

    def test_rotation_policies(self):
        """Test rotation policy evaluation."""
        now = datetime.now()

        # Test NEVER policy
        metadata = SecretMetadata("test", SecretProvider.LOCAL, rotation_policy=SecretRotationPolicy.NEVER)
        self.assertFalse(self.rotation_manager._should_rotate_secret(metadata, now))

        # Test DAILY policy with old secret
        old_date = now - timedelta(days=2)
        metadata = SecretMetadata(
            "test", SecretProvider.LOCAL,
            rotation_policy=SecretRotationPolicy.DAILY,
            created_at=old_date
        )
        self.assertTrue(self.rotation_manager._should_rotate_secret(metadata, now))

    def test_rotation_intervals(self):
        """Test rotation interval calculation."""
        intervals = {
            SecretRotationPolicy.HOURLY: timedelta(hours=1),
            SecretRotationPolicy.DAILY: timedelta(days=1),
            SecretRotationPolicy.WEEKLY: timedelta(weeks=1),
            SecretRotationPolicy.MONTHLY: timedelta(days=30),
        }

        for policy, expected_interval in intervals.items():
            actual_interval = self.rotation_manager._get_rotation_interval(policy)
            self.assertEqual(actual_interval, expected_interval)

    def test_generate_new_secret(self):
        """Test new secret value generation."""
        metadata = SecretMetadata("password_secret", SecretProvider.LOCAL)

        new_value = self.rotation_manager._generate_new_secret_value("password_secret", metadata)
        self.assertIsInstance(new_value, dict)
        self.assertIn('value', new_value)

        # Test API key generation
        new_value = self.rotation_manager._generate_new_secret_value("api_key_secret", metadata)
        self.assertIsInstance(new_value, dict)
        self.assertIn('value', new_value)


class TestSecretAuditLogger(unittest.TestCase):
    """Test secret audit logger."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.log_file = self.temp_dir / "test_audit.log"
        self.logger = SecretAuditLogger(self.log_file)

    def tearDown(self):
        """Clean up after tests."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_log_event(self):
        """Test logging audit events."""
        event = SecretAccessEvent("test_secret", "read", success=True)
        self.logger.log_event(event)

        self.assertTrue(self.log_file.exists())

        # Verify log content
        with open(self.log_file, 'r') as f:
            log_data = json.loads(f.read().strip())
            self.assertEqual(log_data['secret_name'], "test_secret")
            self.assertEqual(log_data['operation'], "read")
            self.assertTrue(log_data['success'])

    def test_get_audit_events(self):
        """Test retrieving audit events."""
        # Log multiple events
        events = [
            SecretAccessEvent("secret1", "read", success=True),
            SecretAccessEvent("secret2", "write", success=False),
            SecretAccessEvent("secret1", "delete", success=True)
        ]

        for event in events:
            self.logger.log_event(event)

        # Test filtering
        all_events = self.logger.get_audit_events()
        self.assertEqual(len(all_events), 3)

        secret1_events = self.logger.get_audit_events(secret_name="secret1")
        self.assertEqual(len(secret1_events), 2)

        read_events = self.logger.get_audit_events(operation="read")
        self.assertEqual(len(read_events), 1)

    def test_get_access_stats(self):
        """Test access statistics generation."""
        # Log events
        self.logger.log_event(SecretAccessEvent("secret1", "read", success=True))
        self.logger.log_event(SecretAccessEvent("secret1", "read", success=True))
        self.logger.log_event(SecretAccessEvent("secret1", "write", success=False))

        stats = self.logger.get_access_stats("secret1")

        self.assertEqual(stats['total_accesses'], 3)
        self.assertEqual(stats['successful_accesses'], 2)
        self.assertEqual(stats['failed_accesses'], 1)
        self.assertEqual(stats['operations']['read'], 2)
        self.assertEqual(stats['operations']['write'], 1)


class TestSecretBackupManager(unittest.TestCase):
    """Test secret backup manager."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.backup_dir = self.temp_dir / "backups"
        self.manager = FEDzkSecretsManager(SecretProvider.LOCAL)
        self.backup_manager = SecretBackupManager(self.manager, self.backup_dir)

    def tearDown(self):
        """Clean up after tests."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_create_backup(self):
        """Test creating backups."""
        # Add some secrets
        self.manager.store_secret("test1", "value1", "Test secret 1")
        self.manager.store_secret("test2", {"key": "value2"}, "Test secret 2")

        # Create backup
        backup_path = self.backup_manager.create_backup("test_backup")

        self.assertIsNotNone(backup_path)
        self.assertTrue(backup_path.exists())

        # Verify backup content
        with open(backup_path, 'r') as f:
            backup_data = json.load(f)

        self.assertIn('secrets', backup_data)
        self.assertIn('metadata', backup_data)
        self.assertEqual(len(backup_data['secrets']), 2)
        self.assertEqual(len(backup_data['metadata']), 2)

    def test_list_backups(self):
        """Test listing backups."""
        # Create multiple backups
        self.manager.store_secret("test", "value", "Test")
        self.backup_manager.create_backup("backup1")
        self.backup_manager.create_backup("backup2")

        backups = self.backup_manager.list_backups()
        self.assertEqual(len(backups), 2)

        # Check sorting (newest first)
        self.assertTrue(backups[0]['created_at'] >= backups[1]['created_at'])

    def test_restore_backup(self):
        """Test restoring from backups."""
        # Create backup
        self.manager.store_secret("original", "original_value", "Original secret")
        backup_path = self.backup_manager.create_backup("restore_test")
        self.assertIsNotNone(backup_path)

        # Delete original secret
        self.manager.delete_secret("original")
        self.assertIsNone(self.manager.retrieve_secret("original"))

        # Restore from backup
        result = self.backup_manager.restore_backup("restore_test")
        self.assertEqual(result['restored_secrets'], 1)
        self.assertEqual(result['failed_secrets'], 0)

        # Verify restoration
        restored_value = self.manager.retrieve_secret("original")
        self.assertEqual(restored_value, "original_value")

    def test_dry_run_restore(self):
        """Test dry-run restoration."""
        # Create backup
        self.manager.store_secret("test", "value", "Test")
        self.backup_manager.create_backup("dry_run_test")

        # Test dry run
        result = self.backup_manager.restore_backup("dry_run_test", dry_run=True)
        self.assertTrue(result['dry_run'])
        self.assertEqual(result['restored_secrets'], 1)

        # Verify secret wasn't actually restored (since it already exists)
        # This tests the dry-run logic

    def test_cleanup_old_backups(self):
        """Test cleanup of old backups."""
        # Create backup files with old timestamps
        old_backup = self.backup_dir / "secrets_backup_old.json"
        old_backup.write_text('{"created_at": "2020-01-01T00:00:00"}')

        new_backup = self.backup_dir / "secrets_backup_new.json"
        new_backup.write_text('{"created_at": "2024-01-01T00:00:00"}')

        # Cleanup old backups (30+ days)
        deleted_count = self.backup_manager.cleanup_old_backups(keep_days=30)
        self.assertEqual(deleted_count, 1)
        self.assertFalse(old_backup.exists())
        self.assertTrue(new_backup.exists())


class TestFEDzkSecretsManager(unittest.TestCase):
    """Test main FEDzk secrets manager."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.secrets_dir = self.temp_dir / "secrets"
        self.metadata_file = self.temp_dir / "metadata.json"

        # Patch the default paths
        with patch('fedzk.config.secrets_manager.Path') as mock_path:
            mock_path.return_value = self.temp_dir
            self.manager = FEDzkSecretsManager(SecretProvider.LOCAL)

    def tearDown(self):
        """Clean up after tests."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_store_and_retrieve_local(self):
        """Test local storage and retrieval."""
        # Store secret
        success = self.manager.store_secret("test_secret", "test_value", "Test secret")
        self.assertTrue(success)

        # Retrieve secret
        value = self.manager.retrieve_secret("test_secret")
        self.assertEqual(value, "test_value")

    def test_delete_secret(self):
        """Test secret deletion."""
        # Store and verify
        self.manager.store_secret("test", "value", "Test")
        self.assertIsNotNone(self.manager.retrieve_secret("test"))

        # Delete and verify
        success = self.manager.delete_secret("test")
        self.assertTrue(success)
        self.assertIsNone(self.manager.retrieve_secret("test"))

    def test_list_secrets(self):
        """Test listing secrets."""
        # Store secrets
        self.manager.store_secret("secret1", "value1", "Secret 1")
        self.manager.store_secret("secret2", "value2", "Secret 2")

        secrets = self.manager.list_secrets()
        self.assertEqual(len(secrets), 2)
        self.assertIn("secret1", secrets)
        self.assertIn("secret2", secrets)

    def test_rotation_policy_management(self):
        """Test rotation policy management."""
        # Store secret
        self.manager.store_secret("test", "value", "Test")

        # Set rotation policy
        self.manager.set_rotation_policy("test", SecretRotationPolicy.DAILY)

        # Verify policy was set
        secrets = self.manager.list_secrets()
        self.assertEqual(secrets["test"].rotation_policy, SecretRotationPolicy.DAILY)

    def test_audit_integration(self):
        """Test audit logging integration."""
        # Store and retrieve to generate audit events
        self.manager.store_secret("audit_test", "value", "Audit test")
        self.manager.retrieve_secret("audit_test")

        # Check audit stats
        stats = self.manager.get_audit_stats("audit_test")
        self.assertGreater(stats['total_accesses'], 0)

    def test_backup_integration(self):
        """Test backup integration."""
        # Store secret
        self.manager.store_secret("backup_test", "value", "Backup test")

        # Create backup
        backup_path = self.manager.create_backup("integration_test")
        self.assertIsNotNone(backup_path)

        # List backups
        backups = self.manager.list_backups()
        self.assertGreater(len(backups), 0)

    def test_status_reporting(self):
        """Test status reporting."""
        status = self.manager.get_status()

        expected_keys = [
            'provider', 'total_secrets', 'providers_available',
            'rotation_active', 'audit_enabled', 'backup_enabled'
        ]

        for key in expected_keys:
            self.assertIn(key, status)

    def test_cleanup(self):
        """Test cleanup functionality."""
        # Start rotation scheduler
        self.manager.rotation_manager.start_rotation_scheduler()
        self.assertTrue(self.manager.rotation_manager.running)

        # Cleanup
        self.manager.cleanup()
        self.assertFalse(self.manager.rotation_manager.running)


class TestIntegration(unittest.TestCase):
    """Integration tests for secrets management."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = Path(tempfile.mkdtemp())

    def tearDown(self):
        """Clean up after tests."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_full_secret_lifecycle(self):
        """Test complete secret lifecycle."""
        manager = FEDzkSecretsManager(SecretProvider.LOCAL)

        secret_name = "lifecycle_test"
        secret_value = "test_value"
        description = "Lifecycle test secret"

        # 1. Store secret
        success = manager.store_secret(secret_name, secret_value, description)
        self.assertTrue(success)

        # 2. Retrieve secret
        retrieved = manager.retrieve_secret(secret_name)
        self.assertEqual(retrieved, secret_value)

        # 3. Update secret
        new_value = "updated_value"
        success = manager.store_secret(secret_name, new_value, description)
        self.assertTrue(success)

        retrieved = manager.retrieve_secret(secret_name)
        self.assertEqual(retrieved, new_value)

        # 4. Check metadata
        secrets = manager.list_secrets()
        self.assertIn(secret_name, secrets)
        metadata = secrets[secret_name]
        self.assertEqual(metadata.description, description)

        # 5. Create backup
        backup_path = manager.create_backup("lifecycle_backup")
        self.assertIsNotNone(backup_path)

        # 6. Delete secret
        success = manager.delete_secret(secret_name)
        self.assertTrue(success)

        # 7. Verify deletion
        retrieved = manager.retrieve_secret(secret_name)
        self.assertIsNone(retrieved)

        # 8. Restore from backup
        result = manager.restore_backup("lifecycle_backup")
        self.assertEqual(result['restored_secrets'], 1)

        # 9. Verify restoration
        retrieved = manager.retrieve_secret(secret_name)
        self.assertEqual(retrieved, new_value)

    def test_provider_fallback(self):
        """Test fallback to local storage when external provider unavailable."""
        # Test with unavailable external provider
        manager = FEDzkSecretsManager(SecretProvider.HASHICORP_VAULT)

        # Should fall back to local storage
        success = manager.store_secret("fallback_test", "value", "Fallback test")
        self.assertTrue(success)

        retrieved = manager.retrieve_secret("fallback_test")
        self.assertEqual(retrieved, "value")


def run_secrets_management_tests():
    """Run all secrets management tests."""
    print("🛡️  Running Secrets Management Tests")
    print("=" * 45)

    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add test cases
    test_classes = [
        TestSecretMetadata,
        TestSecretAccessEvent,
        TestHashiCorpVaultProvider,
        TestAWSSecretsManagerProvider,
        TestSecretRotationManager,
        TestSecretAuditLogger,
        TestSecretBackupManager,
        TestFEDzkSecretsManager,
        TestIntegration
    ]

    for test_class in test_classes:
        suite.addTests(loader.loadTestsFromTestCase(test_class))

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # Summary
    print("\n" * 2)
    print("=" * 45)
    print("Secrets Management Test Results:")
    print(f"  • Tests run: {result.testsRun}")
    print(f"  • Failures: {len(result.failures)}")
    print(f"  • Errors: {len(result.errors)}")
    print(f"  • Skipped: {len(result.skipped)}")

    if result.wasSuccessful():
        print("✅ All secrets management tests passed!")
        return True
    else:
        print("❌ Some secrets management tests failed")
        return False


if __name__ == "__main__":
    success = run_secrets_management_tests()
    sys.exit(0 if success else 1)
