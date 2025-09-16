#!/usr/bin/env python3
"""
Environment Configuration Test Suite
===================================

Comprehensive test suite for Task 8.2.1 Environment Configuration.
Tests 12-factor principles, validation, hot-reloading, and encryption.
"""

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock
import time
import signal

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from fedzk.config.environment import (
    EnvironmentConfig, EnvironmentConfigManager,
    get_config_manager, get_config
)
from fedzk.config.validator import (
    ConfigValidator, ValidationResult, ValidationSeverity
)
from fedzk.config.hot_reload import (
    ConfigHotReload, ConfigChangeNotifier
)
from fedzk.config.encryption import (
    ConfigEncryption, SecureConfigStorage, ConfigEncryptionError
)


class TestEnvironmentConfig(unittest.TestCase):
    """Test EnvironmentConfig dataclass."""

    def setUp(self):
        """Set up test fixtures."""
        self.config = EnvironmentConfig()

    def test_default_values(self):
        """Test default configuration values."""
        self.assertEqual(self.config.app_name, "FEDzk")
        self.assertEqual(self.config.environment.value, "development")
        self.assertEqual(self.config.port, 8000)
        self.assertTrue(self.config.coordinator_enabled)
        self.assertTrue(self.config.mpc_enabled)
        self.assertTrue(self.config.zk_enabled)

    def test_config_attributes(self):
        """Test all configuration attributes exist."""
        expected_attrs = [
            'app_name', 'app_version', 'environment', 'host', 'port', 'debug',
            'coordinator_enabled', 'coordinator_host', 'coordinator_port',
            'mpc_enabled', 'mpc_host', 'mpc_port', 'zk_enabled', 'zk_circuit_path',
            'postgresql_enabled', 'redis_enabled', 'tls_enabled', 'api_keys_enabled',
            'jwt_secret_key', 'log_level', 'metrics_enabled', 'health_check_enabled'
        ]

        for attr in expected_attrs:
            self.assertTrue(hasattr(self.config, attr), f"Missing attribute: {attr}")


class TestEnvironmentConfigManager(unittest.TestCase):
    """Test EnvironmentConfigManager functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.manager = EnvironmentConfigManager()

    def tearDown(self):
        """Clean up after tests."""
        # Reset any environment variables set during tests
        for key in list(os.environ.keys()):
            if key.startswith('FEDZK_'):
                del os.environ[key]

    def test_load_from_environment(self):
        """Test loading configuration from environment variables."""
        # Set test environment variables
        test_vars = {
            'FEDZK_APP_NAME': 'TestApp',
            'FEDZK_PORT': '9000',
            'FEDZK_DEBUG': 'true',
            'FEDZK_ENVIRONMENT': 'testing'
        }

        with patch.dict(os.environ, test_vars):
            manager = EnvironmentConfigManager()
            self.assertEqual(manager.config.app_name, 'TestApp')
            self.assertEqual(manager.config.port, 9000)
            self.assertTrue(manager.config.debug)
            self.assertEqual(manager.config.environment.value, 'testing')

    def test_encrypted_values(self):
        """Test handling of encrypted configuration values."""
        manager = EnvironmentConfigManager()

        # Test encrypted password
        encrypted_password = manager.encryption.encrypt_value('test_password')
        with patch.dict(os.environ, {'FEDZK_POSTGRESQL_PASSWORD_ENCRYPTED': encrypted_password}):
            manager.load_from_environment()
            # Should decrypt the password
            self.assertEqual(manager.config.postgresql_password, 'test_password')

    def test_database_url_generation(self):
        """Test database URL generation."""
        manager = EnvironmentConfigManager()

        # Test default SQLite URL
        url = manager.get_database_url()
        self.assertIn('sqlite', url)

        # Test PostgreSQL URL
        manager.config.postgresql_enabled = True
        manager.config.postgresql_host = 'localhost'
        manager.config.postgresql_port = 5432
        manager.config.postgresql_database = 'testdb'
        manager.config.postgresql_username = 'testuser'
        manager.config.postgresql_password = 'testpass'

        url = manager.get_database_url()
        expected = 'postgresql://testuser:testpass@localhost:5432/testdb'
        self.assertEqual(url, expected)

    def test_config_validation(self):
        """Test configuration validation."""
        manager = EnvironmentConfigManager()

        # Should not raise exception with valid config
        try:
            manager.validate_configuration()
        except ValueError:
            self.fail("Valid configuration should not raise exception")

    def test_config_save_load(self):
        """Test saving and loading configuration from file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_file = Path(temp_dir) / "test_config.yaml"

            # Save configuration
            self.manager.save_to_file(config_file)
            self.assertTrue(config_file.exists())

            # Load configuration
            self.manager.load_from_file(config_file)
            self.assertEqual(self.manager.config.app_name, "FEDzk")


class TestConfigValidator(unittest.TestCase):
    """Test configuration validation functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.validator = ConfigValidator()
        self.config = EnvironmentConfig()

    def test_valid_config_validation(self):
        """Test validation of valid configuration."""
        results = self.validator.validate_config(self.config)
        # Should have no errors for default valid config
        errors = [r for r in results if r.severity == ValidationSeverity.ERROR]
        self.assertEqual(len(errors), 0)

    def test_invalid_config_validation(self):
        """Test validation of invalid configuration."""
        # Create invalid config
        invalid_config = EnvironmentConfig()
        invalid_config.app_name = ""  # Required but empty
        invalid_config.port = 80     # Below minimum
        invalid_config.jwt_secret_key = "short"  # Too short

        results = self.validator.validate_config(invalid_config)
        errors = [r for r in results if r.severity == ValidationSeverity.ERROR]
        self.assertGreater(len(errors), 0)

        # Check specific errors
        error_fields = [r.field for r in errors]
        self.assertIn('app_name', error_fields)
        self.assertIn('port', error_fields)
        self.assertIn('jwt_secret_key', error_fields)

    def test_validation_summary(self):
        """Test validation summary generation."""
        results = [
            ValidationResult('field1', ValidationSeverity.ERROR, 'Error message'),
            ValidationResult('field2', ValidationSeverity.WARNING, 'Warning message'),
            ValidationResult('field3', ValidationSeverity.INFO, 'Info message')
        ]

        summary = self.validator.get_validation_summary(results)
        self.assertEqual(summary['total_issues'], 3)
        self.assertEqual(summary['errors'], 1)
        self.assertEqual(summary['warnings'], 1)
        self.assertEqual(summary['info'], 1)


class TestConfigEncryption(unittest.TestCase):
    """Test configuration encryption functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.encryption = ConfigEncryption()

    def test_encrypt_decrypt(self):
        """Test basic encryption and decryption."""
        test_value = "sensitive_password_123"

        # Encrypt
        encrypted = self.encryption.encrypt_value(test_value, 'test_field')
        self.assertNotEqual(encrypted, test_value)
        self.assertTrue(self.encryption.is_encrypted(encrypted))

        # Decrypt
        decrypted = self.encryption.decrypt_value(encrypted, 'test_field')
        self.assertEqual(decrypted, test_value)

    def test_encryption_with_different_keys(self):
        """Test that different keys produce different results."""
        value = "test_value"

        encryption1 = ConfigEncryption("key1" * 8)
        encryption2 = ConfigEncryption("key2" * 8)

        encrypted1 = encryption1.encrypt_value(value)
        encrypted2 = encryption2.encrypt_value(value)

        self.assertNotEqual(encrypted1, encrypted2)

        # Each should decrypt with its own key
        self.assertEqual(encryption1.decrypt_value(encrypted1), value)
        self.assertEqual(encryption2.decrypt_value(encrypted2), value)

    def test_invalid_encrypted_value(self):
        """Test handling of invalid encrypted values."""
        with self.assertRaises(ConfigEncryptionError):
            self.encryption.decrypt_value("invalid_encrypted_value")

    def test_empty_value_handling(self):
        """Test handling of empty values."""
        with self.assertRaises(ConfigEncryptionError):
            self.encryption.encrypt_value("")

        with self.assertRaises(ConfigEncryptionError):
            self.encryption.decrypt_value("")


class TestSecureConfigStorage(unittest.TestCase):
    """Test secure configuration storage."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.storage_path = self.temp_dir / "test_secrets.yaml"
        self.encryption = ConfigEncryption()
        self.storage = SecureConfigStorage(self.storage_path, self.encryption)

    def tearDown(self):
        """Clean up after tests."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_store_and_retrieve_secret(self):
        """Test storing and retrieving secrets."""
        key = "test_api_key"
        value = "secret_value_123"
        description = "Test API key"

        # Store secret
        self.storage.store_secret(key, value, description)
        self.assertTrue(self.storage_path.exists())

        # Retrieve secret
        retrieved = self.storage.retrieve_secret(key)
        self.assertEqual(retrieved, value)

    def test_list_secrets(self):
        """Test listing stored secrets."""
        # Store multiple secrets
        secrets = {
            'api_key': 'key123',
            'db_password': 'pass456',
            'token': 'token789'
        }

        for key, value in secrets.items():
            self.storage.store_secret(key, value, f"Test {key}")

        # List secrets
        secret_list = self.storage.list_secrets()
        self.assertEqual(len(secret_list), 3)

        for key in secrets.keys():
            self.assertIn(key, secret_list)
            self.assertIn('description', secret_list[key])
            self.assertIn('created_at', secret_list[key])

    def test_delete_secret(self):
        """Test deleting stored secrets."""
        key = "test_key"
        value = "test_value"

        # Store and verify
        self.storage.store_secret(key, value)
        self.assertIsNotNone(self.storage.retrieve_secret(key))

        # Delete and verify
        self.assertTrue(self.storage.delete_secret(key))
        self.assertIsNone(self.storage.retrieve_secret(key))

        # Delete non-existent
        self.assertFalse(self.storage.delete_secret("non_existent"))


class TestConfigHotReload(unittest.TestCase):
    """Test configuration hot-reloading functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.config_file = self.temp_dir / "test_config.yaml"
        self.hot_reload = ConfigHotReload(watch_interval=1)

    def tearDown(self):
        """Clean up after tests."""
        self.hot_reload.stop_watching()
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_hot_reload_initialization(self):
        """Test hot reload initialization."""
        self.assertFalse(self.hot_reload.running)
        self.assertEqual(self.hot_reload.watch_interval, 1)
        self.assertEqual(len(self.hot_reload.reload_callbacks), 0)

    def test_add_remove_callbacks(self):
        """Test adding and removing reload callbacks."""
        def test_callback():
            pass

        # Add callback
        self.hot_reload.add_reload_callback(test_callback)
        self.assertEqual(len(self.hot_reload.reload_callbacks), 1)

        # Add same callback again (should not duplicate)
        self.hot_reload.add_reload_callback(test_callback)
        self.assertEqual(len(self.hot_reload.reload_callbacks), 1)

        # Remove callback
        self.hot_reload.remove_reload_callback(test_callback)
        self.assertEqual(len(self.hot_reload.reload_callbacks), 0)

    def test_get_status(self):
        """Test getting hot reload status."""
        status = self.hot_reload.get_status()
        self.assertIn('running', status)
        self.assertIn('watch_interval', status)
        self.assertIn('active_threads', status)
        self.assertIn('watched_files', status)
        self.assertIn('reload_callbacks', status)


class TestIntegration(unittest.TestCase):
    """Integration tests for the complete configuration system."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.config_file = self.temp_dir / "config.yaml"

    def tearDown(self):
        """Clean up after tests."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_full_config_workflow(self):
        """Test complete configuration workflow."""
        # Create configuration
        config = EnvironmentConfig()
        config.app_name = "TestApp"
        config.port = 9000

        # Create manager
        manager = EnvironmentConfigManager()

        # Validate
        validator = ConfigValidator()
        results = validator.validate_config(config)
        errors = [r for r in results if r.severity == ValidationSeverity.ERROR]
        self.assertEqual(len(errors), 0)

        # Save to file
        manager.save_to_file(self.config_file)
        self.assertTrue(self.config_file.exists())

        # Load from file
        manager.load_from_file(self.config_file)
        self.assertEqual(manager.config.app_name, "FEDzk")  # Should be default

    def test_encryption_integration(self):
        """Test encryption integration with configuration."""
        from fedzk.config.encryption import ConfigEncryptionManager

        # Create encryption manager
        enc_manager = ConfigEncryptionManager()

        # Test config with sensitive data
        test_config = {
            'api_key': 'test_key_123',
            'database_password': 'secret_pass',
            'jwt_secret_key': 'jwt_secret_456'
        }

        # Encrypt sensitive fields
        encrypted_config = enc_manager.encrypt_sensitive_config(test_config)

        # Verify encryption
        self.assertIn('api_key_encrypted', encrypted_config)
        self.assertNotEqual(encrypted_config['api_key'], 'test_key_123')

        # Decrypt
        decrypted_config = enc_manager.decrypt_sensitive_config(encrypted_config)

        # Verify decryption
        self.assertEqual(decrypted_config['api_key'], 'test_key_123')
        self.assertEqual(decrypted_config['database_password'], 'secret_pass')


def run_environment_config_tests():
    """Run all environment configuration tests."""
    print("🧪 Running Environment Configuration Tests")
    print("=" * 50)

    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add test cases
    test_classes = [
        TestEnvironmentConfig,
        TestEnvironmentConfigManager,
        TestConfigValidator,
        TestConfigEncryption,
        TestSecureConfigStorage,
        TestConfigHotReload,
        TestIntegration
    ]

    for test_class in test_classes:
        suite.addTests(loader.loadTestsFromTestCase(test_class))

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # Summary
    print("\n" + "=" * 50)
    print("Test Results Summary:")
    print(f"  • Tests run: {result.testsRun}")
    print(f"  • Failures: {len(result.failures)}")
    print(f"  • Errors: {len(result.errors)}")
    print(f"  • Skipped: {len(result.skipped)}")

    if result.wasSuccessful():
        print("✅ All tests passed!")
        return True
    else:
        print("❌ Some tests failed")
        return False


if __name__ == "__main__":
    success = run_environment_config_tests()
    sys.exit(0 if success else 1)
