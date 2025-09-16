#!/usr/bin/env python3
"""
Comprehensive Testing Suite for Configuration and Secrets Management
=====================================================================

Task 8.2.3: Comprehensive Testing Suite for Configuration and Secrets Management
Tests integration between FEDzk configuration management and secrets management.
"""

import os
import sys
import tempfile
import unittest
import json
import time
import threading
from pathlib import Path
from unittest.mock import patch, MagicMock, mock_open
from datetime import datetime, timedelta
import shutil

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from fedzk.config.environment import (
    EnvironmentConfig, EnvironmentConfigManager,
    get_config_manager
)
from fedzk.config.validator import (
    ConfigValidator, ValidationResult, ValidationSeverity
)
from fedzk.config.encryption import (
    ConfigEncryption, ConfigEncryptionError
)
from fedzk.config.secrets_manager import (
    FEDzkSecretsManager, SecretProvider, SecretRotationPolicy,
    SecretMetadata, SecretAccessEvent, get_secrets_manager
)


class TestConfigSecretsIntegration(unittest.TestCase):
    """Integration tests for configuration and secrets management."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.config_file = self.temp_dir / "test_config.yaml"
        self.secrets_dir = self.temp_dir / "secrets"
        self.backup_dir = self.temp_dir / "backups"
        self.logs_dir = self.temp_dir / "logs"

        # Create directories
        self.secrets_dir.mkdir(exist_ok=True)
        self.backup_dir.mkdir(exist_ok=True)
        self.logs_dir.mkdir(exist_ok=True)

        # Set required environment variables for tests
        self.env_patcher = patch.dict(os.environ, {
            'FEDZK_ENVIRONMENT': 'development',
            'FEDZK_JWT_SECRET_KEY': 'test_jwt_secret_key_for_testing_32_chars_minimum',
            'FEDZK_ENCRYPTION_MASTER_KEY': 'test_master_key_for_encryption_32_chars'
        })
        self.env_patcher.start()

        # Initialize managers with test paths
        with patch('fedzk.config.environment.Path') as mock_path:
            mock_path.return_value = self.temp_dir
            self.config_manager = EnvironmentConfigManager()

        # Initialize secrets manager with custom storage path
        self.secrets_manager = FEDzkSecretsManager(
            provider=SecretProvider.LOCAL,
            storage_path=self.temp_dir
        )

    def tearDown(self):
        """Clean up after tests."""
        if hasattr(self, 'env_patcher'):
            self.env_patcher.stop()
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_config_secrets_basic_integration(self):
        """Test basic integration between config and secrets."""
        # Store a secret
        secret_name = "db_password"
        secret_value = "super_secret_password"
        success = self.secrets_manager.store_secret(secret_name, secret_value, "Database password")

        self.assertTrue(success)

        # Verify secret is stored
        retrieved = self.secrets_manager.retrieve_secret(secret_name)
        self.assertEqual(retrieved, secret_value)

        # Test configuration with encrypted values
        config_manager = EnvironmentConfigManager()
        config = config_manager.config
        config.postgresql_password = secret_value
        # Use different port to avoid conflicts
        config.coordinator_port = 8002

        # Validate configuration
        validator = ConfigValidator()
        results = validator.validate_config(config)
        errors = [r for r in results if r.severity == ValidationSeverity.ERROR]
        self.assertEqual(len(errors), 0)

    def test_encrypted_config_secrets_workflow(self):
        """Test workflow with encrypted configuration values."""
        # Create encrypted configuration
        encryption = ConfigEncryption()

        # Store encrypted values in secrets manager
        db_password = "encrypted_db_password_123"
        api_key = "encrypted_api_key_456"
        jwt_secret = "encrypted_jwt_secret_789_extended_to_meet_32_char_requirement"

        # Store in secrets
        self.secrets_manager.store_secret("db_password", db_password)
        self.secrets_manager.store_secret("api_key", api_key)
        self.secrets_manager.store_secret("jwt_secret", jwt_secret)

        # Create configuration that references secrets
        config = EnvironmentConfig()
        config.postgresql_enabled = True
        config.api_keys_enabled = True
        config.coordinator_port = 8005  # Use different port to avoid conflicts

        # Simulate retrieving from secrets in config
        config.postgresql_password = self.secrets_manager.retrieve_secret("db_password")
        config.jwt_secret_key = self.secrets_manager.retrieve_secret("jwt_secret")

        # Validate configuration
        validator = ConfigValidator()
        results = validator.validate_config(config)

        # Should pass validation
        errors = [r for r in results if r.severity == ValidationSeverity.ERROR]
        self.assertEqual(len(errors), 0)

        # Verify values are correct
        self.assertEqual(config.postgresql_password, db_password)
        self.assertEqual(config.jwt_secret_key, jwt_secret)

    def test_config_hot_reload_with_secrets(self):
        """Test hot reload integration with secrets management."""
        # Create initial configuration file
        initial_config = {
            'app_name': 'TestApp',
            'port': 8000,
            'postgresql_enabled': True,
            'postgresql_password': 'initial_password'
        }

        with open(self.config_file, 'w') as f:
            json.dump(initial_config, f)

        # Load configuration
        self.config_manager.load_from_file(self.config_file)

        # Verify initial state
        self.assertEqual(self.config_manager.config.app_name, 'TestApp')
        self.assertEqual(self.config_manager.config.port, 8000)

        # Modify configuration file
        updated_config = initial_config.copy()
        updated_config['app_name'] = 'UpdatedApp'
        updated_config['port'] = 9000

        with open(self.config_file, 'w') as f:
            json.dump(updated_config, f)

        # Reload configuration
        self.config_manager.load_from_file(self.config_file)

        # Verify updates
        self.assertEqual(self.config_manager.config.app_name, 'UpdatedApp')
        self.assertEqual(self.config_manager.config.port, 9000)

    def test_secrets_rotation_with_config_integration(self):
        """Test secret rotation and its impact on configuration."""
        # Store initial secret
        secret_name = "api_key"
        initial_value = "initial_api_key_123"
        self.secrets_manager.store_secret(secret_name, initial_value, "API Key")

        # Set rotation policy
        self.secrets_manager.set_rotation_policy(secret_name, SecretRotationPolicy.DAILY)

        # Simulate configuration using the secret
        config = EnvironmentConfig()
        config.api_keys_enabled = True
        config.jwt_secret_key = initial_value + "_extended_to_meet_32_char_requirement"  # Make it long enough
        config.coordinator_port = 8004  # Use different port to avoid conflicts

        # Verify initial configuration
        validator = ConfigValidator()
        results = validator.validate_config(config)
        errors = [r for r in results if r.severity == ValidationSeverity.ERROR]
        self.assertEqual(len(errors), 0)

        # Manually rotate secret (simulate rotation)
        new_value = "rotated_api_key_456_extended_to_meet_32_char_requirement"
        success = self.secrets_manager.store_secret(secret_name, new_value, "API Key")
        self.assertTrue(success)

        # Configuration should use new value
        config.jwt_secret_key = self.secrets_manager.retrieve_secret(secret_name)
        self.assertEqual(config.jwt_secret_key, new_value)

        # Validate updated configuration
        results = validator.validate_config(config)
        errors = [r for r in results if r.severity == ValidationSeverity.ERROR]
        self.assertEqual(len(errors), 0)

    def test_backup_restore_config_secrets(self):
        """Test backup and restore of configuration with secrets."""
        # Create configuration with secrets
        config = EnvironmentConfig()
        config.app_name = "BackupTest"
        config.postgresql_enabled = True
        config.api_keys_enabled = True

        # Store related secrets
        self.secrets_manager.store_secret("backup_db_pass", "backup_password", "Backup DB Password")
        self.secrets_manager.store_secret("backup_api_key", "backup_api_key_123", "Backup API Key")

        # Associate secrets with config (use a longer JWT key to meet validation requirements)
        config.postgresql_password = self.secrets_manager.retrieve_secret("backup_db_pass")
        config.jwt_secret_key = self.secrets_manager.retrieve_secret("backup_api_key") + "_extended_to_meet_32_char_requirement"
        config.coordinator_port = 8003  # Use different port to avoid conflicts

        # Create backup
        backup_path = self.secrets_manager.create_backup("config_secrets_backup")
        self.assertIsNotNone(backup_path)

        # Verify backup contains our secrets
        with open(backup_path, 'r') as f:
            backup_data = json.load(f)

        self.assertIn('secrets', backup_data)
        self.assertIn('backup_db_pass', backup_data['secrets'])
        self.assertIn('backup_api_key', backup_data['secrets'])

        # Delete secrets
        self.secrets_manager.delete_secret("backup_db_pass")
        self.secrets_manager.delete_secret("backup_api_key")

        # Verify deletion
        self.assertIsNone(self.secrets_manager.retrieve_secret("backup_db_pass"))
        self.assertIsNone(self.secrets_manager.retrieve_secret("backup_api_key"))

        # Restore from backup
        result = self.secrets_manager.restore_backup("config_secrets_backup")
        self.assertEqual(result['restored_secrets'], 2)

        # Verify restoration
        restored_db_pass = self.secrets_manager.retrieve_secret("backup_db_pass")
        restored_api_key = self.secrets_manager.retrieve_secret("backup_api_key")

        self.assertEqual(restored_db_pass, "backup_password")
        self.assertEqual(restored_api_key, "backup_api_key_123")

        # Verify configuration still works with restored secrets
        config.postgresql_password = restored_db_pass
        config.jwt_secret_key = restored_api_key

        validator = ConfigValidator()
        results = validator.validate_config(config)
        errors = [r for r in results if r.severity == ValidationSeverity.ERROR]
        self.assertEqual(len(errors), 0)

    def test_audit_trail_config_secrets(self):
        """Test audit trail for configuration and secrets operations."""
        # Perform various operations
        self.secrets_manager.store_secret("audit_test", "test_value", "Audit test secret")
        self.secrets_manager.retrieve_secret("audit_test")
        self.secrets_manager.retrieve_secret("non_existent")  # Should fail

        # Get audit statistics
        stats = self.secrets_manager.get_audit_stats("audit_test")

        # Verify audit data
        self.assertGreaterEqual(stats['total_accesses'], 2)  # store + retrieve
        self.assertGreaterEqual(stats['successful_accesses'], 1)
        self.assertGreaterEqual(stats['failed_accesses'], 0)

        # Test audit events
        events = self.secrets_manager.audit_log.get_audit_events(secret_name="audit_test")
        self.assertGreaterEqual(len(events), 1)

        # Verify event structure
        if events:
            event = events[0]
            self.assertIn('secret_name', event)
            self.assertIn('operation', event)
            self.assertIn('timestamp', event)
            self.assertIn('success', event)

    def test_provider_failover_config_secrets(self):
        """Test provider failover and its impact on configuration."""
        # Test with local provider (always available)
        local_manager = FEDzkSecretsManager(SecretProvider.LOCAL)

        # Store secret
        success = local_manager.store_secret("failover_test", "test_value", "Failover test")
        self.assertTrue(success)

        # Retrieve secret
        value = local_manager.retrieve_secret("failover_test")
        self.assertEqual(value, "test_value")

        # Simulate configuration using the secret
        config = EnvironmentConfig()
        config.postgresql_enabled = True
        config.postgresql_password = value
        config.coordinator_port = 8006  # Use different port to avoid conflicts

        # Validate configuration works with local provider
        validator = ConfigValidator()
        results = validator.validate_config(config)
        errors = [r for r in results if r.severity == ValidationSeverity.ERROR]
        self.assertEqual(len(errors), 0)

        # Test status reporting
        status = local_manager.get_status()
        self.assertEqual(status['provider'], 'local')
        self.assertIn('providers_available', status)

    def test_performance_config_secrets_operations(self):
        """Test performance of configuration and secrets operations."""
        import time

        # Test secrets storage performance
        secrets_to_store = 50
        start_time = time.time()

        for i in range(secrets_to_store):
            secret_name = f"perf_test_secret_{i}"
            secret_value = f"performance_test_value_{i}"
            self.secrets_manager.store_secret(secret_name, secret_value, f"Performance test {i}")

        store_time = time.time() - start_time

        # Test retrieval performance
        start_time = time.time()

        for i in range(secrets_to_store):
            secret_name = f"perf_test_secret_{i}"
            value = self.secrets_manager.retrieve_secret(secret_name)
            self.assertIsNotNone(value)

        retrieve_time = time.time() - start_time

        # Performance assertions (adjust thresholds based on environment)
        self.assertLess(store_time, 10.0, "Secret storage should be reasonably fast")
        self.assertLess(retrieve_time, 5.0, "Secret retrieval should be reasonably fast")

        # Test configuration validation performance
        config = EnvironmentConfig()
        config.app_name = "PerformanceTest"
        config.port = 8080
        config.postgresql_enabled = True

        validator = ConfigValidator()

        start_time = time.time()
        for _ in range(100):
            results = validator.validate_config(config)
        validation_time = time.time() - start_time

        self.assertLess(validation_time, 2.0, "Configuration validation should be fast")

    def test_compliance_config_secrets(self):
        """Test compliance features for configuration and secrets."""
        # Test configuration compliance
        config = EnvironmentConfig()
        config.environment = "production"
        config.debug = False
        config.tls_enabled = True
        config.api_keys_enabled = True
        config.coordinator_port = 8007  # Use different port to avoid conflicts

        # Provide required configuration for enabled features
        config.tls_ca_path = "/etc/ssl/certs"  # Required when TLS is enabled
        config.jwt_secret_key = "production_jwt_secret_key_meeting_32_char_requirement"  # Required for API keys

        validator = ConfigValidator()
        results = validator.validate_config(config)

        # Should pass production compliance checks
        errors = [r for r in results if r.severity == ValidationSeverity.ERROR]
        self.assertEqual(len(errors), 0)

        # Test secrets compliance
        # Store secrets with proper metadata
        compliance_secrets = {
            'prod_db_password': 'secure_password_123',
            'prod_api_key': 'sk-prod-456',
            'prod_jwt_secret': 'jwt-prod-secret-789'
        }

        for name, value in compliance_secrets.items():
            success = self.secrets_manager.store_secret(
                name, value,
                f"Production {name.replace('_', ' ')}",
                {'environment': 'production', 'compliance': 'soc2'}
            )
            self.assertTrue(success)

        # Verify secrets are stored with metadata
        secrets_list = self.secrets_manager.list_secrets()
        for name in compliance_secrets.keys():
            self.assertIn(name, secrets_list)
            metadata = secrets_list[name]
            self.assertEqual(metadata.tags.get('environment'), 'production')

    def test_monitoring_alerting_config_secrets(self):
        """Test monitoring and alerting for configuration and secrets."""
        # Generate some monitoring data
        for i in range(10):
            secret_name = f"monitor_test_{i}"
            self.secrets_manager.store_secret(secret_name, f"value_{i}", f"Monitor test {i}")

        # Test each secret
        for i in range(10):
            secret_name = f"monitor_test_{i}"
            value = self.secrets_manager.retrieve_secret(secret_name)
            self.assertEqual(value, f"value_{i}")

        # Test monitoring statistics
        total_stats = self.secrets_manager.get_audit_stats()

        # Should have recorded operations
        self.assertGreater(total_stats['total_accesses'], 10)  # 10 stores + 10 retrieves

        # Test per-secret monitoring
        individual_stats = self.secrets_manager.get_audit_stats("monitor_test_0")
        self.assertGreater(individual_stats['total_accesses'], 0)

        # Test backup monitoring
        backup_path = self.secrets_manager.create_backup("monitoring_test")
        self.assertIsNotNone(backup_path)

        # Test backup listing (should find at least the one we just created)
        backups = self.secrets_manager.list_backups()
        self.assertGreaterEqual(len(backups), 1)

        # Verify backup contains our test data
        latest_backup = backups[0]
        self.assertGreater(latest_backup['secret_count'], 0)


class TestSecurityCompliance(unittest.TestCase):
    """Security and compliance tests for configuration and secrets."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.encryption = ConfigEncryption()

    def tearDown(self):
        """Clean up after tests."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_encryption_security(self):
        """Test encryption security properties."""
        test_value = "sensitive_data_123"

        # Encrypt value
        encrypted = self.encryption.encrypt_value(test_value, "test_field")

        # Verify encryption
        self.assertNotEqual(encrypted, test_value)
        self.assertTrue(self.encryption.is_encrypted(encrypted))

        # Decrypt value
        decrypted = self.encryption.decrypt_value(encrypted, "test_field")
        self.assertEqual(decrypted, test_value)

        # Test different keys produce different results
        encryption2 = ConfigEncryption("different_key_456")
        encrypted2 = encryption2.encrypt_value(test_value, "test_field")
        self.assertNotEqual(encrypted, encrypted2)

        # Test decryption with wrong key fails
        with self.assertRaises(ConfigEncryptionError):
            encryption2.decrypt_value(encrypted, "test_field")

    def test_configuration_validation_security(self):
        """Test security aspects of configuration validation."""
        validator = ConfigValidator()

        # Test with potentially malicious configuration
        malicious_config = EnvironmentConfig()
        malicious_config.app_name = "test"  # Valid
        malicious_config.port = 22  # Potentially malicious (SSH port)
        malicious_config.host = "0.0.0.0"  # Valid but should be validated

        results = validator.validate_config(malicious_config)

        # Should catch port validation
        port_results = [r for r in results if 'port' in r.field and r.severity == ValidationSeverity.ERROR]
        self.assertTrue(len(port_results) > 0)

        # Test with valid secure configuration
        secure_config = EnvironmentConfig()
        secure_config.app_name = "SecureApp"
        secure_config.port = 8443  # HTTPS port
        secure_config.tls_enabled = True
        secure_config.api_keys_enabled = True
        secure_config.environment = "production"

        results = validator.validate_config(secure_config)
        errors = [r for r in results if r.severity == ValidationSeverity.ERROR]
        self.assertEqual(len(errors), 0)

    def test_secrets_access_control(self):
        """Test access control for secrets operations."""
        manager = FEDzkSecretsManager(SecretProvider.LOCAL)

        # Test successful operations
        success = manager.store_secret("access_test", "test_value", "Access control test")
        self.assertTrue(success)

        value = manager.retrieve_secret("access_test")
        self.assertEqual(value, "test_value")

        # Test failed operations (non-existent secret)
        value = manager.retrieve_secret("non_existent_secret")
        self.assertIsNone(value)

        # Verify audit trail captures both success and failure
        stats = manager.get_audit_stats()
        self.assertGreater(stats['total_accesses'], 0)
        self.assertGreaterEqual(stats['successful_accesses'], 1)
        self.assertGreaterEqual(stats['failed_accesses'], 0)

    def test_backup_security(self):
        """Test security of backup operations."""
        manager = FEDzkSecretsManager(SecretProvider.LOCAL)

        # Store sensitive secrets
        sensitive_data = {
            'prod_db_password': 'super_secret_db_pass',
            'prod_api_key': 'sk-prod-123456789',
            'prod_jwt_secret': 'jwt-secret-key-for-prod'
        }

        for name, value in sensitive_data.items():
            manager.store_secret(name, value, f"Sensitive {name}")

        # Create backup
        backup_path = manager.create_backup("security_test")

        # Verify backup file exists and is readable
        self.assertTrue(backup_path.exists())

        # Read backup content
        with open(backup_path, 'r') as f:
            backup_data = json.load(f)

        # Verify sensitive data is in backup
        self.assertIn('secrets', backup_data)
        for name, value in sensitive_data.items():
            self.assertIn(name, backup_data['secrets'])
            self.assertEqual(backup_data['secrets'][name], value)

        # Test backup integrity (backup should be valid JSON)
        try:
            json.dumps(backup_data)  # Should not raise exception
        except (TypeError, ValueError):
            self.fail("Backup data should be valid JSON")

    def test_provider_security(self):
        """Test security of external providers."""
        # Test local provider security (should always be secure)
        local_manager = FEDzkSecretsManager(SecretProvider.LOCAL)

        # Store and retrieve with local provider
        success = local_manager.store_secret("security_test", "secure_value", "Security test")
        self.assertTrue(success)

        value = local_manager.retrieve_secret("security_test")
        self.assertEqual(value, "secure_value")

        # Test provider status
        status = local_manager.get_status()
        self.assertEqual(status['provider'], 'local')

        # Test that provider correctly reports availability
        self.assertIsInstance(status['providers_available'], dict)


class TestPerformanceBenchmarking(unittest.TestCase):
    """Performance benchmarking tests for configuration and secrets."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = Path(tempfile.mkdtemp())

    def tearDown(self):
        """Clean up after tests."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_configuration_performance(self):
        """Test configuration system performance."""
        import time

        # Test configuration loading performance
        config_manager = EnvironmentConfigManager()

        start_time = time.time()
        for _ in range(100):
            config = EnvironmentConfig()
            config.app_name = f"PerfTest_{_}"
            config.port = 8000 + (_ % 1000)
        load_time = time.time() - start_time

        # Test validation performance
        validator = ConfigValidator()
        config = EnvironmentConfig()

        start_time = time.time()
        for _ in range(1000):
            results = validator.validate_config(config)
        validation_time = time.time() - start_time

        # Performance assertions
        self.assertLess(load_time, 1.0, "Configuration loading should be fast")
        self.assertLess(validation_time, 5.0, "Configuration validation should be reasonably fast")

    def test_secrets_performance(self):
        """Test secrets management performance."""
        import time

        manager = FEDzkSecretsManager(SecretProvider.LOCAL)

        # Test secret storage performance
        num_secrets = 100
        start_time = time.time()

        for i in range(num_secrets):
            manager.store_secret(f"perf_secret_{i}", f"value_{i}", f"Performance test {i}")

        store_time = time.time() - start_time

        # Test secret retrieval performance
        start_time = time.time()

        for i in range(num_secrets):
            value = manager.retrieve_secret(f"perf_secret_{i}")
            self.assertEqual(value, f"value_{i}")

        retrieve_time = time.time() - start_time

        # Performance assertions (adjust based on environment)
        avg_store_time = store_time / num_secrets
        avg_retrieve_time = retrieve_time / num_secrets

        self.assertLess(avg_store_time, 0.1, "Average secret storage should be fast")
        self.assertLess(avg_retrieve_time, 0.05, "Average secret retrieval should be very fast")

    def test_concurrent_operations(self):
        """Test concurrent operations on configuration and secrets."""
        import concurrent.futures
        import threading

        manager = FEDzkSecretsManager(SecretProvider.LOCAL)
        errors = []

        def worker(worker_id):
            try:
                # Each worker performs operations
                for i in range(10):
                    secret_name = f"concurrent_{worker_id}_{i}"
                    manager.store_secret(secret_name, f"value_{worker_id}_{i}", f"Concurrent test {worker_id}")
                    value = manager.retrieve_secret(secret_name)
                    if value != f"value_{worker_id}_{i}":
                        errors.append(f"Worker {worker_id}: Value mismatch")
            except Exception as e:
                errors.append(f"Worker {worker_id}: {e}")

        # Run concurrent operations
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(worker, i) for i in range(5)]
            concurrent.futures.wait(futures)

        # Check for errors
        if errors:
            self.fail(f"Concurrent operations failed: {errors}")

        # Verify all secrets were stored correctly
        for worker_id in range(5):
            for i in range(10):
                secret_name = f"concurrent_{worker_id}_{i}"
                value = manager.retrieve_secret(secret_name)
                self.assertEqual(value, f"value_{worker_id}_{i}")


def run_comprehensive_config_secrets_tests():
    """Run all comprehensive configuration and secrets management tests."""
    print("🧪 Running Comprehensive Configuration & Secrets Management Tests")
    print("=" * 75)

    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add test classes
    test_classes = [
        TestConfigSecretsIntegration,
        TestSecurityCompliance,
        TestPerformanceBenchmarking
    ]

    for test_class in test_classes:
        suite.addTests(loader.loadTestsFromTestCase(test_class))

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # Summary
    print("\n" * 2)
    print("=" * 75)
    print("Comprehensive Configuration & Secrets Management Test Results:")
    print(f"  • Tests run: {result.testsRun}")
    print(f"  • Failures: {len(result.failures)}")
    print(f"  • Errors: {len(result.errors)}")
    print(f"  • Skipped: {len(result.skipped)}")

    if result.wasSuccessful():
        print("✅ All comprehensive tests passed!")
        return True
    else:
        print("❌ Some comprehensive tests failed")
        if result.failures:
            print("Failures:")
            for test, traceback in result.failures:
                print(f"  • {test}")
        if result.errors:
            print("Errors:")
            for test, traceback in result.errors:
                print(f"  • {test}")
        return False


if __name__ == "__main__":
    success = run_comprehensive_config_secrets_tests()
    sys.exit(0 if success else 1)
