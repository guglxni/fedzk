#!/usr/bin/env python3
"""
Tests for API Security
======================

Comprehensive test suite for FEDzk API security implementation.
Tests OAuth 2.0, JWT tokens, API keys, and secure request handling.
"""

import unittest
import json
import time
import tempfile
from unittest.mock import patch, MagicMock, mock_open
from pathlib import Path
from datetime import datetime, timedelta

from fedzk.security.api_security import (
    APISecurityManager,
    APISecurityConfig,
    JWTAlgorithm,
    OAuthGrantType,
    APIKeyStatus,
    OAuthClient,
    JWTToken,
    APIKey,
    create_secure_api_manager,
    generate_api_key,
    validate_bearer_token
)

class TestAPISecurityManager(unittest.TestCase):
    """Test APISecurityManager class."""

    def setUp(self):
        """Setup test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.security_dir = Path(self.temp_dir) / "security"
        self.security_dir.mkdir(exist_ok=True)

        # Mock API keys file
        self.api_keys_file = self.security_dir / "api_keys.json"
        self.api_keys_data = [
            {
                "key_id": "test_key_123",
                "key_hash": "mock_hash_123",
                "name": "Test Key",
                "description": "Test API key",
                "status": "active",
                "created_at": "2024-01-01T00:00:00",
                "expires_at": "2025-01-01T00:00:00",
                "last_used_at": None,
                "usage_count": 5,
                "rate_limit": 1000,
                "scopes": ["read", "write"],
                "metadata": {"test": True}
            }
        ]

        self.config = APISecurityConfig(
            jwt_algorithm=JWTAlgorithm.HS256,
            jwt_secret_key="test_secret_key_for_testing_only",
            jwt_expiration_hours=24,
            jwt_issuer="fedzk_test",
            jwt_audience=["fedzk_test"],
            api_key_enabled=True,
            api_key_length=32
        )

    def tearDown(self):
        """Cleanup test environment."""
        import shutil
        shutil.rmtree(self.temp_dir)

    def test_api_security_config(self):
        """Test API security configuration."""
        config = APISecurityConfig()

        self.assertEqual(config.jwt_algorithm, JWTAlgorithm.HS256)
        self.assertIsNotNone(config.jwt_secret_key)
        self.assertEqual(config.jwt_expiration_hours, 24)
        self.assertEqual(config.jwt_issuer, "fedzk")
        self.assertEqual(config.api_key_length, 32)

    def test_api_security_manager_creation(self):
        """Test API security manager creation."""
        manager = APISecurityManager(self.config)

        self.assertIsInstance(manager, APISecurityManager)
        self.assertEqual(manager.config.jwt_issuer, "fedzk_test")
        self.assertEqual(len(manager.active_tokens), 0)
        self.assertEqual(len(manager.api_keys), 0)

    @patch('fedzk.security.api_security.Path')
    def test_api_key_loading(self, mock_path):
        """Test API key loading."""
        mock_path.return_value.exists.return_value = True

        with patch('builtins.open', mock_open(read_data=json.dumps(self.api_keys_data))):
            manager = APISecurityManager()
            manager._load_api_keys()

            self.assertIn("test_key_123", manager.api_keys)
            api_key = manager.api_keys["test_key_123"]
            self.assertEqual(api_key.name, "Test Key")
            self.assertEqual(api_key.usage_count, 5)
            self.assertEqual(api_key.status, APIKeyStatus.ACTIVE)

    def test_jwt_token_creation(self):
        """Test JWT token creation."""
        manager = APISecurityManager(self.config)

        access_token, refresh_token = manager.create_jwt_token(
            subject="test_user",
            scopes=["read", "write"],
            additional_claims={"department": "engineering"}
        )

        self.assertIsInstance(access_token, str)
        self.assertIsInstance(refresh_token, str)
        self.assertGreater(len(access_token), 0)
        self.assertGreater(len(refresh_token), 0)

        # Verify token is stored
        self.assertIn(access_token, manager.active_tokens)
        self.assertIn(refresh_token, manager.refresh_tokens)

    def test_jwt_token_validation(self):
        """Test JWT token validation."""
        manager = APISecurityManager(self.config)

        # Create token
        access_token, _ = manager.create_jwt_token(
            subject="test_user",
            scopes=["read"]
        )

        # Validate token
        token_info = manager.validate_jwt_token(access_token)

        self.assertIsNotNone(token_info)
        self.assertIsInstance(token_info, JWTToken)
        self.assertEqual(token_info.subject, "test_user")
        self.assertEqual(token_info.scopes, ["read"])
        self.assertEqual(token_info.issuer, "fedzk_test")

    def test_jwt_token_refresh(self):
        """Test JWT token refresh."""
        manager = APISecurityManager(self.config)

        # Create initial token
        access_token, refresh_token = manager.create_jwt_token(
            subject="test_user",
            scopes=["read", "write"]
        )

        # Refresh token
        new_access, new_refresh = manager.refresh_jwt_token(refresh_token)

        self.assertIsNotNone(new_access)
        self.assertIsNotNone(new_refresh)
        self.assertNotEqual(new_access, access_token)
        self.assertNotEqual(new_refresh, refresh_token)

        # Old tokens should be cleaned up
        self.assertNotIn(access_token, manager.active_tokens)
        self.assertNotIn(refresh_token, manager.refresh_tokens)

    def test_api_key_creation(self):
        """Test API key creation."""
        manager = APISecurityManager(self.config)

        key_id, key_secret = manager.create_api_key(
            name="Test API Key",
            description="For testing purposes",
            scopes=["read", "write"],
            expires_in_days=365,
            rate_limit=1000
        )

        self.assertIsInstance(key_id, str)
        self.assertIsInstance(key_secret, str)
        self.assertGreater(len(key_id), 0)
        self.assertGreater(len(key_secret), 0)

        # Verify key is stored
        self.assertIn(key_id, manager.api_keys)
        api_key = manager.api_keys[key_id]
        self.assertEqual(api_key.name, "Test API Key")
        self.assertEqual(api_key.scopes, ["read", "write"])
        self.assertEqual(api_key.rate_limit, 1000)

    def test_api_key_validation(self):
        """Test API key validation."""
        manager = APISecurityManager(self.config)

        # Create API key
        key_id, key_secret = manager.create_api_key(
            name="Validation Test Key",
            scopes=["read"]
        )

        # Validate key
        validated_key = manager.validate_api_key(key_secret)

        self.assertIsNotNone(validated_key)
        self.assertIsInstance(validated_key, APIKey)
        self.assertEqual(validated_key.key_id, key_id)
        self.assertEqual(validated_key.name, "Validation Test Key")
        self.assertEqual(validated_key.usage_count, 1)

    def test_api_key_revocation(self):
        """Test API key revocation."""
        manager = APISecurityManager(self.config)

        # Create and validate key
        key_id, key_secret = manager.create_api_key("Revocation Test")
        validated_key = manager.validate_api_key(key_secret)
        self.assertIsNotNone(validated_key)

        # Revoke key
        revoked = manager.revoke_api_key(key_id)
        self.assertTrue(revoked)

        # Verify key is revoked
        api_key = manager.api_keys[key_id]
        self.assertEqual(api_key.status, APIKeyStatus.REVOKED)

        # Verify validation fails
        validated_key = manager.validate_api_key(key_secret)
        self.assertIsNone(validated_key)

    def test_rate_limiting(self):
        """Test API key rate limiting."""
        manager = APISecurityManager(self.config)

        # Create API key with low rate limit
        key_id, key_secret = manager.create_api_key(
            name="Rate Limit Test",
            rate_limit=2  # Only 2 requests per hour
        )

        # Use up the rate limit
        for i in range(2):
            validated_key = manager.validate_api_key(key_secret)
            self.assertIsNotNone(validated_key)

        # Next request should be rate limited
        validated_key = manager.validate_api_key(key_secret)
        self.assertIsNone(validated_key)

    def test_request_encryption(self):
        """Test request data encryption/decryption."""
        config = APISecurityConfig(request_encryption_enabled=True)
        manager = APISecurityManager(config)

        test_data = {"sensitive": "data", "user_id": 12345}
        key = "encryption_test_key"

        # Encrypt data
        encrypted = manager.encrypt_request_data(test_data, key)
        self.assertIsInstance(encrypted, str)
        self.assertNotEqual(encrypted, json.dumps(test_data))

        # Decrypt data
        decrypted = manager.decrypt_request_data(encrypted, key)
        self.assertEqual(decrypted, test_data)

    def test_audit_logging(self):
        """Test audit logging functionality."""
        manager = APISecurityManager(self.config)

        # Perform some operations that should be logged
        manager.create_api_key("Audit Test Key")
        manager.create_jwt_token(subject="audit_test_user")

        # Check audit log
        self.assertGreater(len(manager.audit_log), 0)

        # Verify log entries
        events = [entry["event"] for entry in manager.audit_log]
        self.assertIn("api_key_created", events)
        self.assertIn("token_created", events)

    def test_security_metrics(self):
        """Test security metrics collection."""
        manager = APISecurityManager(self.config)

        # Create some test data
        manager.create_api_key("Metrics Test 1")
        manager.create_api_key("Metrics Test 2", scopes=["read"])
        manager.create_jwt_token(subject="metrics_test_user")

        metrics = manager.get_security_metrics()

        self.assertEqual(metrics["total_api_keys"], 2)
        self.assertEqual(metrics["active_api_keys"], 2)
        self.assertEqual(metrics["active_tokens"], 1)
        self.assertGreater(metrics["audit_log_entries"], 0)

    def test_expired_token_cleanup(self):
        """Test cleanup of expired tokens."""
        manager = APISecurityManager(self.config)

        # Create a token
        access_token, _ = manager.create_jwt_token(subject="cleanup_test")

        # Manually expire the token
        token_info = manager.active_tokens[access_token]
        token_info.expires_at = datetime.utcnow() - timedelta(hours=1)

        # Run cleanup
        manager.cleanup_expired_tokens()

        # Verify token is removed
        self.assertNotIn(access_token, manager.active_tokens)

class TestAPISecurityConfig(unittest.TestCase):
    """Test APISecurityConfig class."""

    def test_default_config(self):
        """Test default API security configuration."""
        config = APISecurityConfig()

        self.assertEqual(config.jwt_algorithm, JWTAlgorithm.HS256)
        self.assertIsNotNone(config.jwt_secret_key)
        self.assertEqual(config.jwt_expiration_hours, 24)
        self.assertEqual(config.jwt_issuer, "fedzk")
        self.assertEqual(config.api_key_length, 32)

    def test_custom_config(self):
        """Test custom API security configuration."""
        config = APISecurityConfig(
            jwt_algorithm=JWTAlgorithm.RS256,
            jwt_expiration_hours=12,
            jwt_issuer="custom_issuer",
            api_key_enabled=False,
            request_encryption_enabled=True
        )

        self.assertEqual(config.jwt_algorithm, JWTAlgorithm.RS256)
        self.assertEqual(config.jwt_expiration_hours, 12)
        self.assertEqual(config.jwt_issuer, "custom_issuer")
        self.assertFalse(config.api_key_enabled)
        self.assertTrue(config.request_encryption_enabled)

class TestOAuthClient(unittest.TestCase):
    """Test OAuthClient class."""

    def test_oauth_client_creation(self):
        """Test OAuth client creation."""
        client = OAuthClient(
            client_id="test_client_123",
            client_secret="test_secret_456",
            redirect_uris=["https://example.com/callback"],
            grant_types=[OAuthGrantType.AUTHORIZATION_CODE],
            response_types=["code"],
            scopes=["read", "write"],
            token_endpoint_auth_method="client_secret_basic",
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow()
        )

        self.assertEqual(client.client_id, "test_client_123")
        self.assertEqual(client.client_secret, "test_secret_456")
        self.assertEqual(client.redirect_uris, ["https://example.com/callback"])
        self.assertEqual(client.grant_types, [OAuthGrantType.AUTHORIZATION_CODE])
        self.assertEqual(client.scopes, ["read", "write"])

class TestJWTToken(unittest.TestCase):
    """Test JWTToken class."""

    def test_jwt_token_creation(self):
        """Test JWT token creation."""
        now = datetime.utcnow()
        expires = now + timedelta(hours=24)

        token = JWTToken(
            token="eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9...",
            token_type="Bearer",
            expires_at=expires,
            issued_at=now,
            issuer="fedzk_test",
            subject="test_user",
            audience=["fedzk_test"],
            scopes=["read", "write"],
            claims={"custom": "data"}
        )

        self.assertEqual(token.token_type, "Bearer")
        self.assertEqual(token.issuer, "fedzk_test")
        self.assertEqual(token.subject, "test_user")
        self.assertEqual(token.scopes, ["read", "write"])
        self.assertEqual(token.claims["custom"], "data")

class TestAPIKey(unittest.TestCase):
    """Test APIKey class."""

    def test_api_key_creation(self):
        """Test API key creation."""
        now = datetime.utcnow()
        expires = now + timedelta(days=365)

        api_key = APIKey(
            key_id="test_key_789",
            key_hash="hash_of_key_secret",
            name="Test API Key",
            description="For testing purposes",
            status=APIKeyStatus.ACTIVE,
            created_at=now,
            expires_at=expires,
            last_used_at=None,
            usage_count=0,
            rate_limit=1000,
            scopes=["read", "write"],
            metadata={"test": True}
        )

        self.assertEqual(api_key.key_id, "test_key_789")
        self.assertEqual(api_key.name, "Test API Key")
        self.assertEqual(api_key.status, APIKeyStatus.ACTIVE)
        self.assertEqual(api_key.rate_limit, 1000)
        self.assertEqual(api_key.scopes, ["read", "write"])
        self.assertTrue(api_key.metadata["test"])

class TestConvenienceFunctions(unittest.TestCase):
    """Test convenience functions."""

    def test_create_secure_api_manager(self):
        """Test create_secure_api_manager function."""
        manager = create_secure_api_manager()

        self.assertIsInstance(manager, APISecurityManager)
        self.assertTrue(manager.config.api_key_enabled)
        self.assertTrue(manager.config.audit_logging_enabled)

    def test_generate_api_key(self):
        """Test generate_api_key function."""
        key_id, key_secret = generate_api_key("Test Generated Key", ["read"])

        self.assertIsInstance(key_id, str)
        self.assertIsInstance(key_secret, str)
        self.assertGreater(len(key_id), 0)
        self.assertGreater(len(key_secret), 0)

    def test_validate_bearer_token(self):
        """Test validate_bearer_token function."""
        manager = APISecurityManager()

        # Create a token
        access_token, _ = manager.create_jwt_token(subject="bearer_test")

        # Validate using convenience function
        token_info = validate_bearer_token(access_token)

        self.assertIsNotNone(token_info)
        self.assertEqual(token_info.subject, "bearer_test")

class TestErrorHandling(unittest.TestCase):
    """Test error handling in API security."""

    def setUp(self):
        self.manager = APISecurityManager()

    def test_invalid_jwt_token(self):
        """Test handling of invalid JWT tokens."""
        # Test with invalid token
        result = self.manager.validate_jwt_token("invalid.jwt.token")
        self.assertIsNone(result)

    def test_expired_jwt_token(self):
        """Test handling of expired JWT tokens."""
        # Create token and manually expire it
        access_token, _ = self.manager.create_jwt_token(subject="expired_test")
        token_info = self.manager.active_tokens[access_token]
        token_info.expires_at = datetime.utcnow() - timedelta(hours=1)

        # Validation should fail
        result = self.manager.validate_jwt_token(access_token)
        self.assertIsNone(result)

    def test_invalid_api_key(self):
        """Test handling of invalid API keys."""
        result = self.manager.validate_api_key("invalid_api_key_secret")
        self.assertIsNone(result)

    def test_revoked_api_key(self):
        """Test handling of revoked API keys."""
        # Create and revoke key
        key_id, key_secret = self.manager.create_api_key("Revoke Test")
        self.manager.revoke_api_key(key_id)

        # Validation should fail
        result = self.manager.validate_api_key(key_secret)
        self.assertIsNone(result)

if __name__ == '__main__':
    unittest.main()

