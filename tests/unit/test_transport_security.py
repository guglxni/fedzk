#!/usr/bin/env python3
"""
Tests for Transport Security
============================

Comprehensive test suite for FEDzk transport security implementation.
Tests TLS 1.3, certificate validation, pinning, and secure connections.
"""

import unittest
import ssl
import socket
import tempfile
import json
from unittest.mock import patch, MagicMock, mock_open
from pathlib import Path

from fedzk.security.transport_security import (
    TLSSecurityManager,
    TLSConfig,
    TLSVersion,
    CertificateValidationLevel,
    KeyExchangeProtocol,
    TLSCertificate,
    TLSConnectionInfo,
    CertificatePin,
    create_secure_tls_context,
    validate_certificate_chain,
    generate_secure_random_bytes
)

class TestTLSSecurityManager(unittest.TestCase):
    """Test TLSSecurityManager class."""

    def setUp(self):
        """Setup test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.cert_dir = Path(self.temp_dir) / "security"
        self.cert_dir.mkdir(exist_ok=True)

        # Mock certificate pins file
        self.pins_file = self.cert_dir / "certificate_pins.json"
        self.pins_data = [
            {
                "hostname": "example.com",
                "fingerprint_sha256": "abc123",
                "created_at": 1234567890,
                "expires_at": None,
                "notes": "Test pin"
            }
        ]

        self.config = TLSConfig(
            version=TLSVersion.TLS_1_3,
            certificate_validation=CertificateValidationLevel.STRICT,
            key_exchange_protocols=[KeyExchangeProtocol.ECDHE],
            cipher_suites=["ECDHE-RSA-AES256-GCM-SHA384"],
            session_timeout=3600,
            handshake_timeout=30
        )

    def tearDown(self):
        """Cleanup test environment."""
        import shutil
        shutil.rmtree(self.temp_dir)

    def test_tls_config_creation(self):
        """Test TLS configuration creation."""
        config = TLSConfig()

        self.assertEqual(config.version, TLSVersion.TLS_1_3)
        self.assertEqual(config.certificate_validation, CertificateValidationLevel.STRICT)
        self.assertIn(KeyExchangeProtocol.ECDHE, config.key_exchange_protocols)
        self.assertGreater(len(config.cipher_suites), 0)

    def test_tls_security_manager_creation(self):
        """Test TLS security manager creation."""
        manager = TLSSecurityManager(self.config)

        self.assertIsInstance(manager, TLSSecurityManager)
        self.assertEqual(manager.config.version, TLSVersion.TLS_1_3)
        self.assertIsInstance(manager.ssl_context, ssl.SSLContext)

    @patch('fedzk.security.transport_security.Path')
    def test_certificate_pin_loading(self, mock_path):
        """Test certificate pin loading."""
        # Mock the path to return our test file
        mock_path.return_value.exists.return_value = True

        with patch('builtins.open', mock_open(read_data=json.dumps(self.pins_data))):
            manager = TLSSecurityManager()
            manager._load_certificate_pins()

            self.assertIn("example.com", manager.certificate_pins)
            pin = manager.certificate_pins["example.com"]
            self.assertEqual(pin.fingerprint_sha256, "abc123")
            self.assertEqual(pin.notes, "Test pin")

    def test_certificate_pin_saving(self):
        """Test certificate pin saving."""
        with patch('builtins.open', mock_open()) as mock_file:
            manager = TLSSecurityManager()
            manager.certificate_pins["test.com"] = CertificatePin(
                hostname="test.com",
                fingerprint_sha256="def456",
                created_at=1234567890,
                expires_at=None,
                notes="Test pin"
            )

            manager._save_certificate_pins()

            # Verify file was written
            mock_file.assert_called()

    @patch('ssl.SSLSocket')
    def test_certificate_pin_validation(self, mock_ssl_sock):
        """Test certificate pin validation."""
        # Mock certificate
        mock_cert = {
            'subject': 'CN=example.com',
            'issuer': 'CN=Test CA',
            'serialNumber': '12345',
            'notBefore': '20230101000000Z',
            'notAfter': '20240101000000Z'
        }
        mock_ssl_sock.getpeercert.return_value = mock_cert

        manager = TLSSecurityManager()

        # Add a pin
        manager.certificate_pins["example.com"] = CertificatePin(
            hostname="example.com",
            fingerprint_sha256="mock_fingerprint",
            created_at=1234567890,
            expires_at=None,
            notes=""
        )

        # Mock the fingerprint calculation to return our expected value
        with patch.object(manager, '_extract_certificate_info') as mock_extract:
            mock_extract.return_value = TLSCertificate(
                subject="CN=example.com",
                issuer="CN=Test CA",
                serial_number="12345",
                not_before="20230101000000Z",
                not_after="20240101000000Z",
                fingerprint_sha256="mock_fingerprint",
                public_key_algorithm="RSA",
                signature_algorithm="SHA256-RSA",
                version=3,
                is_ca=False,
                key_size=2048
            )

            result = manager.validate_certificate_pin("example.com", mock_ssl_sock)
            self.assertTrue(result)

    def test_security_metrics(self):
        """Test security metrics collection."""
        manager = TLSSecurityManager()

        # Simulate some activity
        manager.security_metrics["total_connections"] = 5
        manager.security_metrics["successful_handshakes"] = 4
        manager.security_metrics["failed_handshakes"] = 1

        metrics = manager.get_security_metrics()

        self.assertEqual(metrics["total_connections"], 5)
        self.assertEqual(metrics["successful_handshakes"], 4)
        self.assertEqual(metrics["failed_handshakes"], 1)
        self.assertEqual(metrics["handshake_success_rate"], 0.8)
        self.assertEqual(metrics["tls_version"], "TLSv1.3")

    def test_connection_cleanup(self):
        """Test expired connection cleanup."""
        manager = TLSSecurityManager()

        # Add a mock connection
        import time
        old_time = time.time() - 4000  # Older than session timeout

        connection_info = TLSConnectionInfo(
            version="TLSv1.3",
            cipher_suite="ECDHE-RSA-AES256-GCM-SHA384",
            peer_certificate=None,
            compression="none",
            session_reused=False,
            established_at=old_time,
            bytes_sent=0,
            bytes_received=0
        )

        manager.connection_pool["example.com:443"] = connection_info

        # Cleanup should remove expired connections
        manager.cleanup_expired_connections()

        self.assertEqual(len(manager.connection_pool), 0)

    def test_network_traffic_validation(self):
        """Test network traffic validation."""
        manager = TLSSecurityManager()

        connection_info = TLSConnectionInfo(
            version="TLSv1.3",
            cipher_suite="ECDHE-RSA-AES256-GCM-SHA384",
            peer_certificate=None,
            compression="none",
            session_reused=False,
            established_at=1234567890,
            bytes_sent=0,
            bytes_received=0
        )

        test_data = b"test encrypted data"

        result = manager.validate_network_traffic(test_data, connection_info)

        self.assertTrue(result)
        self.assertEqual(manager.security_metrics["bytes_encrypted"], len(test_data))

    @patch('ssl.create_default_context')
    def test_create_secure_tls_context(self, mock_create_context):
        """Test secure TLS context creation."""
        mock_context = MagicMock()
        mock_create_context.return_value = mock_context

        context = create_secure_tls_context()

        # Verify SSL context was created
        mock_create_context.assert_called_once()

    @patch('socket.create_connection')
    @patch('ssl.create_default_context')
    def test_certificate_chain_validation(self, mock_create_context, mock_create_connection):
        """Test certificate chain validation."""
        # Mock SSL context and socket
        mock_context = MagicMock()
        mock_sock = MagicMock()
        mock_ssl_sock = MagicMock()

        mock_create_context.return_value = mock_context
        mock_create_connection.return_value.__enter__.return_value = mock_sock
        mock_context.wrap_socket.return_value.__enter__.return_value = mock_ssl_sock

        # Mock certificate
        mock_ssl_sock.getpeercert.return_value = {
            'subject': 'CN=test.example.com',
            'issuer': 'CN=Test CA',
            'serialNumber': '12345'
        }

        result = validate_certificate_chain("test.example.com", 443, 5)

        self.assertTrue(result["valid"])
        self.assertEqual(result["hostname"], "test.example.com")
        self.assertIsNotNone(result["certificate"])

    def test_secure_random_generation(self):
        """Test secure random bytes generation."""
        # Test different lengths
        for length in [16, 32, 64, 128]:
            random_bytes = generate_secure_random_bytes(length)
            self.assertEqual(len(random_bytes), length)
            self.assertIsInstance(random_bytes, bytes)

            # Verify randomness (basic check)
            if length > 1:
                # Very unlikely to get all zeros with secure random
                self.assertNotEqual(random_bytes, b'\x00' * length)

    def test_certificate_extraction(self):
        """Test certificate information extraction."""
        manager = TLSSecurityManager()

        # Mock certificate
        mock_cert = {
            'subject': 'CN=test.example.com',
            'issuer': 'CN=Test CA',
            'serialNumber': '123456789',
            'notBefore': '20230101000000Z',
            'notAfter': '20240101000000Z'
        }

        # Mock DER bytes
        mock_der = b'mock_certificate_der_data'

        with patch('ssl.SSLCertificate') as mock_ssl_cert:
            mock_ssl_cert.public_bytes.return_value = mock_der
            mock_ssl_cert.get.return_value = mock_cert.__getitem__

            # Mock public key
            mock_public_key = MagicMock()
            mock_public_key.key_size = 2048
            mock_ssl_cert.public_key.return_value = mock_public_key

            cert_info = manager._extract_certificate_info(mock_ssl_cert)

            self.assertIsInstance(cert_info, TLSCertificate)
            self.assertEqual(cert_info.subject, "CN=test.example.com")
            self.assertEqual(cert_info.issuer, "CN=Test CA")
            self.assertEqual(cert_info.serial_number, "123456789")
            self.assertEqual(cert_info.key_size, 2048)

class TestTLSConnectionInfo(unittest.TestCase):
    """Test TLSConnectionInfo class."""

    def test_connection_info_creation(self):
        """Test TLS connection info creation."""
        import time
        now = time.time()

        conn_info = TLSConnectionInfo(
            version="TLSv1.3",
            cipher_suite="ECDHE-RSA-AES256-GCM-SHA384",
            peer_certificate=None,
            compression="none",
            session_reused=False,
            established_at=now,
            bytes_sent=1024,
            bytes_received=2048
        )

        self.assertEqual(conn_info.version, "TLSv1.3")
        self.assertEqual(conn_info.cipher_suite, "ECDHE-RSA-AES256-GCM-SHA384")
        self.assertEqual(conn_info.bytes_sent, 1024)
        self.assertEqual(conn_info.bytes_received, 2048)

class TestCertificatePin(unittest.TestCase):
    """Test CertificatePin class."""

    def test_certificate_pin_creation(self):
        """Test certificate pin creation."""
        import time

        pin = CertificatePin(
            hostname="secure.example.com",
            fingerprint_sha256="abcdef1234567890",
            created_at=time.time(),
            expires_at=None,
            notes="Production certificate pin"
        )

        self.assertEqual(pin.hostname, "secure.example.com")
        self.assertEqual(pin.fingerprint_sha256, "abcdef1234567890")
        self.assertEqual(pin.notes, "Production certificate pin")
        self.assertIsNone(pin.expires_at)

class TestTLSConfig(unittest.TestCase):
    """Test TLSConfig class."""

    def test_default_config(self):
        """Test default TLS configuration."""
        config = TLSConfig()

        self.assertEqual(config.version, TLSVersion.TLS_1_3)
        self.assertEqual(config.certificate_validation, CertificateValidationLevel.STRICT)
        self.assertIn(KeyExchangeProtocol.ECDHE, config.key_exchange_protocols)
        self.assertGreater(len(config.cipher_suites), 0)
        self.assertEqual(config.session_timeout, 3600)
        self.assertEqual(config.handshake_timeout, 30)

    def test_custom_config(self):
        """Test custom TLS configuration."""
        config = TLSConfig(
            version=TLSVersion.TLS_1_2,
            certificate_validation=CertificateValidationLevel.NONE,
            session_timeout=1800,
            handshake_timeout=15
        )

        self.assertEqual(config.version, TLSVersion.TLS_1_2)
        self.assertEqual(config.certificate_validation, CertificateValidationLevel.NONE)
        self.assertEqual(config.session_timeout, 1800)
        self.assertEqual(config.handshake_timeout, 15)

if __name__ == '__main__':
    unittest.main()

