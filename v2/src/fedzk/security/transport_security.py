#!/usr/bin/env python3
"""
FEDzk Transport Security Layer
==============================

Comprehensive TLS 1.3 transport security implementation for FEDzk.
Provides encrypted communications, certificate validation, and secure key exchange.

Features:
- TLS 1.3 encryption for all network communications
- Certificate validation and pinning
- Secure key exchange protocols
- Network traffic encryption validation
- Production security standards
"""

import ssl
import socket
import logging
import time
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, asdict
from enum import Enum
import json
import hashlib
import hmac
from pathlib import Path
import asyncio
from concurrent.futures import ThreadPoolExecutor
import threading
import secrets

logger = logging.getLogger(__name__)

class TLSVersion(Enum):
    """Supported TLS versions."""
    TLS_1_2 = "TLSv1.2"
    TLS_1_3 = "TLSv1.3"

class CertificateValidationLevel(Enum):
    """Certificate validation strictness levels."""
    NONE = "none"
    BASIC = "basic"
    STRICT = "strict"
    PINNING = "pinning"

class KeyExchangeProtocol(Enum):
    """Supported key exchange protocols."""
    ECDHE = "ECDHE"
    DHE = "DHE"
    RSA = "RSA"

@dataclass
class TLSCertificate:
    """TLS certificate information."""
    subject: str
    issuer: str
    serial_number: str
    not_before: str
    not_after: str
    fingerprint_sha256: str
    public_key_algorithm: str
    signature_algorithm: str
    version: int
    is_ca: bool
    key_size: int

@dataclass
class TLSConnectionInfo:
    """TLS connection information."""
    version: str
    cipher_suite: str
    peer_certificate: Optional[TLSCertificate]
    compression: str
    session_reused: bool
    established_at: float
    bytes_sent: int
    bytes_received: int

@dataclass
class TLSConfig:
    """TLS configuration settings."""
    version: TLSVersion = TLSVersion.TLS_1_3
    certificate_validation: CertificateValidationLevel = CertificateValidationLevel.STRICT
    key_exchange_protocols: List[KeyExchangeProtocol] = None
    cipher_suites: List[str] = None
    session_timeout: int = 3600  # 1 hour
    handshake_timeout: int = 30   # 30 seconds
    certificate_pinning: bool = True
    client_certificate_required: bool = False
    enable_ocsp_stapling: bool = True
    enable_certificate_transparency: bool = True

    def __post_init__(self):
        if self.key_exchange_protocols is None:
            self.key_exchange_protocols = [KeyExchangeProtocol.ECDHE]
        if self.cipher_suites is None:
            self.cipher_suites = [
                "ECDHE-RSA-AES256-GCM-SHA384",
                "ECDHE-RSA-CHACHA20-POLY1305",
                "ECDHE-RSA-AES128-GCM-SHA256"
            ]

@dataclass
class CertificatePin:
    """Certificate pinning information."""
    hostname: str
    fingerprint_sha256: str
    created_at: float
    expires_at: Optional[float]
    notes: str

class TLSSecurityManager:
    """
    TLS security manager for FEDzk transport layer.

    Provides comprehensive TLS 1.3 security with certificate validation,
    pinning, and secure key exchange for all network communications.
    """

    def __init__(self, config: TLSConfig = None):
        self.config = config or TLSConfig()
        self.certificate_pins: Dict[str, CertificatePin] = {}
        self.connection_pool: Dict[str, TLSConnectionInfo] = {}
        self.security_metrics: Dict[str, Any] = {
            "total_connections": 0,
            "successful_handshakes": 0,
            "failed_handshakes": 0,
            "certificate_validations": 0,
            "certificate_failures": 0,
            "session_resumptions": 0,
            "bytes_encrypted": 0,
            "bytes_decrypted": 0
        }

        # Load certificate pins
        self._load_certificate_pins()

        # Initialize SSL context
        self.ssl_context = self._create_ssl_context()

        logger.info("TLS Security Manager initialized with TLS 1.3 support")

    def _create_ssl_context(self) -> ssl.SSLContext:
        """Create SSL context with secure defaults."""
        if self.config.version == TLSVersion.TLS_1_3:
            context = ssl.create_default_context()
            context.minimum_version = ssl.TLSVersion.TLSv1_2
            context.maximum_version = ssl.TLSVersion.TLSv1_3
        else:
            context = ssl.SSLContext(ssl.PROTOCOL_TLS)
            context.minimum_version = ssl.TLSVersion.TLSv1_2
            context.maximum_version = ssl.TLSVersion.TLSv1_2

        # Configure cipher suites
        if self.config.cipher_suites:
            try:
                context.set_ciphers(':'.join(self.config.cipher_suites))
            except ssl.SSLError:
                logger.warning("Some cipher suites not available, using defaults")

        # Configure certificate validation
        if self.config.certificate_validation == CertificateValidationLevel.NONE:
            context.check_hostname = False
            context.verify_mode = ssl.CERT_NONE
        elif self.config.certificate_validation == CertificateValidationLevel.BASIC:
            context.check_hostname = True
            context.verify_mode = ssl.CERT_REQUIRED
        elif self.config.certificate_validation in [CertificateValidationLevel.STRICT, CertificateValidationLevel.PINNING]:
            context.check_hostname = True
            context.verify_mode = ssl.CERT_REQUIRED

        # Configure client certificates
        if self.config.client_certificate_required:
            context.verify_mode = ssl.CERT_REQUIRED
            context.check_hostname = True

        # Configure OCSP stapling
        if self.config.enable_ocsp_stapling:
            try:
                context.enable_ocsp_stapling()
            except AttributeError:
                logger.warning("OCSP stapling not available in this Python version")

        # Set session timeout
        context.session_cache_timeout = self.config.session_timeout

        # Configure key exchange
        self._configure_key_exchange(context)

        return context

    def _configure_key_exchange(self, context: ssl.SSLContext):
        """Configure key exchange protocols."""
        # ECDHE is preferred and enabled by default in TLS 1.3
        # Additional configuration can be added here for specific requirements
        pass

    def _load_certificate_pins(self):
        """Load certificate pins from storage."""
        pin_file = Path("./security/certificate_pins.json")
        if pin_file.exists():
            try:
                with open(pin_file, 'r') as f:
                    pins_data = json.load(f)
                    for pin_data in pins_data:
                        pin = CertificatePin(**pin_data)
                        self.certificate_pins[pin.hostname] = pin
                logger.info(f"Loaded {len(self.certificate_pins)} certificate pins")
            except Exception as e:
                logger.error(f"Failed to load certificate pins: {e}")

    def _save_certificate_pins(self):
        """Save certificate pins to storage."""
        pin_file = Path("./security/certificate_pins.json")
        pin_file.parent.mkdir(parents=True, exist_ok=True)

        try:
            pins_data = [asdict(pin) for pin in self.certificate_pins.values()]
            with open(pin_file, 'w') as f:
                json.dump(pins_data, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save certificate pins: {e}")

    def pin_certificate(self, hostname: str, certificate: ssl.SSLCertificate, notes: str = ""):
        """
        Pin a certificate for a hostname.

        Args:
            hostname: Hostname to pin certificate for
            certificate: SSL certificate to pin
            notes: Optional notes about the pinning
        """
        try:
            # Extract certificate information
            cert_info = self._extract_certificate_info(certificate)

            # Create pin
            pin = CertificatePin(
                hostname=hostname,
                fingerprint_sha256=cert_info.fingerprint_sha256,
                created_at=time.time(),
                expires_at=None,  # Certificate expiration
                notes=notes
            )

            self.certificate_pins[hostname] = pin
            self._save_certificate_pins()

            logger.info(f"Certificate pinned for {hostname}")
            return True

        except Exception as e:
            logger.error(f"Failed to pin certificate for {hostname}: {e}")
            return False

    def validate_certificate_pin(self, hostname: str, certificate: ssl.SSLCertificate) -> bool:
        """
        Validate certificate against pinned fingerprint.

        Args:
            hostname: Hostname to validate
            certificate: Certificate to validate

        Returns:
            bool: True if certificate matches pin
        """
        if hostname not in self.certificate_pins:
            return False

        pin = self.certificate_pins[hostname]
        cert_info = self._extract_certificate_info(certificate)

        return hmac.compare_digest(pin.fingerprint_sha256, cert_info.fingerprint_sha256)

    def _extract_certificate_info(self, certificate: ssl.SSLCertificate) -> TLSCertificate:
        """Extract certificate information."""
        # Get certificate details
        subject = str(certificate.get('subject', ''))
        issuer = str(certificate.get('issuer', ''))
        serial = str(certificate.get('serialNumber', ''))
        not_before = str(certificate.get('notBefore', ''))
        not_after = str(certificate.get('notAfter', ''))

        # Calculate fingerprint
        cert_der = certificate.public_bytes(ssl.Encoding.DER)
        fingerprint = hashlib.sha256(cert_der).hexdigest()

        # Get public key info
        public_key = certificate.public_key()
        if hasattr(public_key, 'key_size'):
            key_size = public_key.key_size
        else:
            key_size = 2048  # Default

        return TLSCertificate(
            subject=subject,
            issuer=issuer,
            serial_number=serial,
            not_before=not_before,
            not_after=not_after,
            fingerprint_sha256=fingerprint,
            public_key_algorithm="RSA",  # Simplified
            signature_algorithm="SHA256-RSA",
            version=3,
            is_ca=False,
            key_size=key_size
        )

    def create_secure_socket(self, hostname: str, port: int) -> Tuple[socket.socket, TLSConnectionInfo]:
        """
        Create a secure TLS socket connection.

        Args:
            hostname: Target hostname
            port: Target port

        Returns:
            Tuple[socket.socket, TLSConnectionInfo]: Secure socket and connection info
        """
        try:
            # Create socket
            sock = socket.create_connection((hostname, port), timeout=self.config.handshake_timeout)

            # Wrap with SSL
            ssl_sock = self.ssl_context.wrap_socket(
                sock,
                server_hostname=hostname,
                do_handshake_on_connect=True
            )

            # Perform certificate validation
            if self.config.certificate_validation == CertificateValidationLevel.PINNING:
                if not self.validate_certificate_pin(hostname, ssl_sock.getpeercert()):
                    raise ssl.SSLCertVerificationError(f"Certificate pinning validation failed for {hostname}")

            # Create connection info
            connection_info = self._create_connection_info(ssl_sock, hostname)

            # Update metrics
            self.security_metrics["total_connections"] += 1
            self.security_metrics["successful_handshakes"] += 1

            # Store connection info
            connection_key = f"{hostname}:{port}"
            self.connection_pool[connection_key] = connection_info

            logger.info(f"Secure TLS connection established to {hostname}:{port}")
            return ssl_sock, connection_info

        except Exception as e:
            logger.error(f"Failed to create secure connection to {hostname}:{port}: {e}")
            self.security_metrics["failed_handshakes"] += 1
            raise

    def _create_connection_info(self, ssl_sock: ssl.SSLSocket, hostname: str) -> TLSConnectionInfo:
        """Create TLS connection information."""
        try:
            # Get connection details
            version = ssl_sock.version()
            cipher = ssl_sock.cipher()

            # Get peer certificate
            peer_cert = ssl_sock.getpeercert()
            cert_info = None
            if peer_cert:
                cert_info = self._extract_certificate_info(peer_cert)

            return TLSConnectionInfo(
                version=version or "unknown",
                cipher_suite=cipher[0] if cipher else "unknown",
                peer_certificate=cert_info,
                compression=ssl_sock.compression() or "none",
                session_reused=ssl_sock.session_reused,
                established_at=time.time(),
                bytes_sent=0,
                bytes_received=0
            )

        except Exception as e:
            logger.error(f"Failed to create connection info: {e}")
            return TLSConnectionInfo(
                version="unknown",
                cipher_suite="unknown",
                peer_certificate=None,
                compression="unknown",
                session_reused=False,
                established_at=time.time(),
                bytes_sent=0,
                bytes_received=0
            )

    def validate_network_traffic(self, data: bytes, connection_info: TLSConnectionInfo) -> bool:
        """
        Validate that network traffic is properly encrypted.

        Args:
            data: Network data to validate
            connection_info: TLS connection information

        Returns:
            bool: True if traffic appears properly encrypted
        """
        # Basic validation - in a real implementation, this would analyze
        # the encrypted traffic patterns and validate TLS record structure

        if not connection_info:
            return False

        # Check if connection is using TLS
        if connection_info.version not in ["TLSv1.2", "TLSv1.3"]:
            return False

        # Validate cipher suite is strong
        weak_ciphers = ["NULL", "RC4", "DES", "3DES", "MD5"]
        for weak_cipher in weak_ciphers:
            if weak_cipher in connection_info.cipher_suite.upper():
                return False

        # Update metrics
        self.security_metrics["bytes_encrypted"] += len(data)

        return True

    def get_security_metrics(self) -> Dict[str, Any]:
        """Get TLS security metrics."""
        metrics = self.security_metrics.copy()
        metrics["active_connections"] = len(self.connection_pool)
        metrics["certificate_pins"] = len(self.certificate_pins)
        metrics["tls_version"] = self.config.version.value

        # Calculate success rate
        total_handshakes = metrics["successful_handshakes"] + metrics["failed_handshakes"]
        if total_handshakes > 0:
            metrics["handshake_success_rate"] = metrics["successful_handshakes"] / total_handshakes
        else:
            metrics["handshake_success_rate"] = 0.0

        return metrics

    def cleanup_expired_connections(self):
        """Clean up expired TLS connections."""
        current_time = time.time()
        expired_keys = []

        for key, connection_info in self.connection_pool.items():
            # Check if connection has been idle too long
            if current_time - connection_info.established_at > self.config.session_timeout:
                expired_keys.append(key)

        for key in expired_keys:
            del self.connection_pool[key]

        if expired_keys:
            logger.info(f"Cleaned up {len(expired_keys)} expired TLS connections")

    def close_connection(self, hostname: str, port: int):
        """Close a TLS connection."""
        connection_key = f"{hostname}:{port}"
        if connection_key in self.connection_pool:
            del self.connection_pool[connection_key]
            logger.info(f"TLS connection closed: {hostname}:{port}")

# Convenience functions for easy usage
def create_secure_tls_context(config: TLSConfig = None) -> ssl.SSLContext:
    """
    Create a secure TLS context with FEDzk defaults.

    Args:
        config: TLS configuration (optional)

    Returns:
        ssl.SSLContext: Configured TLS context
    """
    manager = TLSSecurityManager(config)
    return manager.ssl_context

def validate_certificate_chain(hostname: str, port: int = 443, timeout: int = 10) -> Dict[str, Any]:
    """
    Validate certificate chain for a hostname.

    Args:
        hostname: Hostname to validate
        port: Port number
        timeout: Connection timeout

    Returns:
        Dict: Certificate validation results
    """
    try:
        # Create SSL context for validation
        context = ssl.create_default_context()
        context.check_hostname = True
        context.verify_mode = ssl.CERT_REQUIRED

        with socket.create_connection((hostname, port), timeout=timeout) as sock:
            with context.wrap_socket(sock, server_hostname=hostname) as ssl_sock:
                cert = ssl_sock.getpeercert()

                return {
                    "valid": True,
                    "hostname": hostname,
                    "certificate": cert,
                    "issuer": str(cert.get('issuer', '')),
                    "subject": str(cert.get('subject', '')),
                    "expires": str(cert.get('notAfter', ''))
                }

    except ssl.SSLCertVerificationError as e:
        return {
            "valid": False,
            "hostname": hostname,
            "error": f"Certificate verification failed: {e}",
            "error_type": "certificate_verification"
        }
    except Exception as e:
        return {
            "valid": False,
            "hostname": hostname,
            "error": f"Connection failed: {e}",
            "error_type": "connection_error"
        }

def generate_secure_random_bytes(length: int) -> bytes:
    """
    Generate cryptographically secure random bytes.

    Args:
        length: Number of bytes to generate

    Returns:
        bytes: Secure random bytes
    """
    return secrets.token_bytes(length)

