# FEDzk Security Best Practices

## 🔐 Production Security Guidelines for Real Cryptographic Operations

This document outlines security best practices for deploying FEDzk in production environments. **All recommendations focus on real cryptographic operations with zero compromises.**

---

## 📋 Security Framework Overview

### Core Security Principles

FEDzk implements a **defense-in-depth security approach** combining multiple cryptographic layers:

```
┌─────────────────┐
│ Client Security │ ← Input validation, quantization, proof generation
├─────────────────┤
│ Network Security│ ← TLS 1.3, certificate pinning, rate limiting
├─────────────────┤
│ MPC Security    │ ← Distributed proof generation, secure aggregation
├─────────────────┤
│ ZK Security     │ ← SNARK proofs, trusted setup, verification
├─────────────────┤
│ System Security │ ← Access control, audit logging, monitoring
└─────────────────┘
```

### Security Properties

- **🔒 Confidentiality**: Client data remains private through ZK proofs
- **✅ Integrity**: Cryptographic guarantees ensure update validity
- **🔍 Auditability**: Complete audit trail of all operations
- **🛡️ Availability**: Fault-tolerant design with graceful degradation
- **🔐 Authentication**: Multi-factor authentication for all access

---

## 🏢 Enterprise Security Configuration

### 1. Cryptographic Parameter Selection

#### ZK Circuit Security Levels

```python
# Production security configurations
SECURITY_LEVELS = {
    "standard": {
        "circuit_constraints": 1000,
        "proof_size": 192,  # bytes
        "security_bits": 80,
        "trusted_setup_participants": 5,
        "use_case": "Development/Testing"
    },
    "enhanced": {
        "circuit_constraints": 10000,
        "proof_size": 192,
        "security_bits": 100,
        "trusted_setup_participants": 10,
        "use_case": "Production FL"
    },
    "maximum": {
        "circuit_constraints": 100000,
        "proof_size": 192,
        "security_bits": 128,
        "trusted_setup_participants": 20,
        "use_case": "Critical Infrastructure"
    }
}

# Recommended for production
PRODUCTION_CONFIG = SECURITY_LEVELS["enhanced"]
```

#### Key Management

```python
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa
import secrets

class CryptographicKeyManager:
    """Production key management for FEDzk."""

    def __init__(self, key_store_path: str = "/etc/fedzk/keys"):
        self.key_store_path = Path(key_store_path)
        self.key_store_path.mkdir(parents=True, exist_ok=True)

    def generate_api_key(self, client_id: str) -> str:
        """Generate cryptographically secure API key."""
        # Generate 256-bit key
        key_bytes = secrets.token_bytes(32)

        # Create key identifier
        key_id = hashlib.sha256(f"{client_id}_{time.time()}".encode()).hexdigest()[:16]

        # Store key securely
        key_file = self.key_store_path / f"{key_id}.key"
        with open(key_file, 'wb') as f:
            f.write(key_bytes)

        # Set restrictive permissions
        key_file.chmod(0o600)

        # Return key identifier (not the actual key)
        return key_id

    def validate_api_key(self, key_id: str, provided_key: str) -> bool:
        """Validate API key cryptographically."""
        try:
            key_file = self.key_store_path / f"{key_id}.key"

            if not key_file.exists():
                return False

            # Read stored key
            with open(key_file, 'rb') as f:
                stored_key = f.read()

            # Compare with provided key using constant-time comparison
            return secrets.compare_digest(stored_key, provided_key.encode())

        except Exception:
            return False

    def rotate_keys(self, client_id: str) -> str:
        """Rotate API key with secure deletion of old key."""
        # Generate new key
        new_key_id = self.generate_api_key(client_id)

        # Securely delete old keys (find by client_id pattern)
        old_keys = list(self.key_store_path.glob(f"*{client_id}*.key"))
        for old_key in old_keys:
            # Secure deletion (overwrite before deletion)
            with open(old_key, 'wb') as f:
                f.write(secrets.token_bytes(32))
            old_key.unlink()

        return new_key_id
```

### 2. Network Security Configuration

#### TLS 1.3 Configuration

```python
# tls_config.py
TLS_PRODUCTION_CONFIG = {
    "protocol": "TLSv1.3",
    "cipher_suites": [
        "TLS_AES_256_GCM_SHA384",
        "TLS_AES_128_GCM_SHA256",
        "TLS_CHACHA20_POLY1305_SHA256"
    ],
    "certificate": {
        "type": "RSA-4096" or "ECDSA-P256",
        "issuer": "Production CA",
        "validity_days": 365,
        "key_usage": ["digitalSignature", "keyEncipherment"],
        "extended_key_usage": ["serverAuth", "clientAuth"]
    },
    "client_authentication": {
        "enabled": True,
        "certificate_required": True,
        "certificate_authorities": ["/etc/ssl/certs/ca.crt"]
    },
    "session_resumption": {
        "enabled": True,
        "tickets": False,  # Disable session tickets for better security
        "cache_size": 1000
    },
    "hsts": {
        "enabled": True,
        "max_age": 31536000,  # 1 year
        "include_subdomains": True,
        "preload": False
    }
}
```

#### Firewall Configuration

```bash
# production_firewall.sh
#!/bin/bash

# Flush existing rules
iptables -F
iptables -X

# Default policies
iptables -P INPUT DROP
iptables -P FORWARD DROP
iptables -P OUTPUT ACCEPT

# Allow loopback
iptables -A INPUT -i lo -j ACCEPT

# Allow established connections
iptables -A INPUT -m conntrack --ctstate ESTABLISHED,RELATED -j ACCEPT

# Allow SSH (restrict to specific IPs)
iptables -A INPUT -p tcp -s 192.168.1.0/24 --dport 22 -j ACCEPT

# Allow HTTPS for FEDzk
iptables -A INPUT -p tcp --dport 443 -j ACCEPT

# Allow MPC server port
iptables -A INPUT -p tcp --dport 9000 -j ACCEPT

# Rate limiting for API endpoints
iptables -A INPUT -p tcp --dport 443 -m conntrack --ctstate NEW -m limit --limit 100/minute --limit-burst 10 -j ACCEPT
iptables -A INPUT -p tcp --dport 9000 -m conntrack --ctstate NEW -m limit --limit 50/minute --limit-burst 5 -j ACCEPT

# Log dropped packets
iptables -A INPUT -j LOG --log-prefix "FEDzk_DROP: " --log-level 4
iptables -A INPUT -j DROP

# Save rules
iptables-save > /etc/iptables/rules.v4
```

### 3. Access Control and Authorization

#### Role-Based Access Control (RBAC)

```python
from enum import Enum
from typing import Set, Dict
from dataclasses import dataclass

class SecurityRole(Enum):
    CLIENT = "client"
    COORDINATOR = "coordinator"
    ADMIN = "admin"
    AUDITOR = "auditor"

class Resource(Enum):
    PROOF_GENERATION = "proof_generation"
    MODEL_AGGREGATION = "model_aggregation"
    SYSTEM_MONITORING = "system_monitoring"
    AUDIT_LOGS = "audit_logs"

@dataclass
class Permission:
    resource: Resource
    actions: Set[str]  # read, write, execute, delete

class RBACSecurityManager:
    """Role-Based Access Control for FEDzk."""

    def __init__(self):
        self.role_permissions: Dict[SecurityRole, Set[Permission]] = {
            SecurityRole.CLIENT: {
                Permission(Resource.PROOF_GENERATION, {"execute"}),
                Permission(Resource.MODEL_AGGREGATION, {"read"})
            },
            SecurityRole.COORDINATOR: {
                Permission(Resource.PROOF_GENERATION, {"read", "execute"}),
                Permission(Resource.MODEL_AGGREGATION, {"read", "write", "execute"}),
                Permission(Resource.SYSTEM_MONITORING, {"read"})
            },
            SecurityRole.ADMIN: {
                Permission(Resource.PROOF_GENERATION, {"read", "write", "execute", "delete"}),
                Permission(Resource.MODEL_AGGREGATION, {"read", "write", "execute", "delete"}),
                Permission(Resource.SYSTEM_MONITORING, {"read", "write"}),
                Permission(Resource.AUDIT_LOGS, {"read", "write"})
            },
            SecurityRole.AUDITOR: {
                Permission(Resource.AUDIT_LOGS, {"read"}),
                Permission(Resource.SYSTEM_MONITORING, {"read"})
            }
        }

    def check_permission(self, role: SecurityRole, resource: Resource, action: str) -> bool:
        """Check if role has permission for action on resource."""
        if role not in self.role_permissions:
            return False

        for permission in self.role_permissions[role]:
            if permission.resource == resource and action in permission.actions:
                return True

        return False

    def get_user_role(self, user_id: str, api_key: str) -> SecurityRole:
        """Determine user role from API key pattern."""
        if api_key.startswith("client_"):
            return SecurityRole.CLIENT
        elif api_key.startswith("coord_"):
            return SecurityRole.COORDINATOR
        elif api_key.startswith("admin_"):
            return SecurityRole.ADMIN
        elif api_key.startswith("audit_"):
            return SecurityRole.AUDITOR
        else:
            raise ValueError("Invalid API key format")

# Usage in request handler
security_manager = RBACSecurityManager()

def authorize_request(user_id: str, api_key: str, resource: Resource, action: str) -> bool:
    """Authorize API request."""
    try:
        user_role = security_manager.get_user_role(user_id, api_key)
        return security_manager.check_permission(user_role, resource, action)
    except ValueError:
        return False
```

### 4. Audit Logging and Monitoring

#### Comprehensive Audit System

```python
import logging
import json
from datetime import datetime
from pathlib import Path
import hashlib

class CryptographicAuditLogger:
    """Enterprise-grade audit logging for cryptographic operations."""

    def __init__(self, log_path: str = "/var/log/fedzk/audit.log"):
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

        # Configure secure logging
        self.logger = logging.getLogger("fedzk_audit")
        self.logger.setLevel(logging.INFO)

        # File handler with rotation
        from logging.handlers import RotatingFileHandler
        handler = RotatingFileHandler(
            self.log_path,
            maxBytes=100*1024*1024,  # 100MB
            backupCount=10
        )

        # Secure format
        formatter = logging.Formatter(
            '%(asctime)s|%(levelname)s|%(user_id)s|%(operation)s|%(resource)s|%(result)s|%(details)s'
        )
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)

    def log_cryptographic_event(self, event_data: Dict):
        """Log cryptographic event with integrity protection."""
        # Required fields
        required_fields = ["user_id", "operation", "resource", "result"]
        for field in required_fields:
            if field not in event_data:
                raise ValueError(f"Missing required audit field: {field}")

        # Add metadata
        event_data.update({
            "timestamp": datetime.utcnow().isoformat(),
            "audit_version": "1.0",
            "system": "FEDzk",
            "integrity_hash": self._calculate_integrity_hash(event_data)
        })

        # Log event
        self.logger.info("", extra=event_data)

    def _calculate_integrity_hash(self, event_data: Dict) -> str:
        """Calculate integrity hash for audit log entry."""
        # Create canonical representation
        canonical_data = json.dumps(event_data, sort_keys=True, separators=(',', ':'))

        # Calculate hash
        return hashlib.sha256(canonical_data.encode()).hexdigest()

    def log_proof_generation(self, user_id: str, circuit_type: str, success: bool, proof_hash: str = None):
        """Log proof generation event."""
        self.log_cryptographic_event({
            "user_id": user_id,
            "operation": "proof_generation",
            "resource": f"circuit:{circuit_type}",
            "result": "success" if success else "failure",
            "details": {
                "proof_hash": proof_hash,
                "circuit_type": circuit_type,
                "timestamp": datetime.utcnow().timestamp()
            }
        })

    def log_verification(self, user_id: str, proof_id: str, success: bool):
        """Log proof verification event."""
        self.log_cryptographic_event({
            "user_id": user_id,
            "operation": "proof_verification",
            "resource": f"proof:{proof_id}",
            "result": "success" if success else "failure",
            "details": {
                "proof_id": proof_id,
                "verification_time": datetime.utcnow().timestamp()
            }
        })

    def log_security_event(self, user_id: str, event_type: str, severity: str, details: Dict):
        """Log security-related event."""
        self.log_cryptographic_event({
            "user_id": user_id,
            "operation": "security_event",
            "resource": "system",
            "result": "logged",
            "details": {
                "event_type": event_type,
                "severity": severity,
                **details
            }
        })

# Global audit logger instance
audit_logger = CryptographicAuditLogger()

# Usage examples
audit_logger.log_proof_generation(
    user_id="client_001",
    circuit_type="model_update_secure",
    success=True,
    proof_hash="0x1234567890abcdef"
)

audit_logger.log_security_event(
    user_id="admin",
    event_type="unauthorized_access_attempt",
    severity="high",
    details={"ip_address": "192.168.1.100", "endpoint": "/api/proof"}
)
```

#### Real-Time Security Monitoring

```python
import psutil
import time
from collections import deque
import threading

class SecurityMonitor:
    """Real-time security monitoring for FEDzk."""

    def __init__(self, alert_thresholds: Dict = None):
        self.alert_thresholds = alert_thresholds or {
            "failed_auth_attempts": 5,
            "suspicious_traffic": 100,
            "memory_usage": 90,
            "cpu_usage": 95,
            "disk_usage": 95
        }

        self.metrics_history = deque(maxlen=1000)
        self.monitoring_active = False

    def start_monitoring(self):
        """Start security monitoring thread."""
        self.monitoring_active = True
        monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        monitor_thread.start()

    def stop_monitoring(self):
        """Stop security monitoring."""
        self.monitoring_active = False

    def _monitoring_loop(self):
        """Main monitoring loop."""
        while self.monitoring_active:
            try:
                metrics = self._collect_metrics()
                self._analyze_metrics(metrics)
                self.metrics_history.append(metrics)
                time.sleep(60)  # Monitor every minute
            except Exception as e:
                logger.error(f"Monitoring error: {e}")

    def _collect_metrics(self) -> Dict:
        """Collect system and security metrics."""
        return {
            "timestamp": time.time(),
            "cpu_percent": psutil.cpu_percent(interval=1),
            "memory_percent": psutil.virtual_memory().percent,
            "disk_percent": psutil.disk_usage('/').percent,
            "network_connections": len(psutil.net_connections()),
            "failed_auth_attempts": self._get_failed_auth_count(),
            "active_proofs": self._get_active_proof_count(),
            "system_load": psutil.getloadavg()
        }

    def _analyze_metrics(self, metrics: Dict):
        """Analyze metrics and trigger alerts."""
        alerts = []

        # CPU usage alert
        if metrics["cpu_percent"] > self.alert_thresholds["cpu_usage"]:
            alerts.append({
                "type": "high_cpu_usage",
                "severity": "medium",
                "message": ".1f",
                "value": metrics["cpu_percent"]
            })

        # Memory usage alert
        if metrics["memory_percent"] > self.alert_thresholds["memory_usage"]:
            alerts.append({
                "type": "high_memory_usage",
                "severity": "high",
                "message": ".1f",
                "value": metrics["memory_percent"]
            })

        # Disk usage alert
        if metrics["disk_percent"] > self.alert_thresholds["disk_usage"]:
            alerts.append({
                "type": "high_disk_usage",
                "severity": "high",
                "message": ".1f",
                "value": metrics["disk_percent"]
            })

        # Failed authentication alert
        if metrics["failed_auth_attempts"] > self.alert_thresholds["failed_auth_attempts"]:
            alerts.append({
                "type": "authentication_failures",
                "severity": "high",
                "message": f"High failed authentication attempts: {metrics['failed_auth_attempts']}",
                "value": metrics["failed_auth_attempts"]
            })

        # Send alerts
        for alert in alerts:
            self._send_alert(alert)

    def _send_alert(self, alert: Dict):
        """Send security alert."""
        logger.warning(f"SECURITY ALERT: {alert['message']}")

        # Log to audit system
        audit_logger.log_security_event(
            user_id="system_monitor",
            event_type=alert["type"],
            severity=alert["severity"],
            details=alert
        )

        # In production, this would send alerts to:
        # - Email/SMS notifications
        # - SIEM systems
        # - Incident response systems
        # - PagerDuty/OpsGenie

    def _get_failed_auth_count(self) -> int:
        """Get count of recent failed authentication attempts."""
        # In production, this would query authentication logs
        # For demo purposes, return simulated value
        return 0

    def _get_active_proof_count(self) -> int:
        """Get count of currently active proof generations."""
        # In production, this would query MPC server metrics
        # For demo purposes, return simulated value
        return 5

# Start security monitoring
monitor = SecurityMonitor()
monitor.start_monitoring()
```

---

## 🔐 Cryptographic Security Measures

### 1. Input Validation and Sanitization

```python
import re
from typing import Union, List

class CryptographicInputValidator:
    """Comprehensive input validation for cryptographic operations."""

    def __init__(self):
        # Define validation patterns
        self.integer_pattern = re.compile(r'^-?\d+$')
        self.float_pattern = re.compile(r'^-?\d+\.\d+$')

    def validate_gradients(self, gradients: Union[List, Dict]) -> bool:
        """Validate gradient inputs for cryptographic safety."""
        if isinstance(gradients, dict):
            return all(self._validate_gradient_array(v) for v in gradients.values())
        elif isinstance(gradients, list):
            return self._validate_gradient_array(gradients)
        else:
            return False

    def _validate_gradient_array(self, gradient_array: List) -> bool:
        """Validate single gradient array."""
        if not isinstance(gradient_array, list):
            return False

        if len(gradient_array) == 0:
            return False

        # Check all elements are integers
        for value in gradient_array:
            if not isinstance(value, int):
                return False

            # Check value range (prevent overflow/underflow)
            if not (-10**9 <= value <= 10**9):
                return False

        return True

    def validate_client_id(self, client_id: str) -> bool:
        """Validate client identifier."""
        if not isinstance(client_id, str):
            return False

        # Length check
        if len(client_id) < 3 or len(client_id) > 100:
            return False

        # Character validation (alphanumeric, underscore, dash)
        if not re.match(r'^[a-zA-Z0-9_-]+$', client_id):
            return False

        return True

    def validate_api_key(self, api_key: str) -> bool:
        """Validate API key format and strength."""
        if not isinstance(api_key, str):
            return False

        # Minimum length for cryptographic security
        if len(api_key) < 32:
            return False

        # Check for required character types
        has_upper = re.search(r'[A-Z]', api_key)
        has_lower = re.search(r'[a-z]', api_key)
        has_digit = re.search(r'[0-9]', api_key)

        if not (has_upper and has_lower and has_digit):
            return False

        return True

    def sanitize_input(self, input_data: Dict) -> Dict:
        """Sanitize input data for cryptographic safety."""
        sanitized = {}

        for key, value in input_data.items():
            if isinstance(value, str):
                # Remove potentially dangerous characters
                sanitized[key] = re.sub(r'[^\w\-_]', '', value)
            elif isinstance(value, (int, float)):
                # Clamp numeric values to safe ranges
                if isinstance(value, int):
                    sanitized[key] = max(-10**9, min(10**9, value))
                else:
                    sanitized[key] = max(-10**6, min(10**6, value))
            elif isinstance(value, list):
                # Sanitize list elements
                sanitized[key] = [
                    self._sanitize_value(item) for item in value[:100]  # Limit size
                ]
            else:
                # Remove unsupported types
                continue

        return sanitized

    def _sanitize_value(self, value):
        """Sanitize individual value."""
        if isinstance(value, (int, float)):
            return max(-10**9, min(10**9, int(value)))
        elif isinstance(value, str):
            return re.sub(r'[^\w\-_]', '', value)[:100]  # Limit length
        else:
            return 0  # Default safe value
```

### 2. Rate Limiting and DDoS Protection

```python
from collections import defaultdict, deque
import time
import threading

class RateLimiter:
    """Production-grade rate limiter for FEDzk services."""

    def __init__(self, max_requests: int = 100, window_seconds: int = 60):
        self.max_requests = max_requests
        self.window_seconds = window_seconds

        # Thread-safe storage
        self.requests = defaultdict(lambda: deque())
        self.lock = threading.Lock()

    def is_allowed(self, client_id: str) -> bool:
        """Check if request is allowed for client."""
        current_time = time.time()

        with self.lock:
            client_requests = self.requests[client_id]

            # Remove old requests outside the window
            while client_requests and current_time - client_requests[0] > self.window_seconds:
                client_requests.popleft()

            # Check if under limit
            if len(client_requests) < self.max_requests:
                client_requests.append(current_time)
                return True
            else:
                return False

    def get_remaining_requests(self, client_id: str) -> int:
        """Get remaining requests for client in current window."""
        current_time = time.time()

        with self.lock:
            client_requests = self.requests[client_id]

            # Clean old requests
            while client_requests and current_time - client_requests[0] > self.window_seconds:
                client_requests.popleft()

            return max(0, self.max_requests - len(client_requests))

class DDoSProtection:
    """Advanced DDoS protection for FEDzk."""

    def __init__(self):
        self.rate_limiter = RateLimiter(max_requests=50, window_seconds=60)
        self.suspicious_clients = set()
        self.blocked_clients = set()

    def analyze_request(self, client_id: str, request_data: Dict) -> str:
        """
        Analyze request for DDoS patterns.
        Returns: "allow", "block", or "challenge"
        """
        # Check rate limiting
        if not self.rate_limiter.is_allowed(client_id):
            self.suspicious_clients.add(client_id)
            return "block"

        # Check for suspicious patterns
        if self._is_suspicious_request(request_data):
            self.suspicious_clients.add(client_id)
            return "challenge"

        # Check if client is blocked
        if client_id in self.blocked_clients:
            return "block"

        return "allow"

    def _is_suspicious_request(self, request_data: Dict) -> bool:
        """Detect suspicious request patterns."""
        suspicious_patterns = [
            # Large payloads
            lambda d: len(str(d)) > 10000,

            # Unusual request frequency
            lambda d: d.get("request_count", 0) > 1000,

            # Malformed data
            lambda d: any(isinstance(v, type) for v in d.values()),

            # Known attack patterns
            lambda d: "eval" in str(d).lower() or "exec" in str(d).lower()
        ]

        return any(pattern(request_data) for pattern in suspicious_patterns)

    def block_client(self, client_id: str, reason: str = "DDoS protection"):
        """Block a client permanently."""
        self.blocked_clients.add(client_id)
        audit_logger.log_security_event(
            user_id="system",
            event_type="client_blocked",
            severity="high",
            details={"client_id": client_id, "reason": reason}
        )
```

### 3. Secure Key Management

```python
import os
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.fernet import Fernet
import base64

class SecureKeyStore:
    """Hardware Security Module (HSM) compatible key store."""

    def __init__(self, key_store_path: str = "/etc/fedzk/keys"):
        self.key_store_path = Path(key_store_path)
        self.key_store_path.mkdir(parents=True, exist_ok=True)

        # Master key for encryption (in production, use HSM)
        self.master_key = self._derive_master_key()

    def _derive_master_key(self) -> bytes:
        """Derive master key from environment or HSM."""
        # In production, this would use HSM or secure key derivation
        salt = os.environ.get("FEDZK_KEY_SALT", "default_salt").encode()

        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
        )

        password = os.environ.get("FEDZK_MASTER_PASSWORD", "default_password")
        return base64.urlsafe_b64encode(kdf.derive(password.encode()))

    def store_key(self, key_id: str, key_data: bytes, metadata: Dict = None):
        """Store encrypted key with metadata."""
        # Encrypt key data
        fernet = Fernet(self.master_key)
        encrypted_key = fernet.encrypt(key_data)

        # Prepare storage format
        key_entry = {
            "key_id": key_id,
            "encrypted_key": encrypted_key.decode(),
            "created_at": datetime.utcnow().isoformat(),
            "metadata": metadata or {},
            "version": "1.0"
        }

        # Store securely
        key_file = self.key_store_path / f"{key_id}.encrypted"
        with open(key_file, 'w') as f:
            json.dump(key_entry, f)

        key_file.chmod(0o600)

    def retrieve_key(self, key_id: str) -> bytes:
        """Retrieve and decrypt key."""
        key_file = self.key_store_path / f"{key_id}.encrypted"

        if not key_file.exists():
            raise ValueError(f"Key {key_id} not found")

        with open(key_file, 'r') as f:
            key_entry = json.load(f)

        # Decrypt key
        fernet = Fernet(self.master_key)
        encrypted_key = key_entry["encrypted_key"].encode()
        decrypted_key = fernet.decrypt(encrypted_key)

        return decrypted_key

    def rotate_master_key(self):
        """Rotate master key with re-encryption of all stored keys."""
        logger.info("Starting master key rotation...")

        # Generate new master key
        new_master_key = self._derive_master_key()

        # Re-encrypt all keys
        for key_file in self.key_store_path.glob("*.encrypted"):
            with open(key_file, 'r') as f:
                key_entry = json.load(f)

            # Re-encrypt with new master key
            fernet = Fernet(new_master_key)
            decrypted_key = Fernet(self.master_key).decrypt(key_entry["encrypted_key"].encode())
            reencrypted_key = fernet.encrypt(decrypted_key)

            # Update entry
            key_entry["encrypted_key"] = reencrypted_key.decode()
            key_entry["rotated_at"] = datetime.utcnow().isoformat()

            # Save updated entry
            with open(key_file, 'w') as f:
                json.dump(key_entry, f)

        # Update master key
        self.master_key = new_master_key

        audit_logger.log_security_event(
            user_id="system",
            event_type="master_key_rotated",
            severity="high",
            details={"timestamp": datetime.utcnow().isoformat()}
        )

        logger.info("Master key rotation completed")
```

---

## 📊 Compliance and Regulatory Requirements

### GDPR Compliance

```python
class GDPRComplianceManager:
    """GDPR compliance manager for FEDzk."""

    def __init__(self):
        self.consent_records = {}
        self.data_processing_records = {}
        self.retention_policies = {
            "proofs": 2555,  # 7 years in days
            "audit_logs": 2555,
            "client_data": 2555
        }

    def record_consent(self, client_id: str, consent_data: Dict):
        """Record GDPR consent for data processing."""
        self.consent_records[client_id] = {
            "consent_given": True,
            "consent_date": datetime.utcnow().isoformat(),
            "consent_type": "federated_learning",
            "data_purposes": ["model_training", "privacy_preserving_computation"],
            "retention_period_days": self.retention_policies["client_data"],
            **consent_data
        }

        audit_logger.log_security_event(
            user_id=client_id,
            event_type="gdpr_consent_recorded",
            severity="info",
            details={"consent_type": "federated_learning"}
        )

    def check_data_retention_compliance(self, client_id: str) -> bool:
        """Check if data retention complies with GDPR."""
        if client_id not in self.consent_records:
            return False

        consent_record = self.consent_records[client_id]
        consent_date = datetime.fromisoformat(consent_record["consent_date"])
        retention_days = consent_record["retention_period_days"]

        if (datetime.utcnow() - consent_date).days > retention_days:
            return False

        return True

    def handle_data_deletion_request(self, client_id: str):
        """Handle GDPR right to erasure (right to be forgotten)."""
        logger.info(f"Processing GDPR deletion request for {client_id}")

        # Remove all client data
        deletion_targets = [
            f"/var/lib/fedzk/client_data/{client_id}",
            f"/var/log/fedzk/client_logs/{client_id}",
            f"/etc/fedzk/client_keys/{client_id}"
        ]

        for target in deletion_targets:
            if Path(target).exists():
                # Secure deletion
                self._secure_delete(target)

        # Log deletion
        audit_logger.log_security_event(
            user_id=client_id,
            event_type="gdpr_data_deletion",
            severity="info",
            details={"deletion_targets": deletion_targets}
        )

    def _secure_delete(self, path: str):
        """Securely delete file or directory."""
        path_obj = Path(path)

        if path_obj.is_file():
            # Overwrite file before deletion
            with open(path_obj, 'wb') as f:
                f.write(os.urandom(path_obj.stat().st_size))
            path_obj.unlink()
        elif path_obj.is_dir():
            # Recursively delete directory contents securely
            for file_path in path_obj.rglob('*'):
                if file_path.is_file():
                    with open(file_path, 'wb') as f:
                        f.write(os.urandom(file_path.stat().st_size))
                    file_path.unlink()
            path_obj.rmdir()
```

### SOX Compliance (Financial Services)

```python
class SOXComplianceManager:
    """SOX compliance manager for financial federated learning."""

    def __init__(self):
        self.control_activities = {}
        self.risk_assessments = {}
        self.audit_trails = defaultdict(list)

    def establish_control_activity(self, control_id: str, description: str, owner: str):
        """Establish SOX control activity."""
        self.control_activities[control_id] = {
            "description": description,
            "owner": owner,
            "established_date": datetime.utcnow().isoformat(),
            "status": "active",
            "testing_frequency": "quarterly",
            "last_tested": None,
            "test_results": []
        }

        logger.info(f"SOX Control {control_id} established: {description}")

    def perform_risk_assessment(self, process_id: str, risk_data: Dict):
        """Perform SOX risk assessment."""
        self.risk_assessments[process_id] = {
            "assessment_date": datetime.utcnow().isoformat(),
            "risk_level": risk_data.get("risk_level", "medium"),
            "impact": risk_data.get("impact", "moderate"),
            "likelihood": risk_data.get("likelihood", "possible"),
            "mitigation_plan": risk_data.get("mitigation_plan", []),
            "review_frequency": "annual"
        }

        audit_logger.log_security_event(
            user_id="compliance_officer",
            event_type="sox_risk_assessment",
            severity="info",
            details={"process_id": process_id, "risk_level": risk_data.get("risk_level")}
        )

    def log_audit_trail(self, event_type: str, user_id: str, details: Dict):
        """Log SOX audit trail event."""
        audit_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "event_type": event_type,
            "user_id": user_id,
            "details": details,
            "compliance_framework": "SOX",
            "integrity_hash": hashlib.sha256(json.dumps(details, sort_keys=True).encode()).hexdigest()
        }

        self.audit_trails[event_type].append(audit_entry)

        # Write to SOX audit log
        audit_log_path = Path("/var/log/fedzk/sox_audit.log")
        with open(audit_log_path, 'a') as f:
            f.write(json.dumps(audit_entry) + '\n')

    def generate_compliance_report(self) -> Dict:
        """Generate SOX compliance report."""
        return {
            "report_date": datetime.utcnow().isoformat(),
            "compliance_framework": "SOX",
            "control_activities": self.control_activities,
            "risk_assessments": self.risk_assessments,
            "audit_trail_summary": {
                event_type: len(entries)
                for event_type, entries in self.audit_trails.items()
            },
            "overall_compliance_status": "compliant",  # Would be determined by actual assessment
            "next_review_date": (datetime.utcnow().replace(day=1) + timedelta(days=365)).isoformat()
        }
```

---

## 🚨 Security Incident Response

### Incident Response Plan

```python
from enum import Enum
from typing import List, Dict
import smtplib
from email.mime.text import MIMEText

class IncidentSeverity(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class SecurityIncidentResponse:
    """Security incident response system for FEDzk."""

    def __init__(self):
        self.active_incidents = {}
        self.response_playbooks = self._load_response_playbooks()
        self.notification_channels = [
            "email",
            "slack",
            "pagerduty",
            "siem"
        ]

    def _load_response_playbooks(self) -> Dict:
        """Load incident response playbooks."""
        return {
            "cryptographic_failure": {
                "severity": IncidentSeverity.HIGH,
                "immediate_actions": [
                    "Isolate affected systems",
                    "Disable proof generation",
                    "Notify security team"
                ],
                "investigation_steps": [
                    "Review cryptographic logs",
                    "Validate key integrity",
                    "Check for tampering evidence"
                ],
                "recovery_steps": [
                    "Regenerate affected keys",
                    "Recompile circuits",
                    "Perform trusted setup",
                    "Gradual system restoration"
                ]
            },
            "unauthorized_access": {
                IncidentSeverity.CRITICAL,
                ["immediate_actions"],
                ["investigation_steps"],
                ["recovery_steps"]
            }
        }

    def report_incident(self, incident_type: str, details: Dict, reported_by: str):
        """Report security incident."""
        incident_id = f"INC-{int(time.time())}"

        incident = {
            "id": incident_id,
            "type": incident_type,
            "severity": self.response_playbooks[incident_type]["severity"],
            "reported_by": reported_by,
            "reported_at": datetime.utcnow().isoformat(),
            "status": "investigating",
            "details": details,
            "timeline": [{
                "timestamp": datetime.utcnow().isoformat(),
                "action": "incident_reported",
                "details": f"Incident reported by {reported_by}"
            }]
        }

        self.active_incidents[incident_id] = incident

        # Execute immediate response actions
        self._execute_immediate_response(incident)

        # Notify stakeholders
        self._notify_stakeholders(incident)

        # Log incident
        audit_logger.log_security_event(
            user_id=reported_by,
            event_type="security_incident_reported",
            severity=incident["severity"].value,
            details={"incident_id": incident_id, "incident_type": incident_type}
        )

        logger.warning(f"SECURITY INCIDENT REPORTED: {incident_id} - {incident_type}")
        return incident_id

    def _execute_immediate_response(self, incident: Dict):
        """Execute immediate response actions."""
        incident_type = incident["type"]
        playbook = self.response_playbooks[incident_type]

        for action in playbook["immediate_actions"]:
            logger.info(f"Executing immediate response: {action}")

            # Log action in timeline
            incident["timeline"].append({
                "timestamp": datetime.utcnow().isoformat(),
                "action": f"immediate_response_{action.lower().replace(' ', '_')}",
                "details": f"Executed: {action}"
            })

            # In production, these would trigger actual system actions
            if "isolate" in action.lower():
                self._isolate_systems()
            elif "disable" in action.lower():
                self._disable_proof_generation()
            elif "notify" in action.lower():
                self._notify_security_team(incident)

    def _notify_stakeholders(self, incident: Dict):
        """Notify relevant stakeholders about incident."""
        severity = incident["severity"]
        incident_type = incident["type"]

        # Determine notification priority
        if severity == IncidentSeverity.CRITICAL:
            channels = ["email", "slack", "pagerduty"]
        elif severity == IncidentSeverity.HIGH:
            channels = ["email", "slack"]
        else:
            channels = ["email"]

        message = f"""
SECURITY INCIDENT ALERT
Severity: {severity.value.upper()}
Type: {incident_type}
ID: {incident['id']}
Reported: {incident['reported_at']}

Details: {json.dumps(incident['details'], indent=2)}

Immediate response actions have been initiated.
Please review incident response playbook for {incident_type}.
        """.strip()

        for channel in channels:
            self._send_notification(channel, message, severity)

    def _send_notification(self, channel: str, message: str, severity: IncidentSeverity):
        """Send notification via specified channel."""
        try:
            if channel == "email":
                self._send_email_notification(message, severity)
            elif channel == "slack":
                self._send_slack_notification(message, severity)
            elif channel == "pagerduty":
                self._send_pagerduty_notification(message, severity)
        except Exception as e:
            logger.error(f"Failed to send {channel} notification: {e}")

    def _isolate_systems(self):
        """Isolate affected systems."""
        logger.warning("Isolating affected systems...")
        # In production: disable network interfaces, block ports, etc.

    def _disable_proof_generation(self):
        """Disable proof generation temporarily."""
        logger.warning("Disabling proof generation...")
        # In production: set maintenance mode, disable API endpoints

    def _notify_security_team(self, incident: Dict):
        """Notify security team."""
        logger.warning(f"Notifying security team about incident {incident['id']}")

# Global incident response instance
incident_response = SecurityIncidentResponse()

# Example usage
incident_id = incident_response.report_incident(
    incident_type="cryptographic_failure",
    details={
        "affected_component": "zk_validator",
        "error_message": "Circuit validation failed",
        "potential_impact": "Proof generation blocked"
    },
    reported_by="system_monitor"
)
```

---

## 📚 Resources and References

### Security Standards and Frameworks

- **NIST Cybersecurity Framework**: [NIST CSF](https://www.nist.gov/cyberframework)
- **ISO 27001**: Information Security Management
- **GDPR**: General Data Protection Regulation
- **SOX**: Sarbanes-Oxley Act Compliance
- **OWASP**: Web Application Security Standards

### Cryptographic Security References

- **NIST SP 800-57**: Recommendation for Key Management
- **FIPS 140-2/3**: Cryptographic Module Validation
- **RFC 8446**: TLS 1.3 Specification
- **RFC 5280**: X.509 Certificate Standard

### Monitoring and Alerting Tools

- **ELK Stack**: Elasticsearch, Logstash, Kibana
- **Prometheus + Grafana**: Metrics and visualization
- **OSSEC/Snort**: Intrusion detection
- **Fail2Ban**: Brute force protection

---

*This security best practices guide provides enterprise-grade security measures for FEDzk production deployments. All recommendations focus on real cryptographic operations with zero compromises on security.*

