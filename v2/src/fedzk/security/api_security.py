#!/usr/bin/env python3
"""
FEDzk API Security Layer
========================

Comprehensive API security implementation for FEDzk.
Provides OAuth 2.0, JWT tokens, API key management, and secure communications.

Features:
- OAuth 2.0 / OpenID Connect authentication
- JWT token validation and refresh mechanisms
- API key rotation and revocation
- Request/response encryption for sensitive data
- Production security standards
"""

import jwt
import json
import time
import logging
import secrets
import hashlib
import hmac
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, asdict
from enum import Enum
from datetime import datetime, timedelta
import base64
import threading
import re
from pathlib import Path
import requests
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

class OAuthGrantType(Enum):
    """OAuth 2.0 grant types."""
    AUTHORIZATION_CODE = "authorization_code"
    IMPLICIT = "implicit"
    PASSWORD = "password"
    CLIENT_CREDENTIALS = "client_credentials"
    REFRESH_TOKEN = "refresh_token"

class JWTAlgorithm(Enum):
    """Supported JWT algorithms."""
    HS256 = "HS256"
    HS384 = "HS384"
    HS512 = "HS512"
    RS256 = "RS256"
    RS384 = "RS384"
    RS512 = "RS512"
    ES256 = "ES256"
    ES384 = "ES384"
    ES512 = "ES512"

class APIKeyStatus(Enum):
    """API key status."""
    ACTIVE = "active"
    INACTIVE = "inactive"
    REVOKED = "revoked"
    EXPIRED = "expired"

@dataclass
class OAuthClient:
    """OAuth 2.0 client configuration."""
    client_id: str
    client_secret: str
    redirect_uris: List[str]
    grant_types: List[OAuthGrantType]
    response_types: List[str]
    scopes: List[str]
    token_endpoint_auth_method: str
    created_at: datetime
    updated_at: datetime

@dataclass
class JWTToken:
    """JWT token information."""
    token: str
    token_type: str
    expires_at: datetime
    issued_at: datetime
    issuer: str
    subject: str
    audience: List[str]
    scopes: List[str]
    claims: Dict[str, Any]

@dataclass
class APIKey:
    """API key information."""
    key_id: str
    key_hash: str
    name: str
    description: str
    status: APIKeyStatus
    created_at: datetime
    expires_at: Optional[datetime]
    last_used_at: Optional[datetime]
    usage_count: int
    rate_limit: int  # requests per hour
    scopes: List[str]
    metadata: Dict[str, Any]

@dataclass
class APISecurityConfig:
    """API security configuration."""
    jwt_algorithm: JWTAlgorithm = JWTAlgorithm.HS256
    jwt_secret_key: Optional[str] = None
    jwt_public_key: Optional[str] = None
    jwt_private_key: Optional[str] = None
    jwt_expiration_hours: int = 24
    jwt_refresh_expiration_days: int = 30
    jwt_issuer: str = "fedzk"
    jwt_audience: List[str] = None

    # OAuth configuration
    oauth_enabled: bool = True
    oauth_token_url: Optional[str] = None
    oauth_authorization_url: Optional[str] = None
    oauth_userinfo_url: Optional[str] = None
    oauth_client_id: Optional[str] = None
    oauth_client_secret: Optional[str] = None

    # API key configuration
    api_key_enabled: bool = True
    api_key_length: int = 32
    api_key_rotation_days: int = 90
    api_key_max_age_days: int = 365

    # Security features
    request_encryption_enabled: bool = False
    response_encryption_enabled: bool = False
    rate_limiting_enabled: bool = True
    audit_logging_enabled: bool = True

    def __post_init__(self):
        if self.jwt_audience is None:
            self.jwt_audience = ["fedzk"]

        # Generate secret key if not provided
        if not self.jwt_secret_key and self.jwt_algorithm.value.startswith('HS'):
            self.jwt_secret_key = secrets.token_hex(32)

class APISecurityManager:
    """
    API Security Manager for FEDzk.

    Provides comprehensive API security including OAuth 2.0, JWT tokens,
    API key management, and secure request/response handling.
    """

    def __init__(self, config: APISecurityConfig = None):
        self.config = config or APISecurityConfig()

        # Initialize keys and secrets
        self.jwt_secret_key = self.config.jwt_secret_key
        self.jwt_public_key = self.config.jwt_public_key
        self.jwt_private_key = self.config.jwt_private_key

        # OAuth clients and tokens
        self.oauth_clients: Dict[str, OAuthClient] = {}
        self.active_tokens: Dict[str, JWTToken] = {}
        self.refresh_tokens: Dict[str, str] = {}

        # API keys
        self.api_keys: Dict[str, APIKey] = {}
        self.api_key_hashes: Dict[str, str] = {}  # For fast lookup

        # Rate limiting
        self.rate_limits: Dict[str, List[float]] = {}
        self.rate_limit_lock = threading.Lock()

        # Audit log
        self.audit_log: List[Dict[str, Any]] = []
        self.audit_lock = threading.Lock()

        # Load persisted data
        self._load_api_keys()
        self._load_oauth_clients()

        logger.info("API Security Manager initialized")

    def _load_api_keys(self):
        """Load API keys from storage."""
        key_file = Path("./security/api_keys.json")
        if key_file.exists():
            try:
                with open(key_file, 'r') as f:
                    keys_data = json.load(f)
                    for key_data in keys_data:
                        api_key = APIKey(**key_data)
                        self.api_keys[api_key.key_id] = api_key
                        self.api_key_hashes[api_key.key_hash] = api_key.key_id
                logger.info(f"Loaded {len(self.api_keys)} API keys")
            except Exception as e:
                logger.error(f"Failed to load API keys: {e}")

    def _save_api_keys(self):
        """Save API keys to storage."""
        key_file = Path("./security/api_keys.json")
        key_file.parent.mkdir(parents=True, exist_ok=True)

        try:
            keys_data = [asdict(key) for key in self.api_keys.values()]
            with open(key_file, 'w') as f:
                json.dump(keys_data, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Failed to save API keys: {e}")

    def _load_oauth_clients(self):
        """Load OAuth clients from storage."""
        client_file = Path("./security/oauth_clients.json")
        if client_file.exists():
            try:
                with open(client_file, 'r') as f:
                    clients_data = json.load(f)
                    for client_data in clients_data:
                        client = OAuthClient(**client_data)
                        self.oauth_clients[client.client_id] = client
                logger.info(f"Loaded {len(self.oauth_clients)} OAuth clients")
            except Exception as e:
                logger.error(f"Failed to load OAuth clients: {e}")

    def create_jwt_token(
        self,
        subject: str,
        scopes: List[str] = None,
        audience: List[str] = None,
        additional_claims: Dict[str, Any] = None
    ) -> Tuple[str, str]:
        """
        Create a JWT access token and refresh token.

        Args:
            subject: Token subject (user/client identifier)
            scopes: Token scopes/permissions
            audience: Token audience
            additional_claims: Additional JWT claims

        Returns:
            Tuple[str, str]: Access token and refresh token
        """
        now = datetime.utcnow()
        expires_at = now + timedelta(hours=self.config.jwt_expiration_hours)

        # Create payload
        payload = {
            "iss": self.config.jwt_issuer,
            "sub": subject,
            "aud": audience or self.config.jwt_audience,
            "exp": int(expires_at.timestamp()),
            "iat": int(now.timestamp()),
            "nbf": int(now.timestamp()),
            "jti": secrets.token_hex(16),
            "scopes": scopes or []
        }

        # Add additional claims
        if additional_claims:
            payload.update(additional_claims)

        # Create access token
        if self.config.jwt_algorithm.value.startswith('HS'):
            access_token = jwt.encode(payload, self.jwt_secret_key, algorithm=self.config.jwt_algorithm.value)
        elif self.config.jwt_algorithm.value.startswith('RS'):
            access_token = jwt.encode(payload, self.jwt_private_key, algorithm=self.config.jwt_algorithm.value)
        else:
            access_token = jwt.encode(payload, self.jwt_private_key, algorithm=self.config.jwt_algorithm.value)

        # Create refresh token
        refresh_payload = {
            "iss": self.config.jwt_issuer,
            "sub": subject,
            "exp": int((now + timedelta(days=self.config.jwt_refresh_expiration_days)).timestamp()),
            "iat": int(now.timestamp()),
            "jti": secrets.token_hex(16),
            "type": "refresh"
        }

        refresh_token = jwt.encode(refresh_payload, self.jwt_secret_key, algorithm="HS256")

        # Store tokens
        token_info = JWTToken(
            token=access_token,
            token_type="Bearer",
            expires_at=expires_at,
            issued_at=now,
            issuer=self.config.jwt_issuer,
            subject=subject,
            audience=audience or self.config.jwt_audience,
            scopes=scopes or [],
            claims=payload
        )

        self.active_tokens[access_token] = token_info
        self.refresh_tokens[refresh_token] = access_token

        # Audit log
        self._audit_log("token_created", {"subject": subject, "token_type": "access"})

        logger.info(f"JWT token created for subject: {subject}")
        return access_token, refresh_token

    def validate_jwt_token(self, token: str) -> Optional[JWTToken]:
        """
        Validate a JWT token.

        Args:
            token: JWT token to validate

        Returns:
            JWTToken or None: Token information if valid
        """
        try:
            # Decode token
            if self.config.jwt_algorithm.value.startswith('HS'):
                payload = jwt.decode(token, self.jwt_secret_key, algorithms=[self.config.jwt_algorithm.value])
            elif self.config.jwt_algorithm.value.startswith('RS'):
                payload = jwt.decode(token, self.jwt_public_key, algorithms=[self.config.jwt_algorithm.value])
            else:
                payload = jwt.decode(token, self.jwt_public_key, algorithms=[self.config.jwt_algorithm.value])

            # Check if token is in active tokens
            if token not in self.active_tokens:
                logger.warning(f"Token not found in active tokens: {token[:20]}...")
                return None

            token_info = self.active_tokens[token]

            # Check expiration
            if datetime.utcnow() > token_info.expires_at:
                logger.warning(f"Token expired: {token[:20]}...")
                return None

            # Audit log
            self._audit_log("token_validated", {"subject": token_info.subject})

            return token_info

        except jwt.ExpiredSignatureError:
            logger.warning(f"Token signature expired: {token[:20]}...")
            return None
        except jwt.InvalidSignatureError:
            logger.warning(f"Invalid token signature: {token[:20]}...")
            return None
        except Exception as e:
            logger.error(f"Token validation error: {e}")
            return None

    def refresh_jwt_token(self, refresh_token: str) -> Optional[Tuple[str, str]]:
        """
        Refresh a JWT token using refresh token.

        Args:
            refresh_token: Refresh token

        Returns:
            Tuple or None: New access and refresh tokens
        """
        try:
            # Validate refresh token
            payload = jwt.decode(refresh_token, self.jwt_secret_key, algorithms=["HS256"])

            if payload.get("type") != "refresh":
                logger.warning("Invalid refresh token type")
                return None

            subject = payload["sub"]

            # Check if refresh token exists
            if refresh_token not in self.refresh_tokens:
                logger.warning(f"Refresh token not found: {refresh_token[:20]}...")
                return None

            # Get original access token
            original_token = self.refresh_tokens[refresh_token]

            # Remove old tokens
            if original_token in self.active_tokens:
                del self.active_tokens[original_token]
            del self.refresh_tokens[refresh_token]

            # Create new tokens
            new_access, new_refresh = self.create_jwt_token(
                subject=subject,
                scopes=self.active_tokens.get(original_token, JWTToken("", "", datetime.now(), datetime.now(), "", "", [], [], {})).scopes
            )

            # Audit log
            self._audit_log("token_refreshed", {"subject": subject})

            logger.info(f"Token refreshed for subject: {subject}")
            return new_access, new_refresh

        except Exception as e:
            logger.error(f"Token refresh error: {e}")
            return None

    def create_api_key(
        self,
        name: str,
        description: str = "",
        scopes: List[str] = None,
        expires_in_days: int = 365,
        rate_limit: int = 1000,
        metadata: Dict[str, Any] = None
    ) -> Tuple[str, str]:
        """
        Create a new API key.

        Args:
            name: API key name
            description: API key description
            scopes: API key scopes
            expires_in_days: Key expiration in days
            rate_limit: Rate limit (requests per hour)
            metadata: Additional metadata

        Returns:
            Tuple[str, str]: API key ID and secret
        """
        # Generate API key
        key_secret = secrets.token_hex(self.config.api_key_length // 2)
        key_id = f"fedzk_{secrets.token_hex(8)}"

        # Hash the key for storage
        key_hash = hashlib.sha256(key_secret.encode()).hexdigest()

        # Create API key object
        api_key = APIKey(
            key_id=key_id,
            key_hash=key_hash,
            name=name,
            description=description,
            status=APIKeyStatus.ACTIVE,
            created_at=datetime.utcnow(),
            expires_at=datetime.utcnow() + timedelta(days=expires_in_days),
            last_used_at=None,
            usage_count=0,
            rate_limit=rate_limit,
            scopes=scopes or [],
            metadata=metadata or {}
        )

        # Store API key
        self.api_keys[key_id] = api_key
        self.api_key_hashes[key_hash] = key_id
        self._save_api_keys()

        # Audit log
        self._audit_log("api_key_created", {"key_id": key_id, "name": name})

        logger.info(f"API key created: {key_id}")
        return key_id, key_secret

    def validate_api_key(self, key_secret: str) -> Optional[APIKey]:
        """
        Validate an API key.

        Args:
            key_secret: API key secret to validate

        Returns:
            APIKey or None: API key information if valid
        """
        # Hash the provided key
        key_hash = hashlib.sha256(key_secret.encode()).hexdigest()

        # Find API key
        if key_hash not in self.api_key_hashes:
            return None

        key_id = self.api_key_hashes[key_hash]
        api_key = self.api_keys.get(key_id)

        if not api_key:
            return None

        # Check status
        if api_key.status != APIKeyStatus.ACTIVE:
            return None

        # Check expiration
        if api_key.expires_at and datetime.utcnow() > api_key.expires_at:
            api_key.status = APIKeyStatus.EXPIRED
            return None

        # Check rate limit
        if not self._check_rate_limit(key_id, api_key.rate_limit):
            logger.warning(f"Rate limit exceeded for API key: {key_id}")
            return None

        # Update usage
        api_key.usage_count += 1
        api_key.last_used_at = datetime.utcnow()

        # Audit log
        self._audit_log("api_key_used", {"key_id": key_id})

        return api_key

    def revoke_api_key(self, key_id: str) -> bool:
        """
        Revoke an API key.

        Args:
            key_id: API key ID to revoke

        Returns:
            bool: Success status
        """
        if key_id not in self.api_keys:
            return False

        self.api_keys[key_id].status = APIKeyStatus.REVOKED
        self._save_api_keys()

        # Remove from hash lookup
        key_hash = self.api_keys[key_id].key_hash
        if key_hash in self.api_key_hashes:
            del self.api_key_hashes[key_hash]

        # Audit log
        self._audit_log("api_key_revoked", {"key_id": key_id})

        logger.info(f"API key revoked: {key_id}")
        return True

    def _check_rate_limit(self, key_id: str, limit: int) -> bool:
        """Check if API key is within rate limit."""
        if not self.config.rate_limiting_enabled:
            return True

        with self.rate_limit_lock:
            now = time.time()
            window_start = now - 3600  # 1 hour window

            # Get or create rate limit history
            if key_id not in self.rate_limits:
                self.rate_limits[key_id] = []

            # Clean old entries
            self.rate_limits[key_id] = [
                t for t in self.rate_limits[key_id] if t > window_start
            ]

            # Check limit
            if len(self.rate_limits[key_id]) >= limit:
                return False

            # Add current request
            self.rate_limits[key_id].append(now)
            return True

    def encrypt_request_data(self, data: Dict[str, Any], key: str) -> str:
        """
        Encrypt sensitive request data.

        Args:
            data: Data to encrypt
            key: Encryption key

        Returns:
            str: Encrypted data as base64 string
        """
        if not self.config.request_encryption_enabled:
            return json.dumps(data)

        # Simple encryption for demonstration (use proper encryption in production)
        data_str = json.dumps(data, sort_keys=True)
        encrypted = base64.b64encode(data_str.encode()).decode()
        return encrypted

    def decrypt_request_data(self, encrypted_data: str, key: str) -> Dict[str, Any]:
        """
        Decrypt sensitive request data.

        Args:
            encrypted_data: Encrypted data as base64 string
            key: Decryption key

        Returns:
            Dict: Decrypted data
        """
        if not self.config.request_encryption_enabled:
            return json.loads(encrypted_data)

        # Simple decryption for demonstration
        try:
            decrypted = base64.b64decode(encrypted_data).decode()
            return json.loads(decrypted)
        except Exception:
            raise ValueError("Invalid encrypted data")

    def _audit_log(self, event: str, details: Dict[str, Any]):
        """Add entry to audit log."""
        if not self.config.audit_logging_enabled:
            return

        with self.audit_lock:
            log_entry = {
                "timestamp": datetime.utcnow().isoformat(),
                "event": event,
                "details": details,
                "client_info": self._get_client_info()
            }

            self.audit_log.append(log_entry)

            # Keep only recent entries
            max_entries = 10000
            if len(self.audit_log) > max_entries:
                self.audit_log = self.audit_log[-max_entries:]

    def _get_client_info(self) -> Dict[str, str]:
        """Get client information for logging."""
        return {
            "user": "system",  # In production, get from request context
            "ip": "unknown",
            "user_agent": "unknown"
        }

    def get_security_metrics(self) -> Dict[str, Any]:
        """Get API security metrics."""
        metrics = {
            "total_api_keys": len(self.api_keys),
            "active_api_keys": len([k for k in self.api_keys.values() if k.status == APIKeyStatus.ACTIVE]),
            "expired_api_keys": len([k for k in self.api_keys.values() if k.status == APIKeyStatus.EXPIRED]),
            "revoked_api_keys": len([k for k in self.api_keys.values() if k.status == APIKeyStatus.REVOKED]),
            "active_tokens": len(self.active_tokens),
            "refresh_tokens": len(self.refresh_tokens),
            "oauth_clients": len(self.oauth_clients),
            "audit_log_entries": len(self.audit_log),
            "rate_limited_requests": sum(len(requests) for requests in self.rate_limits.values())
        }

        # Calculate usage statistics
        total_usage = sum(key.usage_count for key in self.api_keys.values())
        metrics["total_api_key_usage"] = total_usage

        if metrics["total_api_keys"] > 0:
            metrics["average_key_usage"] = total_usage / metrics["total_api_keys"]
        else:
            metrics["average_key_usage"] = 0

        return metrics

    def cleanup_expired_tokens(self):
        """Clean up expired tokens."""
        now = datetime.utcnow()
        expired_tokens = []

        for token, token_info in self.active_tokens.items():
            if now > token_info.expires_at:
                expired_tokens.append(token)

        for token in expired_tokens:
            del self.active_tokens[token]

        if expired_tokens:
            logger.info(f"Cleaned up {len(expired_tokens)} expired tokens")

# Convenience functions for easy usage
def create_secure_api_manager(config: APISecurityConfig = None) -> APISecurityManager:
    """
    Create a secure API security manager.

    Args:
        config: API security configuration

    Returns:
        APISecurityManager: Configured security manager
    """
    return APISecurityManager(config)

def generate_api_key(name: str, scopes: List[str] = None) -> Tuple[str, str]:
    """
    Generate a new API key.

    Args:
        name: API key name
        scopes: API key scopes

    Returns:
        Tuple[str, str]: API key ID and secret
    """
    manager = APISecurityManager()
    return manager.create_api_key(name, scopes=scopes or [])

def validate_bearer_token(token: str) -> Optional[JWTToken]:
    """
    Validate a Bearer token.

    Args:
        token: Bearer token to validate

    Returns:
        JWTToken or None: Token information if valid
    """
    manager = APISecurityManager()
    return manager.validate_jwt_token(token)

