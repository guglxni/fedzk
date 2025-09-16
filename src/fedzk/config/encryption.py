#!/usr/bin/env python3
"""
Configuration Encryption System
==============================

Secure encryption and decryption of sensitive configuration values.
Supports multiple encryption backends and secure key management.
"""

import os
import base64
import secrets
import hashlib
from pathlib import Path
from typing import Optional, Dict, Any
from cryptography.fernet import Fernet, InvalidToken
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.backends import default_backend
import logging


class ConfigEncryptionError(Exception):
    """Configuration encryption error."""
    pass


class ConfigEncryption:
    """Advanced configuration encryption with multiple backends."""

    def __init__(self, master_key: Optional[str] = None, salt: Optional[bytes] = None):
        """Initialize encryption with master key and salt."""
        self.logger = logging.getLogger(__name__)

        # Generate or use provided master key
        if master_key is None:
            master_key = self._generate_master_key()

        self.master_key = master_key

        # Generate or use provided salt
        if salt is None:
            salt = self._generate_salt()

        self.salt = salt

        # Create Fernet cipher for symmetric encryption
        self.fernet = self._create_fernet_cipher()

        # Track encrypted fields for audit
        self.encrypted_fields = set()

    def _generate_master_key(self) -> str:
        """Generate a secure master key."""
        # Try to get from environment first
        env_key = os.getenv('FEDZK_ENCRYPTION_MASTER_KEY')
        if env_key and len(env_key) >= 32:
            return env_key

        # Generate a new key (for development/demo purposes)
        self.logger.warning("No master key provided, generating temporary key for development")
        return base64.urlsafe_b64encode(secrets.token_bytes(32)).decode()

    def _generate_salt(self) -> bytes:
        """Generate a secure salt for key derivation."""
        # Use a consistent salt for the same master key
        salt_input = self.master_key + "fedzk_config_salt"
        return hashlib.sha256(salt_input.encode()).digest()[:16]

    def _create_fernet_cipher(self) -> Fernet:
        """Create Fernet cipher from master key."""
        # Derive encryption key using PBKDF2
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=self.salt,
            iterations=100000,
            backend=default_backend()
        )

        key = base64.urlsafe_b64encode(kdf.derive(self.master_key.encode()))
        return Fernet(key)

    def encrypt_value(self, value: str, field_name: Optional[str] = None) -> str:
        """Encrypt a configuration value."""
        if not isinstance(value, str):
            value = str(value)

        if not value:
            raise ConfigEncryptionError("Cannot encrypt empty value")

        try:
            encrypted_bytes = self.fernet.encrypt(value.encode())
            encrypted_value = base64.urlsafe_b64encode(encrypted_bytes).decode()

            if field_name:
                self.encrypted_fields.add(field_name)

            return encrypted_value

        except Exception as e:
            raise ConfigEncryptionError(f"Failed to encrypt value: {e}")

    def decrypt_value(self, encrypted_value: str, field_name: Optional[str] = None) -> str:
        """Decrypt a configuration value."""
        if not encrypted_value:
            raise ConfigEncryptionError("Cannot decrypt empty value")

        try:
            encrypted_bytes = base64.urlsafe_b64decode(encrypted_value.encode())
            decrypted_bytes = self.fernet.decrypt(encrypted_bytes)
            decrypted_value = decrypted_bytes.decode()

            if field_name:
                self.encrypted_fields.add(field_name)

            return decrypted_value

        except InvalidToken:
            raise ConfigEncryptionError("Invalid encrypted value or incorrect key")
        except Exception as e:
            raise ConfigEncryptionError(f"Failed to decrypt value: {e}")

    def is_encrypted(self, value: str) -> bool:
        """Check if a value appears to be encrypted."""
        if not value:
            return False

        try:
            # Try to decode as base64
            base64.urlsafe_b64decode(value.encode())
            return True
        except Exception:
            return False

    def rotate_key(self, new_master_key: str) -> Dict[str, str]:
        """Rotate the encryption key and re-encrypt all values."""
        if not new_master_key or len(new_master_key) < 32:
            raise ConfigEncryptionError("New master key must be at least 32 characters")

        old_fernet = self.fernet

        # Update to new key
        self.master_key = new_master_key
        self.fernet = self._create_fernet_cipher()

        # Return mapping for re-encryption
        return {
            'old_master_key_hash': hashlib.sha256(self.master_key.encode()).hexdigest(),
            'new_master_key_hash': hashlib.sha256(new_master_key.encode()).hexdigest(),
            'rotation_timestamp': str(hashlib.sha256(str(os.time.time()).encode()).hexdigest())
        }

    def get_encryption_info(self) -> Dict[str, Any]:
        """Get information about the encryption system."""
        return {
            'master_key_hash': hashlib.sha256(self.master_key.encode()).hexdigest(),
            'salt_hash': hashlib.sha256(self.salt).hexdigest(),
            'encrypted_fields_count': len(self.encrypted_fields),
            'encrypted_fields': list(self.encrypted_fields),
            'cipher_type': 'Fernet (AES-128-CBC)',
            'key_derivation': 'PBKDF2-SHA256',
            'iterations': 100000
        }


class SecureConfigStorage:
    """Secure configuration storage with encryption."""

    def __init__(self, storage_path: Optional[Path] = None, encryption: Optional[ConfigEncryption] = None):
        """Initialize secure configuration storage."""
        self.storage_path = storage_path or Path("./config/secrets.yaml")
        self.encryption = encryption or ConfigEncryption()
        self.logger = logging.getLogger(__name__)

    def store_secret(self, key: str, value: str, description: Optional[str] = None):
        """Store an encrypted secret."""
        encrypted_value = self.encryption.encrypt_value(value, key)

        secret_data = {
            'key': key,
            'encrypted_value': encrypted_value,
            'description': description or '',
            'created_at': str(hashlib.sha256(str(os.time.time()).encode()).hexdigest()),
            'version': '1.0'
        }

        # Load existing secrets
        secrets = self._load_secrets()

        # Update or add secret
        secrets[key] = secret_data

        # Save secrets
        self._save_secrets(secrets)

        self.logger.info(f"Stored encrypted secret: {key}")

    def retrieve_secret(self, key: str) -> Optional[str]:
        """Retrieve and decrypt a secret."""
        secrets = self._load_secrets()

        if key not in secrets:
            return None

        encrypted_value = secrets[key]['encrypted_value']
        try:
            return self.encryption.decrypt_value(encrypted_value, key)
        except ConfigEncryptionError:
            self.logger.error(f"Failed to decrypt secret: {key}")
            return None

    def list_secrets(self) -> Dict[str, Dict[str, Any]]:
        """List all stored secrets (without decrypted values)."""
        secrets = self._load_secrets()

        # Return metadata only (no encrypted values)
        public_info = {}
        for key, data in secrets.items():
            public_info[key] = {
                'description': data.get('description', ''),
                'created_at': data.get('created_at', ''),
                'version': data.get('version', '1.0'),
                'encrypted': True
            }

        return public_info

    def delete_secret(self, key: str) -> bool:
        """Delete a stored secret."""
        secrets = self._load_secrets()

        if key in secrets:
            del secrets[key]
            self._save_secrets(secrets)
            self.logger.info(f"Deleted secret: {key}")
            return True

        return False

    def _load_secrets(self) -> Dict[str, Dict[str, Any]]:
        """Load secrets from storage."""
        if not self.storage_path.exists():
            return {}

        try:
            import yaml
            with open(self.storage_path, 'r') as f:
                return yaml.safe_load(f) or {}
        except Exception as e:
            self.logger.error(f"Failed to load secrets: {e}")
            return {}

    def _save_secrets(self, secrets: Dict[str, Dict[str, Any]]):
        """Save secrets to storage."""
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            import yaml
            with open(self.storage_path, 'w') as f:
                yaml.dump(secrets, f, default_flow_style=False)
        except Exception as e:
            self.logger.error(f"Failed to save secrets: {e}")
            raise ConfigEncryptionError(f"Failed to save secrets: {e}")


class ConfigEncryptionManager:
    """High-level configuration encryption manager."""

    def __init__(self):
        self.encryption = ConfigEncryption()
        self.storage = SecureConfigStorage(encryption=self.encryption)
        self.logger = logging.getLogger(__name__)

    def encrypt_sensitive_config(self, config_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Encrypt sensitive fields in configuration."""
        sensitive_fields = [
            'postgresql_password', 'redis_password', 'jwt_secret_key',
            'api_key', 'secret_key', 'private_key', 'token',
            'database_url', 'connection_string'
        ]

        encrypted_config = config_dict.copy()

        for field in sensitive_fields:
            if field in encrypted_config and encrypted_config[field]:
                value = encrypted_config[field]
                if isinstance(value, str) and not self.encryption.is_encrypted(value):
                    encrypted_value = self.encryption.encrypt_value(value, field)
                    encrypted_config[field] = encrypted_value
                    encrypted_config[f"{field}_encrypted"] = True

        return encrypted_config

    def decrypt_sensitive_config(self, config_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Decrypt sensitive fields in configuration."""
        decrypted_config = config_dict.copy()

        for key, value in config_dict.items():
            if key.endswith('_encrypted') and isinstance(value, bool) and value:
                base_key = key[:-10]  # Remove '_encrypted' suffix
                if base_key in decrypted_config:
                    encrypted_value = decrypted_config[base_key]
                    if isinstance(encrypted_value, str):
                        try:
                            decrypted_value = self.encryption.decrypt_value(encrypted_value, base_key)
                            decrypted_config[base_key] = decrypted_value
                        except ConfigEncryptionError as e:
                            self.logger.error(f"Failed to decrypt {base_key}: {e}")

        return decrypted_config

    def setup_encryption_keys(self):
        """Setup encryption keys for the application."""
        print("🔐 Configuration Encryption Setup")
        print("=" * 40)

        # Check if master key exists
        master_key = os.getenv('FEDZK_ENCRYPTION_MASTER_KEY')
        if not master_key:
            print("⚠️  No encryption master key found in environment")
            print("   Generating temporary key for development...")
            print("   Set FEDZK_ENCRYPTION_MASTER_KEY environment variable for production")
            print()

        # Display encryption info
        info = self.encryption.get_encryption_info()
        print("Encryption Configuration:")
        print(f"  • Cipher: {info['cipher_type']}")
        print(f"  • Key Derivation: {info['key_derivation']}")
        print(f"  • Iterations: {info['iterations']:,}")
        print(f"  • Master Key Hash: {info['master_key_hash'][:16]}...")
        print()

    def get_security_status(self) -> Dict[str, Any]:
        """Get security status of the encryption system."""
        return {
            'encryption_enabled': True,
            'master_key_configured': bool(os.getenv('FEDZK_ENCRYPTION_MASTER_KEY')),
            'secrets_stored': len(self.storage.list_secrets()),
            'encryption_info': self.encryption.get_encryption_info()
        }


# Utility functions
def generate_secure_key(length: int = 32) -> str:
    """Generate a secure random key."""
    return secrets.token_urlsafe(length)


def hash_sensitive_value(value: str) -> str:
    """Hash a sensitive value for logging/comparison."""
    return hashlib.sha256(value.encode()).hexdigest()[:16]


def demo_encryption():
    """Demonstrate configuration encryption functionality."""
    print("🔐 Configuration Encryption Demonstration")
    print("=" * 50)

    # Create encryption system
    encryption = ConfigEncryption()

    # Sample sensitive configuration
    sensitive_config = {
        'database_password': 'my_secret_db_password_123',
        'api_key': 'sk-1234567890abcdef',
        'jwt_secret': 'super-secret-jwt-key-for-production'
    }

    print("Original Configuration:")
    for key, value in sensitive_config.items():
        print(f"  {key}: {value}")
    print()

    # Encrypt sensitive values
    encrypted_config = {}
    for key, value in sensitive_config.items():
        encrypted_value = encryption.encrypt_value(value, key)
        encrypted_config[key] = encrypted_value
        print(f"Encrypted {key}: {encrypted_value}")

    print()

    # Decrypt values
    print("Decryption Test:")
    for key, encrypted_value in encrypted_config.items():
        try:
            decrypted_value = encryption.decrypt_value(encrypted_value, key)
            original_value = sensitive_config[key]
            success = "✅" if decrypted_value == original_value else "❌"
            print(f"  {key}: {success} {decrypted_value}")
        except Exception as e:
            print(f"  {key}: ❌ Failed to decrypt: {e}")

    print()

    # Secure storage demonstration
    storage = SecureConfigStorage(Path('./demo_secrets.yaml'), encryption)

    print("Secure Storage Demonstration:")
    storage.store_secret('demo_api_key', 'demo-key-12345', 'Demo API key for testing')
    storage.store_secret('demo_db_password', 'demo-password', 'Demo database password')

    retrieved_key = storage.retrieve_secret('demo_api_key')
    print(f"Retrieved API key: {'✅' if retrieved_key == 'demo-key-12345' else '❌'}")

    print("\nStored Secrets:")
    secrets_list = storage.list_secrets()
    for key, info in secrets_list.items():
        print(f"  • {key}: {info['description']}")

    print("\n✅ Configuration encryption demonstration completed")


if __name__ == "__main__":
    demo_encryption()
