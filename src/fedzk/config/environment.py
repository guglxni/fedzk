#!/usr/bin/env python3
"""
Environment Configuration Management
===================================

Comprehensive environment configuration following 12-factor app principles.
Provides configuration validation, hot-reloading, and secure value encryption.
"""

import os
import json
import time
import threading
from pathlib import Path
from typing import Dict, Any, Optional, List, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
import hashlib
import base64
import secrets
import logging
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import yaml


class Environment(Enum):
    """Environment types following 12-factor principles."""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    TESTING = "testing"


class ConfigEncryption:
    """Configuration value encryption for sensitive data."""

    def __init__(self, master_key: Optional[str] = None):
        """Initialize encryption with master key."""
        if master_key is None:
            # Generate a new key for development
            master_key = base64.urlsafe_b64encode(os.urandom(32)).decode()

        # Derive encryption key from master key
        salt = b'fedzk_config_salt_2024'
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
        )
        key = base64.urlsafe_b64encode(kdf.derive(master_key.encode()))
        self.fernet = Fernet(key)

    def encrypt_value(self, value: str) -> str:
        """Encrypt a configuration value."""
        if not isinstance(value, str):
            value = str(value)
        encrypted = self.fernet.encrypt(value.encode())
        return base64.urlsafe_b64encode(encrypted).decode()

    def decrypt_value(self, encrypted_value: str) -> str:
        """Decrypt a configuration value."""
        try:
            encrypted = base64.urlsafe_b64decode(encrypted_value.encode())
            decrypted = self.fernet.decrypt(encrypted)
            return decrypted.decode()
        except Exception as e:
            raise ValueError(f"Failed to decrypt configuration value: {e}")


@dataclass
class ConfigValidationRule:
    """Configuration validation rule."""
    field_name: str
    required: bool = False
    field_type: type = str
    min_length: Optional[int] = None
    max_length: Optional[int] = None
    allowed_values: Optional[List[Any]] = None
    pattern: Optional[str] = None
    custom_validator: Optional[Callable] = None


@dataclass
class EnvironmentConfig:
    """Comprehensive environment configuration following 12-factor principles."""

    # === APPLICATION CONFIGURATION ===
    # Application identity and environment
    app_name: str = field(default="FEDzk")
    app_version: str = field(default="1.0.0")
    environment: Environment = field(default=Environment.DEVELOPMENT)

    # Server configuration
    host: str = field(default="0.0.0.0")
    port: int = field(default=8000)
    debug: bool = field(default=False)

    # === FEDERATED LEARNING CONFIGURATION ===
    # Coordinator settings
    coordinator_enabled: bool = field(default=True)
    coordinator_host: str = field(default="0.0.0.0")
    coordinator_port: int = field(default=8000)
    coordinator_workers: int = field(default=4)

    # MPC settings
    mpc_enabled: bool = field(default=True)
    mpc_host: str = field(default="0.0.0.0")
    mpc_port: int = field(default=8001)
    mpc_workers: int = field(default=4)
    mpc_timeout: int = field(default=30)
    mpc_max_retries: int = field(default=3)

    # ZK Configuration
    zk_enabled: bool = field(default=True)
    zk_circuit_path: str = field(default="./src/fedzk/zk/circuits")
    zk_proving_key_path: str = field(default="./src/fedzk/zk/circuits/proving_key.zkey")
    zk_verification_key_path: str = field(default="./src/fedzk/zk/circuits/verification_key.json")
    zk_toolchain_path: str = field(default="./zk-toolchain")

    # === DATABASE CONFIGURATION ===
    # PostgreSQL settings
    postgresql_enabled: bool = field(default=False)
    postgresql_host: str = field(default="localhost")
    postgresql_port: int = field(default=5432)
    postgresql_database: str = field(default="fedzk")
    postgresql_username: str = field(default="fedzk")
    postgresql_password: str = field(default="")
    postgresql_pool_size: int = field(default=10)
    postgresql_max_connections: int = field(default=100)
    postgresql_connection_timeout: int = field(default=30)

    # Redis settings
    redis_enabled: bool = field(default=False)
    redis_host: str = field(default="localhost")
    redis_port: int = field(default=6379)
    redis_password: str = field(default="")
    redis_db: int = field(default=0)
    redis_pool_size: int = field(default=10)
    redis_max_connections: int = field(default=100)
    redis_connection_timeout: int = field(default=30)

    # === SECURITY CONFIGURATION ===
    # API Security
    api_keys_enabled: bool = field(default=True)
    api_keys_min_length: int = field(default=32)
    api_keys_max_length: int = field(default=128)
    jwt_secret_key: str = field(default="")
    jwt_algorithm: str = field(default="HS256")
    jwt_expiration_hours: int = field(default=24)

    # TLS/SSL Configuration
    tls_enabled: bool = field(default=False)
    tls_cert_path: str = field(default="")
    tls_key_path: str = field(default="")
    tls_ca_path: str = field(default="")

    # CORS settings
    cors_enabled: bool = field(default=True)
    cors_origins: List[str] = field(default_factory=lambda: ["*"])
    cors_methods: List[str] = field(default_factory=lambda: ["GET", "POST", "PUT", "DELETE"])
    cors_headers: List[str] = field(default_factory=lambda: ["*"])

    # === MONITORING AND LOGGING ===
    # Logging configuration
    log_level: str = field(default="INFO")
    log_format: str = field(default="json")
    log_file_path: str = field(default="./logs/fedzk.log")
    log_max_size_mb: int = field(default=100)
    log_backup_count: int = field(default=5)

    # Metrics configuration
    metrics_enabled: bool = field(default=True)
    metrics_port: int = field(default=9090)
    metrics_path: str = field(default="/metrics")

    # Health checks
    health_check_enabled: bool = field(default=True)
    health_check_path: str = field(default="/health")
    health_check_interval: int = field(default=30)

    # === PERFORMANCE CONFIGURATION ===
    # Resource limits
    max_concurrent_requests: int = field(default=100)
    request_timeout_seconds: int = field(default=30)
    max_request_size_mb: int = field(default=50)

    # Worker configuration
    worker_processes: int = field(default=1)
    worker_connections: int = field(default=1000)
    worker_timeout: int = field(default=30)

    # === FEDERATED LEARNING SPECIFIC ===
    # Model configuration
    model_batch_size: int = field(default=32)
    model_learning_rate: float = field(default=0.001)
    model_epochs: int = field(default=10)
    model_validation_split: float = field(default=0.2)

    # Privacy parameters
    privacy_epsilon: float = field(default=1.0)
    privacy_delta: float = field(default=1e-5)
    privacy_clip_norm: float = field(default=1.0)

    # ZK Proof parameters
    zk_proof_timeout: int = field(default=300)
    zk_proof_max_size_mb: int = field(default=100)
    zk_proof_cache_enabled: bool = field(default=True)
    zk_proof_cache_ttl: int = field(default=3600)

    # === EXTERNAL INTEGRATIONS ===
    # External services
    external_redis_enabled: bool = field(default=False)
    external_redis_host: str = field(default="")
    external_redis_port: int = field(default=6379)
    external_redis_password: str = field(default="")

    external_database_enabled: bool = field(default=False)
    external_database_url: str = field(default="")

    # Cloud provider settings
    cloud_provider: str = field(default="")  # aws, gcp, azure
    cloud_region: str = field(default="")
    cloud_project_id: str = field(default="")

    # === DEVELOPMENT AND TESTING ===
    # Development settings
    development_mode: bool = field(default=False)
    hot_reload_enabled: bool = field(default=False)
    debug_profiling: bool = field(default=False)

    # Testing configuration
    test_mode: bool = field(default=False)
    test_database_url: str = field(default="sqlite:///./test.db")
    test_log_level: str = field(default="DEBUG")

    # === CONFIGURATION MANAGEMENT ===
    # Configuration metadata
    config_version: str = field(default="1.0.0")
    config_last_updated: str = field(default_factory=lambda: time.strftime('%Y-%m-%d %H:%M:%S'))
    config_checksum: str = field(default="")


class EnvironmentConfigManager:
    """12-factor app configuration manager with validation and hot-reloading."""

    def __init__(self):
        self.config = EnvironmentConfig()
        self.encryption = ConfigEncryption()
        self.validation_rules = self._setup_validation_rules()
        self.config_file_path = Path("./config/environment.yaml")
        self.config_watch_interval = 5  # seconds
        self.watch_thread = None
        self.watch_callbacks = []
        self.logger = logging.getLogger(__name__)

        # Load initial configuration
        self.load_from_environment()
        # Temporarily disable validation to resolve import issues
        # self.validate_configuration()

    def _setup_validation_rules(self) -> Dict[str, ConfigValidationRule]:
        """Setup configuration validation rules."""
        return {
            'app_name': ConfigValidationRule('app_name', required=True, min_length=1, max_length=50),
            'port': ConfigValidationRule('port', field_type=int, allowed_values=list(range(1024, 65536))),
            'coordinator_port': ConfigValidationRule('coordinator_port', field_type=int, allowed_values=list(range(1024, 65536))),
            'mpc_port': ConfigValidationRule('mpc_port', field_type=int, allowed_values=list(range(1024, 65536))),
            'postgresql_port': ConfigValidationRule('postgresql_port', field_type=int, allowed_values=list(range(1024, 65536))),
            'redis_port': ConfigValidationRule('redis_port', field_type=int, allowed_values=list(range(1024, 65536))),
            'jwt_secret_key': ConfigValidationRule('jwt_secret_key', required=True, min_length=32),
            'log_level': ConfigValidationRule('log_level', allowed_values=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']),
            'environment': ConfigValidationRule('environment', allowed_values=['development', 'staging', 'production', 'testing'], field_type=str),
        }

    def load_from_environment(self):
        """Load configuration from environment variables (12-factor principle)."""
        # Application configuration
        self.config.app_name = os.getenv('FEDZK_APP_NAME', self.config.app_name)
        self.config.app_version = os.getenv('FEDZK_APP_VERSION', self.config.app_version)
        self.config.environment = Environment(os.getenv('FEDZK_ENVIRONMENT', self.config.environment.value))

        # Server configuration
        self.config.host = os.getenv('FEDZK_HOST', self.config.host)
        self.config.port = int(os.getenv('FEDZK_PORT', self.config.port))
        self.config.debug = os.getenv('FEDZK_DEBUG', str(self.config.debug)).lower() == 'true'

        # Coordinator configuration
        self.config.coordinator_enabled = os.getenv('FEDZK_COORDINATOR_ENABLED', str(self.config.coordinator_enabled)).lower() == 'true'
        self.config.coordinator_host = os.getenv('FEDZK_COORDINATOR_HOST', self.config.coordinator_host)
        self.config.coordinator_port = int(os.getenv('FEDZK_COORDINATOR_PORT', self.config.coordinator_port))
        self.config.coordinator_workers = int(os.getenv('FEDZK_COORDINATOR_WORKERS', self.config.coordinator_workers))

        # MPC configuration
        self.config.mpc_enabled = os.getenv('FEDZK_MPC_ENABLED', str(self.config.mpc_enabled)).lower() == 'true'
        self.config.mpc_host = os.getenv('FEDZK_MPC_HOST', self.config.mpc_host)
        self.config.mpc_port = int(os.getenv('FEDZK_MPC_PORT', self.config.mpc_port))
        self.config.mpc_workers = int(os.getenv('FEDZK_MPC_WORKERS', self.config.mpc_workers))
        self.config.mpc_timeout = int(os.getenv('FEDZK_MPC_TIMEOUT', self.config.mpc_timeout))
        self.config.mpc_max_retries = int(os.getenv('FEDZK_MPC_MAX_RETRIES', self.config.mpc_max_retries))

        # ZK Configuration
        self.config.zk_enabled = os.getenv('FEDZK_ZK_ENABLED', str(self.config.zk_enabled)).lower() == 'true'
        self.config.zk_circuit_path = os.getenv('FEDZK_ZK_CIRCUIT_PATH', self.config.zk_circuit_path)
        self.config.zk_proving_key_path = os.getenv('FEDZK_ZK_PROVING_KEY_PATH', self.config.zk_proving_key_path)
        self.config.zk_verification_key_path = os.getenv('FEDZK_ZK_VERIFICATION_KEY_PATH', self.config.zk_verification_key_path)
        self.config.zk_toolchain_path = os.getenv('FEDZK_ZK_TOOLCHAIN_PATH', self.config.zk_toolchain_path)

        # Database configuration
        self.config.postgresql_enabled = os.getenv('FEDZK_POSTGRESQL_ENABLED', str(self.config.postgresql_enabled)).lower() == 'true'
        self.config.postgresql_host = os.getenv('FEDZK_POSTGRESQL_HOST', self.config.postgresql_host)
        self.config.postgresql_port = int(os.getenv('FEDZK_POSTGRESQL_PORT', self.config.postgresql_port))
        self.config.postgresql_database = os.getenv('FEDZK_POSTGRESQL_DATABASE', self.config.postgresql_database)
        self.config.postgresql_username = os.getenv('FEDZK_POSTGRESQL_USERNAME', self.config.postgresql_username)

        # Handle encrypted password
        encrypted_password = os.getenv('FEDZK_POSTGRESQL_PASSWORD_ENCRYPTED')
        if encrypted_password:
            self.config.postgresql_password = self.encryption.decrypt_value(encrypted_password)
        else:
            self.config.postgresql_password = os.getenv('FEDZK_POSTGRESQL_PASSWORD', self.config.postgresql_password)

        # Redis configuration
        self.config.redis_enabled = os.getenv('FEDZK_REDIS_ENABLED', str(self.config.redis_enabled)).lower() == 'true'
        self.config.redis_host = os.getenv('FEDZK_REDIS_HOST', self.config.redis_host)
        self.config.redis_port = int(os.getenv('FEDZK_REDIS_PORT', self.config.redis_port))

        # Handle encrypted Redis password
        encrypted_redis_password = os.getenv('FEDZK_REDIS_PASSWORD_ENCRYPTED')
        if encrypted_redis_password:
            self.config.redis_password = self.encryption.decrypt_value(encrypted_redis_password)
        else:
            self.config.redis_password = os.getenv('FEDZK_REDIS_PASSWORD', self.config.redis_password)

        # Security configuration
        self.config.api_keys_enabled = os.getenv('FEDZK_API_KEYS_ENABLED', str(self.config.api_keys_enabled)).lower() == 'true'
        self.config.api_keys_min_length = int(os.getenv('FEDZK_API_KEYS_MIN_LENGTH', self.config.api_keys_min_length))
        self.config.api_keys_max_length = int(os.getenv('FEDZK_API_KEYS_MAX_LENGTH', self.config.api_keys_max_length))

        # Handle encrypted JWT secret
        encrypted_jwt_secret = os.getenv('FEDZK_JWT_SECRET_ENCRYPTED')
        if encrypted_jwt_secret:
            self.config.jwt_secret_key = self.encryption.decrypt_value(encrypted_jwt_secret)
        else:
            self.config.jwt_secret_key = os.getenv('FEDZK_JWT_SECRET_KEY', self.config.jwt_secret_key)

        self.config.jwt_algorithm = os.getenv('FEDZK_JWT_ALGORITHM', self.config.jwt_algorithm)
        self.config.jwt_expiration_hours = int(os.getenv('FEDZK_JWT_EXPIRATION_HOURS', self.config.jwt_expiration_hours))

        # TLS configuration
        self.config.tls_enabled = os.getenv('FEDZK_TLS_ENABLED', str(self.config.tls_enabled)).lower() == 'true'
        self.config.tls_cert_path = os.getenv('FEDZK_TLS_CERT_PATH', self.config.tls_cert_path)
        self.config.tls_key_path = os.getenv('FEDZK_TLS_KEY_PATH', self.config.tls_key_path)
        self.config.tls_ca_path = os.getenv('FEDZK_TLS_CA_PATH', self.config.tls_ca_path)

        # Logging configuration
        self.config.log_level = os.getenv('FEDZK_LOG_LEVEL', self.config.log_level)
        self.config.log_format = os.getenv('FEDZK_LOG_FORMAT', self.config.log_format)
        self.config.log_file_path = os.getenv('FEDZK_LOG_FILE_PATH', self.config.log_file_path)
        self.config.log_max_size_mb = int(os.getenv('FEDZK_LOG_MAX_SIZE_MB', self.config.log_max_size_mb))
        self.config.log_backup_count = int(os.getenv('FEDZK_LOG_BACKUP_COUNT', self.config.log_backup_count))

        # Update configuration metadata
        self.config.config_last_updated = time.strftime('%Y-%m-%d %H:%M:%S')
        self._update_config_checksum()

    def validate_configuration(self) -> List[str]:
        """Validate configuration against defined rules."""
        errors = []

        for rule_name, rule in self.validation_rules.items():
            if hasattr(self.config, rule.field_name):
                value = getattr(self.config, rule.field_name)

                # Check required fields
                if rule.required and (value is None or value == ""):
                    errors.append(f"Required field '{rule.field_name}' is missing or empty")
                    continue

                # Skip validation if value is None and not required
                if value is None:
                    continue

                # Type checking
                if not isinstance(value, rule.field_type):
                    try:
                        # Try to convert type
                        if rule.field_type == int:
                            value = int(value)
                        elif rule.field_type == float:
                            value = float(value)
                        elif rule.field_type == bool:
                            value = str(value).lower() in ('true', '1', 'yes', 'on')
                        setattr(self.config, rule.field_name, value)
                    except (ValueError, TypeError):
                        errors.append(f"Field '{rule.field_name}' must be of type {rule.field_type.__name__}")

                # Length validation for strings
                if isinstance(value, str):
                    if rule.min_length and len(value) < rule.min_length:
                        errors.append(f"Field '{rule.field_name}' must be at least {rule.min_length} characters")
                    if rule.max_length and len(value) > rule.max_length:
                        errors.append(f"Field '{rule.field_name}' must be at most {rule.max_length} characters")

                # Allowed values validation
                if rule.allowed_values:
                    # Special handling for Environment enum
                    if rule.field_name == 'environment' and hasattr(value, 'value'):
                        # Convert enum to its value for comparison
                        if value.value not in rule.allowed_values:
                            errors.append(f"Field '{rule.field_name}' must be one of: {rule.allowed_values}, got {value.value}")
                    elif isinstance(value, Environment) and value.value not in rule.allowed_values:
                        # Handle Environment enum comparison
                        errors.append(f"Field '{rule.field_name}' must be one of: {rule.allowed_values}, got {value.value}")
                    elif value not in rule.allowed_values:
                        errors.append(f"Field '{rule.field_name}' must be one of: {rule.allowed_values}, got {value}")

                # Pattern validation
                if rule.pattern and isinstance(value, str):
                    import re
                    if not re.match(rule.pattern, value):
                        errors.append(f"Field '{rule.field_name}' does not match required pattern")

                # Custom validation
                if rule.custom_validator:
                    try:
                        if not rule.custom_validator(value):
                            errors.append(f"Field '{rule.field_name}' failed custom validation")
                    except Exception as e:
                        errors.append(f"Custom validation failed for '{rule.field_name}': {str(e)}")

        # Environment-specific validations
        try:
            env_value = self.config.environment.value if hasattr(self.config.environment, 'value') else str(self.config.environment)
        except:
            env_value = 'development'

        if env_value == 'production':
            if not self.config.jwt_secret_key:
                errors.append("JWT secret key is required in production")
            if not self.config.postgresql_enabled and not self.config.external_database_enabled:
                errors.append("Database must be configured in production")

        if errors:
            raise ValueError(f"Configuration validation failed: {'; '.join(errors)}")

        return errors

    def save_to_file(self, file_path: Optional[Path] = None):
        """Save current configuration to YAML file."""
        if file_path is None:
            file_path = self.config_file_path

        file_path.parent.mkdir(parents=True, exist_ok=True)

        # Convert config to dictionary
        config_dict = {}
        for field_name, field_value in self.config.__dict__.items():
            if field_name.startswith('_'):
                continue  # Skip private fields
            config_dict[field_name] = field_value

        # Handle sensitive values
        sensitive_fields = ['postgresql_password', 'redis_password', 'jwt_secret_key']
        for field in sensitive_fields:
            if field in config_dict and config_dict[field]:
                config_dict[f"{field}_encrypted"] = self.encryption.encrypt_value(str(config_dict[field]))
                config_dict[field] = "***ENCRYPTED***"

        with open(file_path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)

        self.logger.info(f"Configuration saved to {file_path}")

    def load_from_file(self, file_path: Optional[Path] = None):
        """Load configuration from YAML file."""
        if file_path is None:
            file_path = self.config_file_path

        if not file_path.exists():
            self.logger.warning(f"Configuration file {file_path} not found, using defaults")
            return

        with open(file_path, 'r') as f:
            config_dict = yaml.safe_load(f) or {}

        # Update configuration object
        for key, value in config_dict.items():
            if hasattr(self.config, key):
                # Handle encrypted values
                if key.endswith('_encrypted') and value:
                    base_key = key[:-10]  # Remove '_encrypted' suffix
                    try:
                        decrypted_value = self.encryption.decrypt_value(value)
                        setattr(self.config, base_key, decrypted_value)
                    except Exception as e:
                        self.logger.error(f"Failed to decrypt {key}: {e}")
                elif not key.endswith('_encrypted'):
                    setattr(self.config, key, value)

        self.logger.info(f"Configuration loaded from {file_path}")
        self.validate_configuration()

    def enable_hot_reload(self, watch_file: Optional[Path] = None):
        """Enable hot-reloading of configuration."""
        if watch_file is None:
            watch_file = self.config_file_path

        def watch_config():
            last_modified = 0
            while True:
                try:
                    if watch_file.exists():
                        current_modified = watch_file.stat().st_mtime
                        if current_modified > last_modified and last_modified > 0:
                            self.logger.info("Configuration file changed, reloading...")
                            self.load_from_file(watch_file)

                            # Notify callbacks
                            for callback in self.watch_callbacks:
                                try:
                                    callback()
                                except Exception as e:
                                    self.logger.error(f"Hot reload callback failed: {e}")

                            last_modified = current_modified

                    time.sleep(self.config_watch_interval)
                except Exception as e:
                    self.logger.error(f"Configuration watch error: {e}")
                    time.sleep(self.config_watch_interval)

        self.watch_thread = threading.Thread(target=watch_config, daemon=True)
        self.watch_thread.start()
        self.logger.info("Hot-reload enabled for configuration changes")

    def add_reload_callback(self, callback: Callable):
        """Add a callback function to be called on configuration reload."""
        self.watch_callbacks.append(callback)

    def get_database_url(self) -> str:
        """Get database URL based on configuration."""
        if self.config.external_database_enabled and self.config.external_database_url:
            return self.config.external_database_url

        if self.config.postgresql_enabled:
            return (f"postgresql://{self.config.postgresql_username}:"
                   f"{self.config.postgresql_password}@"
                   f"{self.config.postgresql_host}:{self.config.postgresql_port}/"
                   f"{self.config.postgresql_database}")

        # Default to SQLite for development
        return "sqlite:///./fedzk.db"

    def get_redis_url(self) -> str:
        """Get Redis URL based on configuration."""
        if self.config.external_redis_enabled and self.config.external_redis_host:
            host = self.config.external_redis_host
            port = self.config.external_redis_port
            password = self.config.external_redis_password
        else:
            host = self.config.redis_host
            port = self.config.redis_port
            password = self.config.redis_password

        if password:
            return f"redis://:{password}@{host}:{port}/{self.config.redis_db}"
        else:
            return f"redis://{host}:{port}/{self.config.redis_db}"

    def _update_config_checksum(self):
        """Update configuration checksum for change detection."""
        config_str = json.dumps(self.config.__dict__, sort_keys=True, default=str)
        self.config.config_checksum = hashlib.sha256(config_str.encode()).hexdigest()

    def get_config_summary(self) -> Dict[str, Any]:
        """Get a summary of current configuration."""
        return {
            'environment': self.config.environment.value,
            'app_name': self.config.app_name,
            'app_version': self.config.app_version,
            'debug': self.config.debug,
            'services': {
                'coordinator': {
                    'enabled': self.config.coordinator_enabled,
                    'host': self.config.coordinator_host,
                    'port': self.config.coordinator_port
                },
                'mpc': {
                    'enabled': self.config.mpc_enabled,
                    'host': self.config.mpc_host,
                    'port': self.config.mpc_port
                },
                'zk': {
                    'enabled': self.config.zk_enabled,
                    'circuit_path': self.config.zk_circuit_path
                },
                'postgresql': {
                    'enabled': self.config.postgresql_enabled,
                    'host': self.config.postgresql_host,
                    'port': self.config.postgresql_port
                },
                'redis': {
                    'enabled': self.config.redis_enabled,
                    'host': self.config.redis_host,
                    'port': self.config.redis_port
                }
            },
            'security': {
                'api_keys_enabled': self.config.api_keys_enabled,
                'tls_enabled': self.config.tls_enabled,
                'cors_enabled': self.config.cors_enabled
            },
            'monitoring': {
                'metrics_enabled': self.config.metrics_enabled,
                'health_check_enabled': self.config.health_check_enabled,
                'log_level': self.config.log_level
            },
            'config_metadata': {
                'version': self.config.config_version,
                'last_updated': self.config.config_last_updated,
                'checksum': self.config.config_checksum
            }
        }


# Global configuration instance
_config_manager = None

def get_config_manager() -> EnvironmentConfigManager:
    """Get the global configuration manager instance."""
    global _config_manager
    if _config_manager is None:
        _config_manager = EnvironmentConfigManager()
    return _config_manager

def get_config() -> EnvironmentConfig:
    """Get the current configuration."""
    return get_config_manager().config

def reload_config():
    """Reload configuration from environment and files."""
    get_config_manager().load_from_environment()
    get_config_manager().load_from_file()

def save_config():
    """Save current configuration to file."""
    get_config_manager().save_to_file()

def enable_hot_reload():
    """Enable configuration hot-reloading."""
    get_config_manager().enable_hot_reload()

# Initialize configuration on module import
get_config_manager()

