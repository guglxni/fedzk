# FEDzk Environment Configuration System

## Overview

The FEDzk Environment Configuration System implements comprehensive configuration management following 12-factor app principles. It provides secure, validated, and hot-reloadable configuration for production deployments.

## Features

### ✅ 12-Factor App Principles
- **Config as Environment Variables**: All configuration through environment variables
- **Environment Parity**: Consistent configuration across development, staging, and production
- **Backing Services**: Configurable database and external service connections
- **Strict Separation**: Clear separation between code and configuration

### ✅ Configuration Validation & Type Checking
- **Type Validation**: Automatic type checking for all configuration values
- **Range Validation**: Min/max value validation for numeric fields
- **Pattern Validation**: Regex pattern validation for strings
- **Required Field Validation**: Ensures critical configuration is present
- **Cross-Field Validation**: Validates relationships between configuration fields

### ✅ Hot-Reloading
- **File Watching**: Automatic reload when configuration files change
- **Environment Monitoring**: Detects environment variable changes
- **Signal-Based Reload**: SIGHUP signal support for manual reload
- **Callback System**: Notifies components of configuration changes
- **Thread-Safe**: Safe concurrent access and updates

### ✅ Security & Encryption
- **Sensitive Value Encryption**: Automatic encryption of passwords, keys, and secrets
- **Secure Key Management**: PBKDF2-based key derivation with salt
- **Fernet Encryption**: AES-128-CBC encryption for sensitive data
- **Secure Storage**: Encrypted storage for secrets and sensitive configuration
- **Key Rotation**: Support for encryption key rotation

## Quick Start

### 1. Basic Usage

```python
from fedzk.config.environment import get_config_manager

# Get configuration manager
config_manager = get_config_manager()

# Access configuration
config = config_manager.config
print(f"App: {config.app_name}")
print(f"Port: {config.port}")
print(f"Environment: {config.environment.value}")
```

### 2. Environment Variables

Set configuration via environment variables:

```bash
# Basic configuration
export FEDZK_APP_NAME="MyFEDzkApp"
export FEDZK_PORT=8080
export FEDZK_ENVIRONMENT=production
export FEDZK_DEBUG=false

# Database configuration
export FEDZK_POSTGRESQL_ENABLED=true
export FEDZK_POSTGRESQL_HOST=localhost
export FEDZK_POSTGRESQL_PASSWORD=my_secure_password

# Security configuration
export FEDZK_ENCRYPTION_MASTER_KEY=your_master_key_here
export FEDZK_JWT_SECRET_KEY=your_jwt_secret_here
```

### 3. Configuration Files

Create YAML configuration files:

```yaml
# config/environment.yaml
app_name: "FEDzk"
environment: "production"
port: 8000
debug: false

coordinator_enabled: true
mpc_enabled: true
zk_enabled: true

postgresql_enabled: true
redis_enabled: true
tls_enabled: true
```

### 4. Encryption Setup

```python
from fedzk.config.encryption import ConfigEncryptionManager

# Setup encryption
enc_manager = ConfigEncryptionManager()
enc_manager.setup_encryption_keys()

# Encrypt sensitive values
config = {
    'database_password': 'my_secret_password',
    'api_key': 'sk-123456789'
}

encrypted_config = enc_manager.encrypt_sensitive_config(config)
```

### 5. Hot-Reloading

```python
from fedzk.config.hot_reload import setup_hot_reload

# Setup hot reload
config_manager = get_config_manager()
hot_reload = setup_hot_reload(config_manager)

# Start watching for changes
hot_reload.start_watching()

# Configuration will automatically reload on changes
```

## Configuration Fields

### Application Configuration
- `FEDZK_APP_NAME`: Application name (default: "FEDzk")
- `FEDZK_APP_VERSION`: Application version
- `FEDZK_ENVIRONMENT`: Environment (development/staging/production/testing)
- `FEDZK_DEBUG`: Debug mode (true/false)

### Server Configuration
- `FEDZK_HOST`: Server host (default: "0.0.0.0")
- `FEDZK_PORT`: Server port (1024-65535)

### FEDzk Services
- `FEDZK_COORDINATOR_ENABLED`: Enable coordinator service
- `FEDZK_COORDINATOR_HOST`: Coordinator host
- `FEDZK_COORDINATOR_PORT`: Coordinator port
- `FEDZK_MPC_ENABLED`: Enable MPC service
- `FEDZK_MPC_HOST`: MPC server host
- `FEDZK_MPC_PORT`: MPC server port
- `FEDZK_ZK_ENABLED`: Enable ZK circuits

### Database Configuration
- `FEDZK_POSTGRESQL_ENABLED`: Enable PostgreSQL
- `FEDZK_POSTGRESQL_HOST`: PostgreSQL host
- `FEDZK_POSTGRESQL_PORT`: PostgreSQL port
- `FEDZK_POSTGRESQL_DATABASE`: Database name
- `FEDZK_POSTGRESQL_USERNAME`: Database username
- `FEDZK_POSTGRESQL_PASSWORD`: Database password (or use _ENCRYPTED)

### Security Configuration
- `FEDZK_ENCRYPTION_MASTER_KEY`: Master encryption key
- `FEDZK_JWT_SECRET_KEY`: JWT signing secret
- `FEDZK_API_KEYS_ENABLED`: Enable API key authentication
- `FEDZK_TLS_ENABLED`: Enable TLS/SSL
- `FEDZK_TLS_CERT_PATH`: TLS certificate path
- `FEDZK_TLS_KEY_PATH`: TLS private key path

### Monitoring & Logging
- `FEDZK_LOG_LEVEL`: Logging level (DEBUG/INFO/WARNING/ERROR/CRITICAL)
- `FEDZK_METRICS_ENABLED`: Enable metrics collection
- `FEDZK_HEALTH_CHECK_ENABLED`: Enable health checks

## Validation Rules

The system includes comprehensive validation:

### Type Validation
```python
# String fields
app_name: str (required, 1-50 chars)

# Numeric fields
port: int (1024-65535)
model_batch_size: int (1-1024)

# Boolean fields
debug: bool
tls_enabled: bool
```

### Pattern Validation
```python
# Hostname validation
host: must be valid hostname or IP

# Version validation
app_version: must match semantic versioning (x.y.z)
```

### Cross-Field Validation
```python
# Port uniqueness
coordinator_port != mpc_port != port

# Service dependencies
if mpc_enabled: coordinator_enabled must be true

# Environment-specific requirements
if environment == "production": jwt_secret_key required
```

## Security Features

### Encryption
- **Algorithm**: Fernet (AES-128-CBC)
- **Key Derivation**: PBKDF2 with SHA256
- **Iterations**: 100,000
- **Salt**: Per-master-key salt

### Secure Storage
```python
from fedzk.config.encryption import SecureConfigStorage

storage = SecureConfigStorage()
storage.store_secret('api_key', 'sk-12345', 'Production API key')
retrieved_key = storage.retrieve_secret('api_key')
```

### Key Rotation
```python
# Rotate encryption key
new_key_info = encryption.rotate_key('new_master_key')
print(f"Key rotated: {new_key_info['rotation_timestamp']}")
```

## Hot-Reload Features

### File Watching
- Watches `config/environment.yaml`
- Watches `config/secrets.yaml`
- Automatic reload on file changes
- Configurable watch interval

### Environment Monitoring
- Monitors FEDZK_ environment variables
- Detects variable changes
- Immediate reload on environment updates

### Signal Handling
```bash
# Manual reload
kill -HUP <process_id>
```

### Callbacks
```python
def on_config_reload():
    print("Configuration reloaded!")

hot_reload.add_reload_callback(on_config_reload)
```

## Environment Examples

### Development
```bash
export FEDZK_ENVIRONMENT=development
export FEDZK_DEBUG=true
export FEDZK_LOG_LEVEL=DEBUG
export FEDZK_COORDINATOR_ENABLED=true
export FEDZK_MPC_ENABLED=true
export FEDZK_ZK_ENABLED=true
```

### Production
```bash
export FEDZK_ENVIRONMENT=production
export FEDZK_DEBUG=false
export FEDZK_LOG_LEVEL=WARNING
export FEDZK_COORDINATOR_ENABLED=true
export FEDZK_MPC_ENABLED=true
export FEDZK_ZK_ENABLED=true
export FEDZK_POSTGRESQL_ENABLED=true
export FEDZK_REDIS_ENABLED=true
export FEDZK_TLS_ENABLED=true
export FEDZK_METRICS_ENABLED=true
export FEDZK_ENCRYPTION_MASTER_KEY=<secure_key>
export FEDZK_JWT_SECRET_KEY=<jwt_secret>
```

## Testing

Run the configuration test suite:

```bash
python -m pytest tests/test_environment_config.py -v
```

Or run the demonstration:

```bash
python demo_task_8_2_1_environment_config.py
```

## Best Practices

### 1. Environment Variables
- Use environment variables for all configuration
- Never hardcode sensitive values in code
- Use descriptive variable names with FEDZK_ prefix

### 2. Security
- Always set FEDZK_ENCRYPTION_MASTER_KEY in production
- Use strong, unique keys for JWT and encryption
- Rotate encryption keys regularly
- Store secrets securely (not in version control)

### 3. Validation
- Validate configuration on startup
- Use type hints for better IDE support
- Test configuration validation thoroughly

### 4. Hot-Reloading
- Use hot-reload in development environments
- Implement proper callback handling
- Monitor for reload-related issues

### 5. File Organization
```
config/
├── environment.yaml          # Main configuration
├── environment.production.yaml   # Production template
├── environment.sample.yaml     # Sample configuration
└── secrets.yaml               # Encrypted secrets (not in git)
```

## Troubleshooting

### Common Issues

1. **Configuration not loading**
   - Check environment variable names (FEDZK_ prefix)
   - Verify file paths and permissions
   - Check YAML syntax in configuration files

2. **Encryption errors**
   - Verify FEDZK_ENCRYPTION_MASTER_KEY is set
   - Check encrypted value format
   - Ensure same key is used for encrypt/decrypt

3. **Validation failures**
   - Review validation error messages
   - Check field types and ranges
   - Verify cross-field dependencies

4. **Hot-reload not working**
   - Check file permissions
   - Verify watch paths exist
   - Check for thread-related issues

### Debug Mode

Enable debug logging for troubleshooting:

```bash
export FEDZK_DEBUG=true
export FEDZK_LOG_LEVEL=DEBUG
```

## API Reference

### EnvironmentConfigManager
- `load_from_environment()`: Load from environment variables
- `load_from_file(path)`: Load from YAML file
- `save_to_file(path)`: Save to YAML file
- `validate_configuration()`: Validate current configuration
- `get_database_url()`: Get formatted database URL
- `get_redis_url()`: Get formatted Redis URL

### ConfigValidator
- `validate_config(config)`: Validate configuration object
- `print_validation_report(results)`: Print formatted validation report
- `get_validation_summary(results)`: Get validation summary

### ConfigEncryption
- `encrypt_value(value, field_name)`: Encrypt a value
- `decrypt_value(encrypted_value, field_name)`: Decrypt a value
- `is_encrypted(value)`: Check if value is encrypted

### ConfigHotReload
- `start_watching()`: Start file/environment watching
- `stop_watching()`: Stop watching
- `force_reload()`: Force configuration reload
- `add_reload_callback(callback)`: Add reload callback

## Migration Guide

### From Previous Versions

If migrating from older FEDzk versions:

1. **Environment Variables**: Update variable names to use FEDZK_ prefix
2. **Configuration Files**: Convert to new YAML format
3. **Encryption**: Set up new encryption master key
4. **Validation**: Update custom validation logic

### Legacy Compatibility

The system maintains backward compatibility where possible, but some breaking changes may require updates:

- Old environment variable names (without FEDZK_ prefix)
- Legacy configuration file formats
- Deprecated encryption methods

---

*This configuration system ensures FEDzk follows modern DevOps practices with security, reliability, and maintainability as core principles.*
