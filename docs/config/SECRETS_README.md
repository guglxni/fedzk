# FEDzk Secrets Management System

## Overview

The FEDzk Secrets Management System provides enterprise-grade secret storage, rotation, auditing, and backup capabilities. It supports multiple external providers and ensures secure, compliant secret management for production environments.

## Features

### ✅ External Provider Integration
- **HashiCorp Vault**: Enterprise-grade secret storage with advanced access controls
- **AWS Secrets Manager**: Cloud-native secret management with automatic rotation
- **Azure Key Vault**: Microsoft's enterprise secret management solution
- **GCP Secret Manager**: Google Cloud's secure secret storage
- **Local Encrypted Storage**: Secure fallback with AES-128 encryption

### ✅ Secure Secret Rotation
- **Automatic Rotation**: Background scheduler for secret rotation
- **Rotation Policies**: Never, Hourly, Daily, Weekly, Monthly, On-Access
- **Key Generation**: Secure random key generation for new secrets
- **Zero-Downtime Rotation**: Seamless rotation without service interruption

### ✅ Comprehensive Audit Logging
- **Access Logging**: All secret access operations logged
- **Security Events**: Failed access attempts and suspicious activity
- **Audit Trails**: Complete history of secret operations
- **Compliance Reports**: Generate audit reports for compliance

### ✅ Backup and Recovery
- **Point-in-Time Backups**: Create backups at any time
- **Incremental Backups**: Efficient backup of only changed secrets
- **Disaster Recovery**: Restore secrets from backups
- **Backup Integrity**: Verify backup integrity and completeness
- **Automated Cleanup**: Remove old backups automatically

## Quick Start

### 1. Basic Usage

```python
from fedzk.config.secrets_manager import get_secrets_manager

# Get secrets manager
secrets = get_secrets_manager()

# Store a secret
secrets.store_secret('db_password', 'my_secret_password', 'Database password')

# Retrieve a secret
password = secrets.retrieve_secret('db_password')

# List all secrets
all_secrets = secrets.list_secrets()
```

### 2. External Provider Configuration

#### HashiCorp Vault
```bash
export FEDZK_SECRETS_PROVIDER=hashicorp_vault
export VAULT_ADDR=https://vault.company.com:8200
export VAULT_TOKEN=your_vault_token
```

#### AWS Secrets Manager
```bash
export FEDZK_SECRETS_PROVIDER=aws_secrets_manager
export AWS_REGION=us-east-1
# AWS credentials via IAM roles or environment variables
```

#### Azure Key Vault
```bash
export FEDZK_SECRETS_PROVIDER=azure_key_vault
export AZURE_CLIENT_ID=your_client_id
export AZURE_CLIENT_SECRET=your_client_secret
export AZURE_TENANT_ID=your_tenant_id
```

### 3. Secret Rotation Setup

```python
from fedzk.config.secrets_manager import SecretRotationPolicy

# Set rotation policy
secrets.set_rotation_policy('api_key', SecretRotationPolicy.DAILY)

# Rotation runs automatically in background
```

### 4. Audit Logging

```python
# Get audit statistics
stats = secrets.get_audit_stats('db_password')
print(f"Total accesses: {stats['total_accesses']}")
print(f"Successful: {stats['successful_accesses']}")
print(f"Failed: {stats['failed_accesses']}")
```

### 5. Backup and Recovery

```python
# Create backup
backup_path = secrets.create_backup('production_backup')

# List backups
backups = secrets.list_backups()

# Restore from backup
result = secrets.restore_backup('production_backup')
print(f"Restored {result['restored_secrets']} secrets")
```

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `FEDZK_SECRETS_PROVIDER` | Secret storage provider | `local` |
| `FEDZK_ENCRYPTION_MASTER_KEY` | Master encryption key | Auto-generated |
| `VAULT_ADDR` | HashiCorp Vault address | - |
| `VAULT_TOKEN` | HashiCorp Vault token | - |
| `AWS_REGION` | AWS region for Secrets Manager | `us-east-1` |

### Provider-Specific Configuration

#### HashiCorp Vault
- `VAULT_ADDR`: Vault server URL
- `VAULT_TOKEN`: Authentication token
- `VAULT_MOUNT_POINT`: KV mount point (default: `secret`)

#### AWS Secrets Manager
- `AWS_REGION`: AWS region
- `AWS_ACCESS_KEY_ID`: AWS access key
- `AWS_SECRET_ACCESS_KEY`: AWS secret key
- Or use IAM roles/instance profiles

#### Azure Key Vault
- `AZURE_CLIENT_ID`: Service principal client ID
- `AZURE_CLIENT_SECRET`: Service principal secret
- `AZURE_TENANT_ID`: Azure tenant ID
- `AZURE_VAULT_URL`: Key Vault URL

## Secret Operations

### Storing Secrets

```python
# Simple string secret
secrets.store_secret('api_key', 'sk-123456789', 'Production API key')

# Complex secret (dictionary)
db_config = {
    'host': 'prod-db.company.com',
    'port': 5432,
    'username': 'app_user',
    'password': 'secret_password'
}
secrets.store_secret('database_config', db_config, 'Database configuration')
```

### Retrieving Secrets

```python
# Simple retrieval
api_key = secrets.retrieve_secret('api_key')

# Complex secret retrieval
db_config = secrets.retrieve_secret('database_config')
print(db_config['host'])  # prod-db.company.com
```

### Managing Secrets

```python
# List all secrets
all_secrets = secrets.list_secrets()
for name, metadata in all_secrets.items():
    print(f"{name}: {metadata.description}")

# Delete a secret
secrets.delete_secret('old_secret')

# Update secret metadata
secrets.update_secret_metadata('api_key', new_metadata)
```

## Rotation Policies

### Available Policies

- **`NEVER`**: Secrets never rotate automatically
- **`HOURLY`**: Rotate every hour
- **`DAILY`**: Rotate every day
- **`WEEKLY`**: Rotate every week
- **`MONTHLY`**: Rotate every month
- **`ON_ACCESS`**: Rotate on every access (not recommended for production)

### Setting Rotation Policies

```python
from fedzk.config.secrets_manager import SecretRotationPolicy

# Set daily rotation for API keys
secrets.set_rotation_policy('api_key', SecretRotationPolicy.DAILY)

# Set monthly rotation for database passwords
secrets.set_rotation_policy('db_password', SecretRotationPolicy.MONTHLY)

# Disable rotation
secrets.set_rotation_policy('static_key', SecretRotationPolicy.NEVER)
```

### Automatic Rotation

The rotation scheduler runs every hour and:
1. Checks all secrets against their rotation policies
2. Identifies secrets that need rotation
3. Generates new values for expiring secrets
4. Updates secrets with new values
5. Logs rotation events

## Audit Logging

### Access Events

All secret operations are logged with:
- **Timestamp**: When the operation occurred
- **User**: Who performed the operation
- **Operation**: read, write, delete, rotate
- **Success/Failure**: Whether the operation succeeded
- **Secret Name**: Which secret was accessed
- **IP Address**: Client IP address (if available)

### Audit Reports

```python
# Get audit statistics for a secret
stats = secrets.get_audit_stats('api_key')

# Get audit events with filtering
events = secrets.audit_log.get_audit_events(
    secret_name='api_key',
    operation='read',
    start_time=datetime.now() - timedelta(days=7)
)
```

### Audit Log Location

Audit logs are stored in: `./logs/secrets_audit.log`

Example log entry:
```json
{
  "secret_name": "api_key",
  "operation": "read",
  "timestamp": "2024-01-15T10:30:00",
  "user": "app_user",
  "success": true,
  "error_message": null
}
```

## Backup and Recovery

### Creating Backups

```python
# Create named backup
backup_path = secrets.create_backup('weekly_backup_2024_01_15')

# Create automatic backup
backup_path = secrets.create_backup()  # Uses timestamp
```

### Backup Contents

Each backup contains:
- **Secret Values**: Encrypted secret data
- **Metadata**: Access counts, creation dates, rotation policies
- **Audit Information**: Recent access logs
- **Integrity Checksums**: For backup verification

### Listing Backups

```python
backups = secrets.list_backups()
for backup in backups:
    print(f"Name: {backup['name']}")
    print(f"Created: {backup['created_at']}")
    print(f"Secrets: {backup['secret_count']}")
    print(f"Size: {backup['size_mb']:.2f} MB")
```

### Restoring from Backups

```python
# Dry run first
result = secrets.restore_backup('weekly_backup', dry_run=True)
print(f"Would restore {result['restored_secrets']} secrets")

# Perform actual restore
result = secrets.restore_backup('weekly_backup')
print(f"Restored {result['restored_secrets']} secrets")
if result['errors']:
    print("Errors:", result['errors'])
```

### Backup Management

```python
# Clean up old backups (older than 30 days)
deleted = secrets.backup_manager.cleanup_old_backups(keep_days=30)
print(f"Deleted {deleted} old backups")
```

## Security Features

### Encryption
- **Algorithm**: AES-128-CBC via Fernet
- **Key Derivation**: PBKDF2 with SHA256, 100,000 iterations
- **Salt**: Unique salt per master key
- **Master Key**: Configurable via environment variable

### Access Controls
- **Provider Permissions**: Respects external provider access controls
- **Local Encryption**: Additional encryption layer for local storage
- **Audit Trail**: Complete logging of all access attempts
- **Rotation Policies**: Automatic credential rotation

### Compliance
- **SOC 2**: Audit logging and access controls
- **GDPR**: Data encryption and access logging
- **PCI DSS**: Secure credential storage
- **HIPAA**: Healthcare data protection

## Integration Examples

### With FEDzk Configuration

```python
# config/environment.py integration
from fedzk.config.secrets_manager import get_secrets_manager

class EnvironmentConfig:
    def __init__(self):
        self.secrets = get_secrets_manager()

    @property
    def database_url(self):
        # Automatically resolve from secrets
        db_config = self.secrets.retrieve_secret('database_config')
        return f"postgresql://{db_config['username']}:{db_config['password']}@{db_config['host']}:{db_config['port']}/{db_config['database']}"
```

### With Application Code

```python
# app.py
from fedzk.config.secrets_manager import get_secrets_manager

class DatabaseConnection:
    def __init__(self):
        self.secrets = get_secrets_manager()

    def get_connection(self):
        password = self.secrets.retrieve_secret('db_password')
        # Use password for database connection
        return create_connection(password=password)
```

### With CI/CD Pipelines

```bash
# Deploy script
#!/bin/bash

# Store new secrets for deployment
python -c "
from fedzk.config.secrets_manager import store_secret
store_secret('deploy_key', '$DEPLOY_KEY', 'Deployment key for $ENV')
"

# Rotate secrets after deployment
python -c "
from fedzk.config.secrets_manager import get_secrets_manager
secrets = get_secrets_manager()
secrets.set_rotation_policy('deploy_key', 'DAILY')
"
```

## Monitoring and Alerting

### Key Metrics

- **Access Patterns**: Normal vs. suspicious access patterns
- **Failure Rates**: High failure rates may indicate attacks
- **Rotation Success**: Monitor automatic rotation success
- **Backup Status**: Ensure backups are created and valid

### Alert Conditions

```python
# Monitor for suspicious activity
stats = secrets.get_audit_stats('critical_secret')

if stats['failed_accesses'] > 10:
    alert_security_team("High failed access rate for critical_secret")

if stats['total_accesses'] > 1000:  # per hour
    alert_security_team("Unusual access volume for critical_secret")
```

### Log Analysis

```bash
# Search for failed access attempts
grep '"success": false' logs/secrets_audit.log

# Count operations by type
grep '"operation"' logs/secrets_audit.log | jq -r '.operation' | sort | uniq -c

# Find secrets with high access frequency
jq -r '.secret_name' logs/secrets_audit.log | sort | uniq -c | sort -nr | head -10
```

## Troubleshooting

### Common Issues

1. **Provider Connection Failed**
   ```
   Error: HashiCorp Vault connection failed
   Solution: Check VAULT_ADDR and VAULT_TOKEN environment variables
   ```

2. **Encryption Key Missing**
   ```
   Error: Master encryption key not found
   Solution: Set FEDZK_ENCRYPTION_MASTER_KEY environment variable
   ```

3. **Backup Restore Failed**
   ```
   Error: Backup integrity check failed
   Solution: Verify backup file hasn't been tampered with
   ```

4. **Rotation Not Working**
   ```
   Error: Secret rotation failed
   Solution: Check rotation policy and secret permissions
   ```

### Debug Mode

Enable debug logging:
```bash
export FEDZK_LOG_LEVEL=DEBUG
```

### Health Checks

```python
# Check secrets manager health
status = secrets.get_status()
print(f"Provider: {status['provider']}")
print(f"Secrets: {status['total_secrets']}")
print(f"Rotation Active: {status['rotation_active']}")
```

## Best Practices

### Production Deployment

1. **Use External Providers**: Always use HashiCorp Vault or AWS Secrets Manager in production
2. **Enable Rotation**: Set appropriate rotation policies for all secrets
3. **Regular Backups**: Create backups regularly and test restoration
4. **Monitor Access**: Set up alerting for suspicious access patterns
5. **Access Controls**: Implement least-privilege access to secrets

### Secret Management

1. **Naming Conventions**: Use consistent naming (e.g., `service_env_purpose`)
2. **Documentation**: Document all secrets and their purposes
3. **Ownership**: Assign ownership of secrets to teams
4. **Review Process**: Regular review of secret access and rotation policies

### Security

1. **Key Management**: Store master encryption keys securely
2. **Network Security**: Use TLS for all secret provider communications
3. **Access Logging**: Enable comprehensive audit logging
4. **Regular Rotation**: Rotate secrets regularly, especially in production

---

*The FEDzk Secrets Management System provides enterprise-grade security with comprehensive audit trails, automatic rotation, and disaster recovery capabilities. It ensures secrets are stored securely, accessed appropriately, and rotated regularly to maintain security posture.*
