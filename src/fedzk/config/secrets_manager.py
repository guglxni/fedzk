#!/usr/bin/env python3
"""
Secrets Management System
========================

Enterprise-grade secrets management with external provider integration.
Supports HashiCorp Vault, AWS Secrets Manager, and secure local storage.
"""

import os
import json
import time
import hashlib
import threading
from pathlib import Path
from typing import Dict, Any, Optional, List, Union, Callable
from datetime import datetime, timedelta
from enum import Enum
import logging
import base64
import secrets

from .encryption import ConfigEncryption, ConfigEncryptionError


class SecretProvider(Enum):
    """Supported secret providers."""
    LOCAL = "local"
    HASHICORP_VAULT = "hashicorp_vault"
    AWS_SECRETS_MANAGER = "aws_secrets_manager"
    AZURE_KEY_VAULT = "azure_key_vault"
    GCP_SECRET_MANAGER = "gcp_secret_manager"


class SecretRotationPolicy(Enum):
    """Secret rotation policies."""
    NEVER = "never"
    HOURLY = "hourly"
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    ON_ACCESS = "on_access"


class SecretMetadata:
    """Metadata for stored secrets."""

    def __init__(self, name: str, provider: SecretProvider, created_at: datetime = None,
                 last_accessed: datetime = None, last_rotated: datetime = None,
                 access_count: int = 0, rotation_policy: SecretRotationPolicy = SecretRotationPolicy.NEVER,
                 tags: Dict[str, str] = None, description: str = ""):
        self.name = name
        self.provider = provider
        self.created_at = created_at or datetime.now()
        self.last_accessed = last_accessed or datetime.now()
        self.last_rotated = last_rotated
        self.access_count = access_count
        self.rotation_policy = rotation_policy
        self.tags = tags or {}
        self.description = description
        self.version = 1

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'name': self.name,
            'provider': self.provider.value,
            'created_at': self.created_at.isoformat(),
            'last_accessed': self.last_accessed.isoformat(),
            'last_rotated': self.last_rotated.isoformat() if self.last_rotated else None,
            'access_count': self.access_count,
            'rotation_policy': self.rotation_policy.value,
            'tags': self.tags,
            'description': self.description,
            'version': self.version
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SecretMetadata':
        """Create from dictionary."""
        return cls(
            name=data['name'],
            provider=SecretProvider(data['provider']),
            created_at=datetime.fromisoformat(data['created_at']),
            last_accessed=datetime.fromisoformat(data['last_accessed']),
            last_rotated=datetime.fromisoformat(data['last_rotated']) if data.get('last_rotated') else None,
            access_count=data.get('access_count', 0),
            rotation_policy=SecretRotationPolicy(data.get('rotation_policy', 'never')),
            tags=data.get('tags', {}),
            description=data.get('description', '')
        )


class SecretAccessEvent:
    """Audit event for secret access."""

    def __init__(self, secret_name: str, operation: str, user: str = None,
                 ip_address: str = None, user_agent: str = None,
                 success: bool = True, error_message: str = None):
        self.secret_name = secret_name
        self.operation = operation  # 'read', 'write', 'delete', 'rotate'
        self.timestamp = datetime.now()
        self.user = user or os.getenv('USER', 'unknown')
        self.ip_address = ip_address
        self.user_agent = user_agent
        self.success = success
        self.error_message = error_message

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging."""
        return {
            'secret_name': self.secret_name,
            'operation': self.operation,
            'timestamp': self.timestamp.isoformat(),
            'user': self.user,
            'ip_address': self.ip_address,
            'user_agent': self.user_agent,
            'success': self.success,
            'error_message': self.error_message
        }


class HashiCorpVaultProvider:
    """HashiCorp Vault integration."""

    def __init__(self, vault_url: str = None, token: str = None, mount_point: str = "secret"):
        self.vault_url = vault_url or os.getenv('VAULT_ADDR')
        self.token = token or os.getenv('VAULT_TOKEN')
        self.mount_point = mount_point
        self.logger = logging.getLogger(__name__)

        # Lazy import to avoid hard dependency
        try:
            import hvac
            self.client = hvac.Client(url=self.vault_url, token=self.token)
            self.available = True
        except ImportError:
            self.logger.warning("hvac package not installed, HashiCorp Vault support disabled")
            self.client = None
            self.available = False

    def is_available(self) -> bool:
        """Check if Vault is available."""
        if not self.available or not self.client:
            return False
        try:
            return self.client.is_authenticated()
        except Exception:
            return False

    def store_secret(self, path: str, data: Dict[str, Any]) -> bool:
        """Store secret in Vault."""
        if not self.is_available():
            return False
        try:
            full_path = f"{self.mount_point}/data/{path}"
            self.client.secrets.kv.v2.create_or_update_secret(
                path=path,
                secret=data,
                mount_point=self.mount_point
            )
            return True
        except Exception as e:
            self.logger.error(f"Failed to store secret in Vault: {e}")
            return False

    def retrieve_secret(self, path: str) -> Optional[Dict[str, Any]]:
        """Retrieve secret from Vault."""
        if not self.is_available():
            return None
        try:
            response = self.client.secrets.kv.v2.read_secret_version(
                path=path,
                mount_point=self.mount_point
            )
            return response['data']['data']
        except Exception as e:
            self.logger.error(f"Failed to retrieve secret from Vault: {e}")
            return None

    def delete_secret(self, path: str) -> bool:
        """Delete secret from Vault."""
        if not self.is_available():
            return False
        try:
            self.client.secrets.kv.v2.delete_metadata_and_all_versions(
                path=path,
                mount_point=self.mount_point
            )
            return True
        except Exception as e:
            self.logger.error(f"Failed to delete secret from Vault: {e}")
            return False


class AWSSecretsManagerProvider:
    """AWS Secrets Manager integration."""

    def __init__(self, region_name: str = None):
        self.region_name = region_name or os.getenv('AWS_REGION', 'us-east-1')
        self.logger = logging.getLogger(__name__)

        # Lazy import to avoid hard dependency
        try:
            import boto3
            self.client = boto3.client('secretsmanager', region_name=self.region_name)
            self.available = True
        except ImportError:
            self.logger.warning("boto3 package not installed, AWS Secrets Manager support disabled")
            self.client = None
            self.available = False

    def is_available(self) -> bool:
        """Check if AWS Secrets Manager is available."""
        if not self.available or not self.client:
            return False
        try:
            # Test with a simple operation
            self.client.list_secrets(MaxResults=1)
            return True
        except Exception:
            return False

    def store_secret(self, name: str, data: Dict[str, Any]) -> bool:
        """Store secret in AWS Secrets Manager."""
        if not self.is_available():
            return False
        try:
            secret_string = json.dumps(data)
            self.client.create_secret(
                Name=name,
                SecretString=secret_string,
                Tags=[{'Key': 'managed_by', 'Value': 'fedzk'}]
            )
            return True
        except self.client.exceptions.ResourceExistsException:
            # Update existing secret
            try:
                secret_string = json.dumps(data)
                self.client.update_secret(
                    SecretId=name,
                    SecretString=secret_string
                )
                return True
            except Exception as e:
                self.logger.error(f"Failed to update secret in AWS: {e}")
                return False
        except Exception as e:
            self.logger.error(f"Failed to store secret in AWS: {e}")
            return False

    def retrieve_secret(self, name: str) -> Optional[Dict[str, Any]]:
        """Retrieve secret from AWS Secrets Manager."""
        if not self.is_available():
            return None
        try:
            response = self.client.get_secret_value(SecretId=name)
            secret_string = response['SecretString']
            return json.loads(secret_string)
        except Exception as e:
            self.logger.error(f"Failed to retrieve secret from AWS: {e}")
            return None

    def delete_secret(self, name: str) -> bool:
        """Delete secret from AWS Secrets Manager."""
        if not self.is_available():
            return False
        try:
            self.client.delete_secret(
                SecretId=name,
                ForceDeleteWithoutRecovery=True
            )
            return True
        except Exception as e:
            self.logger.error(f"Failed to delete secret from AWS: {e}")
            return False


class SecretRotationManager:
    """Manages automatic secret rotation."""

    def __init__(self, secrets_manager):
        self.secrets_manager = secrets_manager
        self.rotation_thread = None
        self.running = False
        self.logger = logging.getLogger(__name__)

    def start_rotation_scheduler(self):
        """Start the automatic rotation scheduler."""
        if self.running:
            return

        self.running = True
        self.rotation_thread = threading.Thread(target=self._rotation_worker, daemon=True)
        self.rotation_thread.start()
        self.logger.info("Secret rotation scheduler started")

    def stop_rotation_scheduler(self):
        """Stop the automatic rotation scheduler."""
        self.running = False
        if self.rotation_thread:
            self.rotation_thread.join(timeout=5)
        self.logger.info("Secret rotation scheduler stopped")

    def _rotation_worker(self):
        """Background worker for secret rotation."""
        while self.running:
            try:
                self._check_and_rotate_secrets()
                time.sleep(3600)  # Check every hour
            except Exception as e:
                self.logger.error(f"Error in rotation worker: {e}")
                time.sleep(60)  # Wait a minute before retrying

    def _check_and_rotate_secrets(self):
        """Check all secrets and rotate if needed."""
        secrets = self.secrets_manager.list_secrets()
        now = datetime.now()

        for secret_name, metadata in secrets.items():
            if not isinstance(metadata, SecretMetadata):
                continue

            if self._should_rotate_secret(metadata, now):
                self.logger.info(f"Rotating secret: {secret_name}")
                if self._rotate_secret(secret_name, metadata):
                    self.logger.info(f"Successfully rotated secret: {secret_name}")
                else:
                    self.logger.error(f"Failed to rotate secret: {secret_name}")

    def _should_rotate_secret(self, metadata: SecretMetadata, now: datetime) -> bool:
        """Determine if a secret should be rotated."""
        if metadata.rotation_policy == SecretRotationPolicy.NEVER:
            return False

        if metadata.last_rotated is None:
            # Never rotated, rotate if it's old enough
            age = now - metadata.created_at
            return age > self._get_rotation_interval(metadata.rotation_policy)

        # Check if rotation interval has passed
        time_since_rotation = now - metadata.last_rotated
        return time_since_rotation > self._get_rotation_interval(metadata.rotation_policy)

    def _get_rotation_interval(self, policy: SecretRotationPolicy) -> timedelta:
        """Get the rotation interval for a policy."""
        intervals = {
            SecretRotationPolicy.HOURLY: timedelta(hours=1),
            SecretRotationPolicy.DAILY: timedelta(days=1),
            SecretRotationPolicy.WEEKLY: timedelta(weeks=1),
            SecretRotationPolicy.MONTHLY: timedelta(days=30),
        }
        return intervals.get(policy, timedelta(days=365*100))  # Never

    def _rotate_secret(self, secret_name: str, metadata: SecretMetadata) -> bool:
        """Rotate a specific secret."""
        try:
            # Generate new secret value
            new_value = self._generate_new_secret_value(secret_name, metadata)

            # Store new value
            success = self.secrets_manager.store_secret(
                secret_name, new_value, metadata.description
            )

            if success:
                # Update metadata
                metadata.last_rotated = datetime.now()
                metadata.version += 1
                self.secrets_manager.update_secret_metadata(secret_name, metadata)

                # Log rotation event
                self.secrets_manager.audit_log.log_event(
                    SecretAccessEvent(secret_name, 'rotate', success=True)
                )

            return success

        except Exception as e:
            self.logger.error(f"Error rotating secret {secret_name}: {e}")
            return False

    def _generate_new_secret_value(self, secret_name: str, metadata: SecretMetadata) -> Dict[str, Any]:
        """Generate a new value for a secret."""
        # This is a simplified implementation
        # In practice, this would depend on the secret type
        if 'password' in secret_name.lower():
            return {'value': secrets.token_urlsafe(32)}
        elif 'key' in secret_name.lower():
            return {'value': base64.b64encode(secrets.token_bytes(32)).decode()}
        elif 'token' in secret_name.lower():
            return {'value': secrets.token_hex(32)}
        else:
            # Generic rotation
            return {'value': secrets.token_urlsafe(32), 'rotated_at': datetime.now().isoformat()}


class SecretAuditLogger:
    """Audit logging for secret access."""

    def __init__(self, log_file: Path = None):
        self.log_file = log_file or Path("./logs/secrets_audit.log")
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger(__name__)

    def log_event(self, event: SecretAccessEvent):
        """Log a secret access event."""
        try:
            with open(self.log_file, 'a') as f:
                json.dump(event.to_dict(), f)
                f.write('\n')
        except Exception as e:
            self.logger.error(f"Failed to log audit event: {e}")

    def get_audit_events(self, secret_name: str = None, operation: str = None,
                        start_time: datetime = None, end_time: datetime = None) -> List[Dict[str, Any]]:
        """Retrieve audit events with optional filtering."""
        events = []

        try:
            with open(self.log_file, 'r') as f:
                for line in f:
                    try:
                        event_data = json.loads(line.strip())
                        # Remove timestamp from event_data since it's set automatically in constructor
                        filtered_data = {k: v for k, v in event_data.items() if k != 'timestamp'}
                        event = SecretAccessEvent(**filtered_data)

                        # Apply filters
                        if secret_name and event.secret_name != secret_name:
                            continue
                        if operation and event.operation != operation:
                            continue
                        if start_time and event.timestamp < start_time:
                            continue
                        if end_time and event.timestamp > end_time:
                            continue

                        events.append(event_data)
                    except json.JSONDecodeError:
                        continue

        except FileNotFoundError:
            self.logger.warning("Audit log file not found")
        except Exception as e:
            self.logger.error(f"Error reading audit log: {e}")

        return events

    def get_access_stats(self, secret_name: str = None) -> Dict[str, Any]:
        """Get access statistics for secrets."""
        events = self.get_audit_events(secret_name=secret_name)
        stats = {
            'total_accesses': len(events),
            'successful_accesses': 0,
            'failed_accesses': 0,
            'operations': {},
            'recent_accesses': []
        }

        for event in events[-100:]:  # Last 100 events
            if event['success']:
                stats['successful_accesses'] += 1
            else:
                stats['failed_accesses'] += 1

            op = event['operation']
            stats['operations'][op] = stats['operations'].get(op, 0) + 1

            if len(stats['recent_accesses']) < 10:
                stats['recent_accesses'].append(event)

        return stats


class SecretBackupManager:
    """Manages secret backups and recovery."""

    def __init__(self, secrets_manager, backup_dir: Path = None):
        self.secrets_manager = secrets_manager
        self.backup_dir = backup_dir or Path("./backups/secrets")
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger(__name__)

    def create_backup(self, backup_name: str = None) -> Optional[Path]:
        """Create a backup of all secrets."""
        if backup_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_name = f"secrets_backup_{timestamp}"

        backup_file = self.backup_dir / f"{backup_name}.json"

        try:
            # Gather all secrets and metadata
            backup_data = {
                'created_at': datetime.now().isoformat(),
                'version': '1.0',
                'secrets': {},
                'metadata': {}
            }

            secrets = self.secrets_manager.list_secrets()
            for secret_name, metadata in secrets.items():
                # Get secret value
                secret_value = self.secrets_manager.retrieve_secret(secret_name)
                if secret_value is not None:
                    backup_data['secrets'][secret_name] = secret_value

                # Store metadata
                if isinstance(metadata, SecretMetadata):
                    backup_data['metadata'][secret_name] = metadata.to_dict()

            # Write backup file
            with open(backup_file, 'w') as f:
                json.dump(backup_data, f, indent=2)

            self.logger.info(f"Created secrets backup: {backup_file}")
            return backup_file

        except Exception as e:
            self.logger.error(f"Failed to create secrets backup: {e}")
            return None

    def list_backups(self) -> List[Dict[str, Any]]:
        """List available backups."""
        backups = []

        try:
            for backup_file in self.backup_dir.glob("*.json"):
                try:
                    with open(backup_file, 'r') as f:
                        backup_data = json.load(f)

                    backups.append({
                        'name': backup_file.stem,
                        'file': backup_file,
                        'created_at': backup_data.get('created_at'),
                        'secret_count': len(backup_data.get('secrets', {})),
                        'size_mb': backup_file.stat().st_size / (1024 * 1024)
                    })
                except Exception as e:
                    self.logger.warning(f"Error reading backup {backup_file}: {e}")

        except Exception as e:
            self.logger.error(f"Error listing backups: {e}")

        # Sort by creation time, newest first
        backups.sort(key=lambda x: x.get('created_at', ''), reverse=True)
        return backups

    def restore_backup(self, backup_name: str, dry_run: bool = False) -> Dict[str, Any]:
        """Restore secrets from a backup."""
        backup_file = self.backup_dir / f"{backup_name}.json"

        if not backup_file.exists():
            raise FileNotFoundError(f"Backup file not found: {backup_file}")

        result = {
            'restored_secrets': 0,
            'failed_secrets': 0,
            'errors': [],
            'dry_run': dry_run
        }

        try:
            with open(backup_file, 'r') as f:
                backup_data = json.load(f)

            secrets = backup_data.get('secrets', {})
            metadata = backup_data.get('metadata', {})

            for secret_name, secret_value in secrets.items():
                try:
                    if not dry_run:
                        # Restore secret
                        description = ""
                        if secret_name in metadata:
                            meta_dict = metadata[secret_name]
                            description = meta_dict.get('description', '')

                        success = self.secrets_manager.store_secret(
                            secret_name, secret_value, description
                        )

                        if success and secret_name in metadata:
                            # Restore metadata
                            meta_obj = SecretMetadata.from_dict(metadata[secret_name])
                            self.secrets_manager.update_secret_metadata(secret_name, meta_obj)

                    result['restored_secrets'] += 1

                except Exception as e:
                    result['failed_secrets'] += 1
                    result['errors'].append(f"Failed to restore {secret_name}: {e}")

            if not dry_run:
                self.logger.info(f"Restored {result['restored_secrets']} secrets from backup {backup_name}")
            else:
                self.logger.info(f"Dry run: would restore {result['restored_secrets']} secrets from backup {backup_name}")

        except Exception as e:
            result['errors'].append(f"Failed to read backup file: {e}")

        return result

    def cleanup_old_backups(self, keep_days: int = 30) -> int:
        """Clean up old backup files."""
        cutoff_date = datetime.now() - timedelta(days=keep_days)
        deleted_count = 0

        try:
            for backup_file in self.backup_dir.glob("secrets_backup_*.json"):
                try:
                    file_date = datetime.fromtimestamp(backup_file.stat().st_mtime)
                    if file_date < cutoff_date:
                        backup_file.unlink()
                        deleted_count += 1
                        self.logger.info(f"Deleted old backup: {backup_file}")
                except Exception as e:
                    self.logger.warning(f"Error checking backup file {backup_file}: {e}")

        except Exception as e:
            self.logger.error(f"Error during backup cleanup: {e}")

        return deleted_count


class FEDzkSecretsManager:
    """Main secrets management interface for FEDzk."""

    def __init__(self, provider: SecretProvider = SecretProvider.LOCAL,
                 encryption: Optional[ConfigEncryption] = None,
                 storage_path: Optional[Path] = None):
        self.provider = provider
        self.encryption = encryption or ConfigEncryption()
        self.logger = logging.getLogger(__name__)

        # Initialize providers
        self.providers = {
            SecretProvider.HASHICORP_VAULT: HashiCorpVaultProvider(),
            SecretProvider.AWS_SECRETS_MANAGER: AWSSecretsManagerProvider(),
        }

        # Initialize components
        self.audit_log = SecretAuditLogger()
        self.rotation_manager = SecretRotationManager(self)
        self.backup_manager = SecretBackupManager(self)

        # Local storage for metadata - use provided path or default
        if storage_path:
            self.metadata_file = storage_path / "secret_metadata.json"
            self.secrets_dir = storage_path / "secrets"
        else:
            self.metadata_file = Path("./config/secret_metadata.json")
            self.secrets_dir = Path("./secrets")

        # Ensure directories exist
        self.metadata_file.parent.mkdir(parents=True, exist_ok=True)
        self.secrets_dir.mkdir(parents=True, exist_ok=True)

        self._load_metadata()

        # Start rotation scheduler
        self.rotation_manager.start_rotation_scheduler()

    def _load_metadata(self):
        """Load secret metadata from storage."""
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, 'r') as f:
                    metadata_data = json.load(f)
                    self.metadata = {}
                    for name, data in metadata_data.items():
                        self.metadata[name] = SecretMetadata.from_dict(data)
            except Exception as e:
                self.logger.error(f"Failed to load secret metadata: {e}")
                self.metadata = {}
        else:
            self.metadata = {}

    def _save_metadata(self):
        """Save secret metadata to storage."""
        try:
            metadata_data = {}
            for name, meta in self.metadata.items():
                metadata_data[name] = meta.to_dict()

            with open(self.metadata_file, 'w') as f:
                json.dump(metadata_data, f, indent=2)
        except Exception as e:
            self.logger.error(f"Failed to save secret metadata: {e}")

    def store_secret(self, name: str, value: Union[str, Dict[str, Any]],
                    description: str = "", tags: Dict[str, str] = None) -> bool:
        """Store a secret."""
        try:
            # Ensure value is a dictionary
            if isinstance(value, str):
                secret_data = {'value': value}
            else:
                secret_data = value.copy()

            # Add metadata
            secret_data['_metadata'] = {
                'created_at': datetime.now().isoformat(),
                'description': description,
                'tags': tags or {}
            }

            success = False

            # Try external provider first
            if self.provider != SecretProvider.LOCAL:
                provider = self.providers.get(self.provider)
                if provider and provider.is_available():
                    success = provider.store_secret(name, secret_data)

            # Fall back to local storage
            if not success:
                encrypted_data = self.encryption.encrypt_value(json.dumps(secret_data), name)
                local_path = self.secrets_dir / f"{name}.enc"

                with open(local_path, 'w') as f:
                    f.write(encrypted_data)
                success = True

            if success:
                # Update metadata
                self.metadata[name] = SecretMetadata(
                    name=name,
                    provider=self.provider,
                    description=description,
                    tags=tags or {}
                )
                self._save_metadata()

                # Audit log
                self.audit_log.log_event(
                    SecretAccessEvent(name, 'write', success=True)
                )

            return success

        except Exception as e:
            self.logger.error(f"Failed to store secret {name}: {e}")
            self.audit_log.log_event(
                SecretAccessEvent(name, 'write', success=False, error_message=str(e))
            )
            return False

    def retrieve_secret(self, name: str) -> Optional[Union[str, Dict[str, Any]]]:
        """Retrieve a secret."""
        try:
            secret_data = None

            # Try external provider first
            if self.provider != SecretProvider.LOCAL:
                provider = self.providers.get(self.provider)
                if provider and provider.is_available():
                    secret_data = provider.retrieve_secret(name)

            # Fall back to local storage
            if secret_data is None:
                local_path = self.secrets_dir / f"{name}.enc"
                if local_path.exists():
                    with open(local_path, 'r') as f:
                        encrypted_data = f.read()
                    decrypted_data = self.encryption.decrypt_value(encrypted_data, name)
                    secret_data = json.loads(decrypted_data)

            if secret_data:
                # Update metadata
                if name in self.metadata:
                    self.metadata[name].last_accessed = datetime.now()
                    self.metadata[name].access_count += 1
                    self._save_metadata()

                # Audit log
                self.audit_log.log_event(
                    SecretAccessEvent(name, 'read', success=True)
                )

                # Return the actual value
                if isinstance(secret_data, dict) and 'value' in secret_data:
                    return secret_data['value']
                return secret_data

        except Exception as e:
            self.logger.error(f"Failed to retrieve secret {name}: {e}")
            self.audit_log.log_event(
                SecretAccessEvent(name, 'read', success=False, error_message=str(e))
            )

        return None

    def delete_secret(self, name: str) -> bool:
        """Delete a secret."""
        try:
            success = False

            # Try external provider first
            if self.provider != SecretProvider.LOCAL:
                provider = self.providers.get(self.provider)
                if provider and provider.is_available():
                    success = provider.delete_secret(name)

            # Delete local storage
            local_path = self.secrets_dir / f"{name}.enc"
            if local_path.exists():
                local_path.unlink()
                success = True

            if success:
                # Remove metadata
                if name in self.metadata:
                    del self.metadata[name]
                    self._save_metadata()

                # Audit log
                self.audit_log.log_event(
                    SecretAccessEvent(name, 'delete', success=True)
                )

            return success

        except Exception as e:
            self.logger.error(f"Failed to delete secret {name}: {e}")
            self.audit_log.log_event(
                SecretAccessEvent(name, 'delete', success=False, error_message=str(e))
            )
            return False

    def list_secrets(self) -> Dict[str, SecretMetadata]:
        """List all secrets with metadata."""
        return self.metadata.copy()

    def update_secret_metadata(self, name: str, metadata: SecretMetadata):
        """Update metadata for a secret."""
        self.metadata[name] = metadata
        self._save_metadata()

    def set_rotation_policy(self, name: str, policy: SecretRotationPolicy):
        """Set rotation policy for a secret."""
        if name in self.metadata:
            self.metadata[name].rotation_policy = policy
            self._save_metadata()

    def get_audit_stats(self, name: str = None) -> Dict[str, Any]:
        """Get audit statistics for secrets."""
        return self.audit_log.get_access_stats(name)

    def create_backup(self, name: str = None) -> Optional[Path]:
        """Create a backup of all secrets."""
        return self.backup_manager.create_backup(name)

    def list_backups(self) -> List[Dict[str, Any]]:
        """List available backups."""
        return self.backup_manager.list_backups()

    def restore_backup(self, name: str, dry_run: bool = False) -> Dict[str, Any]:
        """Restore secrets from a backup."""
        return self.backup_manager.restore_backup(name, dry_run)

    def get_status(self) -> Dict[str, Any]:
        """Get the status of the secrets management system."""
        return {
            'provider': self.provider.value,
            'total_secrets': len(self.metadata),
            'providers_available': {
                name.value: provider.is_available()
                for name, provider in self.providers.items()
            },
            'rotation_active': self.rotation_manager.running,
            'audit_enabled': True,
            'backup_enabled': True
        }

    def cleanup(self):
        """Clean up resources."""
        self.rotation_manager.stop_rotation_scheduler()


# Global instance
_secrets_manager = None

def get_secrets_manager() -> FEDzkSecretsManager:
    """Get the global secrets manager instance."""
    global _secrets_manager
    if _secrets_manager is None:
        # Determine provider from environment
        provider_env = os.getenv('FEDZK_SECRETS_PROVIDER', 'local')
        try:
            provider = SecretProvider(provider_env)
        except ValueError:
            provider = SecretProvider.LOCAL

        _secrets_manager = FEDzkSecretsManager(provider)
    return _secrets_manager

# Utility functions
def store_secret(name: str, value: Union[str, Dict[str, Any]], description: str = "") -> bool:
    """Store a secret."""
    return get_secrets_manager().store_secret(name, value, description)

def get_secret(name: str) -> Optional[Union[str, Dict[str, Any]]]:
    """Retrieve a secret."""
    return get_secrets_manager().retrieve_secret(name)

def delete_secret(name: str) -> bool:
    """Delete a secret."""
    return get_secrets_manager().delete_secret(name)

def list_secrets() -> Dict[str, SecretMetadata]:
    """List all secrets."""
    return get_secrets_manager().list_secrets()
