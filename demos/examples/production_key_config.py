#!/usr/bin/env python3
"""
FEDzk Production Key Management Configuration
=============================================

Production-ready configuration for FEDzk key management system.
Demonstrates enterprise deployment scenarios and security best practices.
"""

import os
from pathlib import Path
from fedzk.security.key_manager import (
    KeyManager,
    KeyType,
    KeyStorageType,
    KeyRotationPolicy,
    KeySecurityConfig,
    KeyRotationConfig,
    FileKeyStorage,
    EnvironmentKeyStorage,
    VaultKeyStorage,
    create_secure_key_manager
)

def create_development_configuration():
    """
    Create development environment key management configuration.

    Development configuration with relaxed security for ease of development.
    """
    print("🏗️ Setting up Development Key Management Configuration")

    # Development key directory
    dev_keys_dir = Path("./dev_keys")
    dev_keys_dir.mkdir(exist_ok=True)

    # Development security configuration
    dev_security_config = KeySecurityConfig(
        encryption_enabled=False,  # Disable encryption for easier debugging
        master_key_rotation_days=365,
        integrity_check_enabled=True,
        access_logging_enabled=True,
        audit_retention_days=90,  # Shorter retention for development
        max_keys_per_type=20,  # Higher limit for development
        key_backup_encryption=False
    )

    # Development rotation configuration
    dev_rotation_config = KeyRotationConfig(
        enabled=True,
        max_age_days=30,  # More frequent rotation in development
        max_usage_count=1000,
        rotation_window_hours=1,
        backup_old_keys=True,
        backup_retention_days=30,
        auto_rotation=False  # Manual rotation in development
    )

    # Development storage backends
    dev_storage_backends = {
        KeyStorageType.FILE: FileKeyStorage({
            "path": str(dev_keys_dir),
            "encryption": False
        }),
        KeyStorageType.ENVIRONMENT: EnvironmentKeyStorage({
            "prefix": "FEDZK_DEV_KEY_"
        })
    }

    # Create development key manager
    dev_key_manager = KeyManager(
        security_config=dev_security_config,
        rotation_config=dev_rotation_config,
        storage_backends=dev_storage_backends
    )

    print("✅ Development configuration created")
    print(f"   📁 Key directory: {dev_keys_dir}")
    print("   🔓 Encryption: Disabled")
    print("   🔄 Auto-rotation: Disabled")
    print("   📊 Audit retention: 90 days")

    return dev_key_manager

def create_staging_configuration():
    """
    Create staging environment key management configuration.

    Staging configuration with moderate security for testing.
    """
    print("\n🏭 Setting up Staging Key Management Configuration")

    # Staging key directory
    staging_keys_dir = Path("./staging_keys")
    staging_keys_dir.mkdir(exist_ok=True)

    # Staging security configuration
    staging_security_config = KeySecurityConfig(
        encryption_enabled=True,
        master_key_rotation_days=180,
        integrity_check_enabled=True,
        access_logging_enabled=True,
        audit_retention_days=180,
        max_keys_per_type=15,
        key_backup_encryption=True
    )

    # Staging rotation configuration
    staging_rotation_config = KeyRotationConfig(
        enabled=True,
        max_age_days=60,
        max_usage_count=5000,
        rotation_window_hours=12,
        backup_old_keys=True,
        backup_retention_days=90,
        auto_rotation=True
    )

    # Staging storage backends
    staging_storage_backends = {
        KeyStorageType.FILE: FileKeyStorage({
            "path": str(staging_keys_dir),
            "encryption": True
        })
    }

    # Create staging key manager
    staging_key_manager = KeyManager(
        security_config=staging_security_config,
        rotation_config=staging_rotation_config,
        storage_backends=staging_storage_backends
    )

    print("✅ Staging configuration created")
    print(f"   📁 Key directory: {staging_keys_dir}")
    print("   🔐 Encryption: Enabled")
    print("   🔄 Auto-rotation: Enabled")
    print("   📊 Audit retention: 180 days")

    return staging_key_manager

def create_production_configuration():
    """
    Create production environment key management configuration.

    Production configuration with maximum security and compliance features.
    """
    print("\n🏢 Setting up Production Key Management Configuration")

    # Production key directory
    prod_keys_dir = Path("./prod_keys")
    prod_keys_dir.mkdir(exist_ok=True)

    # Production security configuration
    prod_security_config = KeySecurityConfig(
        encryption_enabled=True,
        master_key_rotation_days=90,  # More frequent master key rotation
        integrity_check_enabled=True,
        access_logging_enabled=True,
        audit_retention_days=2555,  # 7 years for compliance
        max_keys_per_type=10,  # Stricter limits
        key_backup_encryption=True
    )

    # Production rotation configuration
    prod_rotation_config = KeyRotationConfig(
        enabled=True,
        max_age_days=90,  # SOX compliant rotation
        max_usage_count=10000,
        rotation_window_hours=24,
        backup_old_keys=True,
        backup_retention_days=365,  # 1 year backup retention
        auto_rotation=True
    )

    # Production storage backends
    prod_storage_backends = {
        KeyStorageType.FILE: FileKeyStorage({
            "path": str(prod_keys_dir),
            "encryption": True
        })
    }

    # Add Vault integration if available
    if os.getenv("VAULT_ADDR"):
        vault_config = {
            "vault_addr": os.getenv("VAULT_ADDR"),
            "token": os.getenv("VAULT_TOKEN"),
            "mount_point": "fedzk",
            "namespace": os.getenv("VAULT_NAMESPACE", "")
        }
        prod_storage_backends[KeyStorageType.VAULT] = VaultKeyStorage(vault_config)
        print("   🔐 Vault integration: Enabled")

    # Create production key manager
    prod_key_manager = KeyManager(
        security_config=prod_security_config,
        rotation_config=prod_rotation_config,
        storage_backends=prod_storage_backends
    )

    print("✅ Production configuration created")
    print(f"   📁 Key directory: {prod_keys_dir}")
    print("   🔐 Encryption: Enabled")
    print("   🔄 Auto-rotation: Enabled")
    print("   📊 Audit retention: 7 years (SOX compliant)")
    print("   🔒 Master key rotation: 90 days")

    return prod_key_manager

def setup_environment_keys(key_manager, environment):
    """
    Setup environment-specific keys for FEDzk operations.

    Args:
        key_manager: Configured key manager
        environment: Environment name (development, staging, production)
    """
    print(f"\n🔑 Setting up {environment.title()} Environment Keys")

    # Environment-specific key configurations
    env_configs = {
        "development": {
            "prefix": "fedzk_dev",
            "key_types": [KeyType.SYMMETRIC, KeyType.VERIFICATION_KEY],
            "rotation_policy": KeyRotationPolicy.TIME_BASED
        },
        "staging": {
            "prefix": "fedzk_staging",
            "key_types": [KeyType.SYMMETRIC, KeyType.VERIFICATION_KEY, KeyType.HMAC],
            "rotation_policy": KeyRotationPolicy.TIME_BASED
        },
        "production": {
            "prefix": "fedzk_prod",
            "key_types": [KeyType.SYMMETRIC, KeyType.VERIFICATION_KEY, KeyType.HMAC],
            "rotation_policy": KeyRotationPolicy.TIME_BASED
        }
    }

    config = env_configs.get(environment, env_configs["development"])

    # Setup standard FEDzk keys
    keys_created = []

    # Model encryption key
    model_key_id = f"{config['prefix']}_model_encryption"
    key_manager.store_key(
        model_key_id,
        os.urandom(32),  # 256-bit key
        KeyType.SYMMETRIC,
        rotation_policy=config["rotation_policy"],
        tags={"environment": environment, "purpose": "model_encryption"}
    )
    keys_created.append(model_key_id)

    # Proof verification key
    proof_key_id = f"{config['prefix']}_proof_verification"
    key_manager.store_key(
        proof_key_id,
        os.urandom(32),
        KeyType.VERIFICATION_KEY,
        rotation_policy=config["rotation_policy"],
        tags={"environment": environment, "purpose": "proof_verification"}
    )
    keys_created.append(proof_key_id)

    # Secure aggregation key
    agg_key_id = f"{config['prefix']}_aggregation"
    key_manager.store_key(
        agg_key_id,
        os.urandom(32),
        KeyType.SYMMETRIC,
        rotation_policy=config["rotation_policy"],
        tags={"environment": environment, "purpose": "secure_aggregation"}
    )
    keys_created.append(agg_key_id)

    # HMAC key for integrity
    hmac_key_id = f"{config['prefix']}_integrity_hmac"
    key_manager.store_key(
        hmac_key_id,
        os.urandom(32),
        KeyType.HMAC,
        rotation_policy=config["rotation_policy"],
        tags={"environment": environment, "purpose": "integrity_verification"}
    )
    keys_created.append(hmac_key_id)

    print(f"✅ Created {len(keys_created)} keys for {environment} environment:")
    for key_id in keys_created:
        print(f"   - {key_id}")

    return keys_created

def demonstrate_secure_configuration():
    """Demonstrate secure configuration best practices."""
    print("\n🔒 Demonstrating Secure Configuration Best Practices")

    # Create secure configuration
    secure_config = create_secure_key_manager()

    print("✅ Secure key manager created with:")
    print("   🔐 Encryption: Enabled")
    print("   ✅ Integrity checking: Enabled")
    print("   📝 Access logging: Enabled")
    print("   🔄 Auto-rotation: Enabled")
    print("   💾 Backup system: Enabled")

    # Demonstrate key operations
    test_key_id = "secure_demo_key"
    test_key_data = b"This is a secure demonstration key"

    # Store key securely
    success = secure_config.store_key(
        test_key_id,
        test_key_data,
        KeyType.SYMMETRIC,
        tags={"demo": "secure_config", "classification": "confidential"}
    )

    print(f"✅ Secure key storage: {'SUCCESS' if success else 'FAILED'}")

    # Retrieve with integrity verification
    retrieved_data, metadata = secure_config.retrieve_key(test_key_id)
    integrity_verified = metadata.tags.get("integrity_hash") is not None

    print(f"✅ Integrity verification: {'ENABLED' if integrity_verified else 'DISABLED'}")
    print(f"✅ Data integrity: {'VERIFIED' if retrieved_data == test_key_data else 'FAILED'}")

    # Show security metrics
    metrics = secure_config.get_security_metrics()
    print("📊 Security metrics:")
    print(f"   📂 Total keys: {metrics['total_keys']}")
    print(f"   📝 Access log entries: {metrics['access_log_entries']}")

    return secure_config

def create_environment_variables_template():
    """Create environment variables template for production deployment."""
    template = """
# FEDzk Production Environment Variables Template
# ================================================

# Key Management Configuration
export FEDZK_KEY_ENCRYPTION_ENABLED=true
export FEDZK_KEY_INTEGRITY_CHECK_ENABLED=true
export FEDZK_KEY_ACCESS_LOGGING_ENABLED=true
export FEDZK_KEY_AUTO_ROTATION_ENABLED=true

# Master Key Configuration
export FEDZK_MASTER_KEY_ROTATION_DAYS=90
export FEDZK_MASTER_KEY_PATH="./keys/master.key"

# Key Storage Configuration
export FEDZK_KEY_STORAGE_TYPE="file"
export FEDZK_KEY_STORAGE_PATH="./prod_keys"
export FEDZK_KEY_BACKUP_ENABLED=true
export FEDZK_KEY_BACKUP_RETENTION_DAYS=365

# Vault Integration (if using HashiCorp Vault)
export VAULT_ADDR="https://vault.example.com:8200"
export VAULT_TOKEN="hvs.your-vault-token-here"
export VAULT_NAMESPACE="fedzk/production"

# Key Rotation Configuration
export FEDZK_KEY_ROTATION_MAX_AGE_DAYS=90
export FEDZK_KEY_ROTATION_MAX_USAGE_COUNT=10000
export FEDZK_KEY_ROTATION_WINDOW_HOURS=24

# Security Monitoring
export FEDZK_SECURITY_AUDIT_RETENTION_DAYS=2555
export FEDZK_SECURITY_METRICS_ENABLED=true
export FEDZK_SECURITY_ALERT_EMAIL="security@example.com"

# Compliance Configuration
export FEDZK_COMPLIANCE_SOX_ENABLED=true
export FEDZK_COMPLIANCE_GDPR_ENABLED=true
export FEDZK_COMPLIANCE_LOG_ENCRYPTION=true

# Environment-Specific Keys
export FEDZK_PROD_MODEL_ENCRYPTION_KEY_ID="fedzk_prod_model_encryption"
export FEDZK_PROD_PROOF_VERIFICATION_KEY_ID="fedzk_prod_proof_verification"
export FEDZK_PROD_AGGREGATION_KEY_ID="fedzk_prod_aggregation"
export FEDZK_PROD_INTEGRITY_HMAC_KEY_ID="fedzk_prod_integrity_hmac"

# Logging Configuration
export FEDZK_LOG_LEVEL="INFO"
export FEDZK_SECURITY_LOG_PATH="./logs/security.log"
export FEDZK_AUDIT_LOG_PATH="./logs/audit.log"
"""

    template_path = Path("./production_env_template.sh")
    with open(template_path, 'w') as f:
        f.write(template.strip())

    print(f"\n📄 Environment variables template created: {template_path}")
    print("   Copy this template to set up your production environment")

def demonstrate_compliance_features():
    """Demonstrate compliance features for regulatory requirements."""
    print("\n📋 Demonstrating Compliance Features")

    # SOX Compliance Configuration
    sox_config = KeySecurityConfig(
        encryption_enabled=True,
        integrity_check_enabled=True,
        access_logging_enabled=True,
        audit_retention_days=2555,  # 7 years for SOX
        key_backup_encryption=True
    )

    sox_rotation = KeyRotationConfig(
        enabled=True,
        max_age_days=90,  # SOX quarterly rotation requirement
        backup_old_keys=True,
        backup_retention_days=365
    )

    print("✅ SOX Compliance Configuration:")
    print("   📅 Audit retention: 7 years")
    print("   🔄 Key rotation: 90 days")
    print("   💾 Encrypted backups: Enabled")
    print("   📝 Access logging: Enabled")

    # GDPR Compliance Configuration
    gdpr_config = KeySecurityConfig(
        encryption_enabled=True,
        integrity_check_enabled=True,
        access_logging_enabled=True,
        audit_retention_days=2555,
        key_backup_encryption=True
    )

    print("\n✅ GDPR Compliance Configuration:")
    print("   🔐 Data encryption: Enabled")
    print("   ✅ Integrity verification: Enabled")
    print("   📊 Audit trails: Enabled")
    print("   🗑️ Secure deletion: Supported")

    return sox_config, gdpr_config

def main():
    """Main configuration demonstration."""
    print("🏗️ FEDzk Production Key Management Configuration")
    print("=" * 60)
    print()
    print("This script demonstrates production-ready key management")
    print("configurations for different deployment environments.")
    print()

    # Create environment configurations
    dev_manager = create_development_configuration()
    staging_manager = create_staging_configuration()
    prod_manager = create_production_configuration()

    # Setup environment-specific keys
    dev_keys = setup_environment_keys(dev_manager, "development")
    staging_keys = setup_environment_keys(staging_manager, "staging")
    prod_keys = setup_environment_keys(prod_manager, "production")

    # Demonstrate secure configuration
    secure_manager = demonstrate_secure_configuration()

    # Create environment variables template
    create_environment_variables_template()

    # Demonstrate compliance features
    demonstrate_compliance_features()

    print("\n" + "=" * 60)
    print("CONFIGURATION COMPLETE")
    print("=" * 60)
    print()
    print("🎯 Summary:")
    print(f"   🏗️ Development keys: {len(dev_keys)}")
    print(f"   🏭 Staging keys: {len(staging_keys)}")
    print(f"   🏢 Production keys: {len(prod_keys)}")
    print("   🔒 Secure configuration: Implemented")
    print("   📄 Environment template: Created")
    print("   📋 Compliance features: Configured")
    print()
    print("🚀 Ready for production deployment!")

if __name__ == "__main__":
    main()

