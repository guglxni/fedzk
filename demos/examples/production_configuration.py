#!/usr/bin/env python3
"""
FEDzk Production Configuration Example
=====================================

This example demonstrates how to properly configure FEDzk for production deployment
with real cryptographic operations. It shows secure configuration patterns,
environment management, and production-ready parameter selection.

Real-World Configuration:
- Secure environment variable management
- TLS certificate configuration
- API key management
- Production logging and monitoring
- Resource limits and scaling
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional
import hashlib
import secrets
from datetime import datetime

# FEDzk imports
from fedzk.config import FedZKConfig
from fedzk.prover.zk_validator import ZKValidator
from fedzk.mpc.client import MPCClient
from fedzk.coordinator import SecureCoordinator

# Setup production logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/var/log/fedzk/production.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ProductionConfigurator:
    """Production configuration manager for FEDzk deployments."""

    def __init__(self, environment: str = "production"):
        self.environment = environment
        self.config_dir = Path("/etc/fedzk")
        self.secrets_dir = Path("/etc/fedzk/secrets")
        self.logs_dir = Path("/var/log/fedzk")
        self.data_dir = Path("/var/lib/fedzk")

        # Create required directories
        for directory in [self.config_dir, self.secrets_dir, self.logs_dir, self.data_dir]:
            directory.mkdir(parents=True, exist_ok=True)

        logger.info(f"Production configurator initialized for {environment} environment")

    def generate_secure_api_keys(self, num_keys: int = 5) -> Dict[str, str]:
        """Generate cryptographically secure API keys."""
        logger.info(f"Generating {num_keys} secure API keys")

        keys = {}
        for i in range(num_keys):
            # Generate 32-character URL-safe key
            raw_key = secrets.token_urlsafe(32)
            key_hash = hashlib.sha256(raw_key.encode()).hexdigest()

            keys[f"client_{i+1:03d}"] = {
                "raw_key": raw_key,
                "hash": key_hash,
                "created": datetime.now().isoformat(),
                "environment": self.environment
            }

        # Save keys securely
        keys_file = self.secrets_dir / "api_keys.json"
        with open(keys_file, 'w') as f:
            json.dump(keys, f, indent=2)

        # Set restrictive permissions
        keys_file.chmod(0o600)

        logger.info(f"API keys saved to {keys_file}")
        return keys

    def setup_tls_certificates(self) -> Dict[str, str]:
        """Configure TLS certificates for secure communication."""
        logger.info("Setting up TLS certificates")

        cert_config = {
            "certificate_file": "/etc/ssl/certs/fedzk.crt",
            "key_file": "/etc/ssl/private/fedzk.key",
            "ca_certificate": "/etc/ssl/certs/ca.crt",
            "tls_version": "TLSv1.3",
            "cipher_suites": [
                "ECDHE-RSA-AES256-GCM-SHA384",
                "ECDHE-RSA-AES128-GCM-SHA256"
            ]
        }

        # In production, certificates should be obtained from a proper CA
        # This is just a configuration template
        config_file = self.config_dir / "tls_config.json"
        with open(config_file, 'w') as f:
            json.dump(cert_config, f, indent=2)

        logger.info(f"TLS configuration saved to {config_file}")
        return cert_config

    def create_production_environment(self) -> Dict[str, str]:
        """Create production environment configuration."""
        logger.info("Creating production environment configuration")

        env_config = {
            # Core FEDzk settings
            "ENVIRONMENT": self.environment,
            "LOG_LEVEL": "INFO",
            "FEDZK_CONFIG_FILE": str(self.config_dir / "fedzk_config.json"),

            # Network configuration
            "FEDZK_COORDINATOR_HOST": "coordinator.production.company.com",
            "FEDZK_COORDINATOR_PORT": "8443",
            "FEDZK_MPC_SERVER_URL": "https://mpc.production.company.com:9000",

            # Security settings
            "FEDZK_TLS_ENABLED": "true",
            "FEDZK_TLS_CERT_FILE": "/etc/ssl/certs/fedzk.crt",
            "FEDZK_TLS_KEY_FILE": "/etc/ssl/private/fedzk.key",
            "FEDZK_API_KEYS_FILE": str(self.secrets_dir / "api_keys.json"),

            # ZK configuration
            "FEDZK_ZK_CIRCUITS_PATH": "/opt/fedzk/circuits",
            "FEDZK_VERIFICATION_KEYS_PATH": "/opt/fedzk/keys",
            "FEDZK_ZK_PROOF_CACHE": "/var/cache/fedzk/proofs",

            # Resource limits
            "FEDZK_MAX_MEMORY_MB": "4096",
            "FEDZK_MAX_CPU_CORES": "4",
            "FEDZK_REQUEST_TIMEOUT": "300",

            # Monitoring
            "FEDZK_METRICS_ENABLED": "true",
            "FEDZK_METRICS_PORT": "9090",
            "FEDZK_HEALTH_CHECK_INTERVAL": "30",

            # Compliance
            "FEDZK_AUDIT_LOG_ENABLED": "true",
            "FEDZK_AUDIT_LOG_PATH": "/var/log/fedzk/audit.log",
            "FEDZK_COMPLIANCE_LEVEL": "enterprise"
        }

        # Save environment file
        env_file = self.config_dir / f"{self.environment}.env"
        with open(env_file, 'w') as f:
            for key, value in env_config.items():
                f.write(f"{key}={value}\n")

        logger.info(f"Environment configuration saved to {env_file}")
        return env_config

    def setup_monitoring(self) -> Dict[str, Any]:
        """Configure production monitoring and alerting."""
        logger.info("Setting up production monitoring")

        monitoring_config = {
            "metrics": {
                "enabled": True,
                "port": 9090,
                "path": "/metrics",
                "collectors": [
                    "proof_generation_time",
                    "proof_verification_time",
                    "client_connections",
                    "memory_usage",
                    "cpu_usage",
                    "network_io"
                ]
            },
            "health_checks": {
                "enabled": True,
                "endpoint": "/health",
                "interval": 30,
                "timeout": 10,
                "checks": [
                    "zk_toolchain",
                    "database_connection",
                    "external_services",
                    "resource_limits"
                ]
            },
            "alerting": {
                "enabled": True,
                "rules": [
                    {
                        "name": "high_memory_usage",
                        "condition": "memory_usage > 80%",
                        "severity": "warning"
                    },
                    {
                        "name": "proof_generation_failure",
                        "condition": "proof_failure_rate > 5%",
                        "severity": "critical"
                    },
                    {
                        "name": "network_connectivity",
                        "condition": "coordinator_unreachable > 5min",
                        "severity": "critical"
                    }
                ]
            },
            "logging": {
                "level": "INFO",
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                "handlers": [
                    {
                        "type": "file",
                        "path": "/var/log/fedzk/production.log",
                        "max_size": "100MB",
                        "backup_count": 5
                    },
                    {
                        "type": "syslog",
                        "facility": "local0"
                    }
                ]
            }
        }

        config_file = self.config_dir / "monitoring.json"
        with open(config_file, 'w') as f:
            json.dump(monitoring_config, f, indent=2)

        logger.info(f"Monitoring configuration saved to {config_file}")
        return monitoring_config

    def create_deployment_manifests(self) -> Dict[str, str]:
        """Create deployment manifests for different environments."""
        logger.info("Creating deployment manifests")

        manifests = {}

        # Docker Compose for development/testing
        docker_compose = {
            "version": "3.8",
            "services": {
                "mpc-server": {
                    "image": "fedzk/mpc-server:latest",
                    "ports": ["9000:9000"],
                    "environment": [
                        "ENVIRONMENT=production",
                        "FEDZK_TLS_ENABLED=true"
                    ],
                    "volumes": [
                        "./keys:/app/keys:ro",
                        "./circuits:/app/circuits:ro"
                    ],
                    "restart": "unless-stopped"
                },
                "coordinator": {
                    "image": "fedzk/coordinator:latest",
                    "ports": ["8443:8443"],
                    "environment": [
                        "ENVIRONMENT=production",
                        "FEDZK_TLS_ENABLED=true"
                    ],
                    "volumes": [
                        "./keys:/app/keys:ro",
                        "./data:/app/data"
                    ],
                    "restart": "unless-stopped",
                    "depends_on": ["mpc-server"]
                }
            }
        }

        manifests["docker-compose.yml"] = json.dumps(docker_compose, indent=2)

        # Kubernetes deployment
        k8s_deployment = {
            "apiVersion": "apps/v1",
            "kind": "Deployment",
            "metadata": {
                "name": "fedzk-production",
                "namespace": "fedzk"
            },
            "spec": {
                "replicas": 3,
                "selector": {
                    "matchLabels": {"app": "fedzk"}
                },
                "template": {
                    "metadata": {
                        "labels": {"app": "fedzk"}
                    },
                    "spec": {
                        "containers": [{
                            "name": "fedzk",
                            "image": "fedzk/fedzk:latest",
                            "ports": [
                                {"containerPort": 9000, "name": "mpc"},
                                {"containerPort": 8443, "name": "coordinator"}
                            ],
                            "envFrom": [{"configMapRef": {"name": "fedzk-config"}}],
                            "resources": {
                                "requests": {"memory": "2Gi", "cpu": "1000m"},
                                "limits": {"memory": "4Gi", "cpu": "2000m"}
                            },
                            "livenessProbe": {
                                "httpGet": {"path": "/health", "port": 9000},
                                "initialDelaySeconds": 30,
                                "periodSeconds": 10
                            }
                        }]
                    }
                }
            }
        }

        manifests["k8s-deployment.json"] = json.dumps(k8s_deployment, indent=2)

        # Save manifests
        for filename, content in manifests.items():
            manifest_file = self.config_dir / filename
            with open(manifest_file, 'w') as f:
                f.write(content)
            logger.info(f"Deployment manifest saved: {manifest_file}")

        return manifests

    def validate_configuration(self) -> Dict[str, Any]:
        """Validate complete production configuration."""
        logger.info("Validating production configuration")

        validation_results = {
            "timestamp": datetime.now().isoformat(),
            "environment": self.environment,
            "checks": {}
        }

        # Check required files
        required_files = [
            self.config_dir / f"{self.environment}.env",
            self.secrets_dir / "api_keys.json",
            self.config_dir / "tls_config.json"
        ]

        for file_path in required_files:
            exists = file_path.exists()
            validation_results["checks"][f"file_{file_path.name}"] = {
                "exists": exists,
                "path": str(file_path),
                "status": "pass" if exists else "fail"
            }

        # Check required environment variables
        required_env_vars = [
            "FEDZK_COORDINATOR_HOST",
            "FEDZK_MPC_SERVER_URL",
            "FEDZK_TLS_CERT_FILE"
        ]

        for env_var in required_env_vars:
            exists = env_var in os.environ
            validation_results["checks"][f"env_{env_var}"] = {
                "exists": exists,
                "value": os.environ.get(env_var, "NOT_SET"),
                "status": "pass" if exists else "fail"
            }

        # Check TLS certificates
        tls_cert = os.environ.get("FEDZK_TLS_CERT_FILE", "/etc/ssl/certs/fedzk.crt")
        tls_key = os.environ.get("FEDZK_TLS_KEY_FILE", "/etc/ssl/private/fedzk.key")

        for cert_file in [tls_cert, tls_key]:
            exists = Path(cert_file).exists()
            validation_results["checks"][f"tls_{Path(cert_file).name}"] = {
                "exists": exists,
                "path": cert_file,
                "status": "pass" if exists else "fail"
            }

        # Calculate overall status
        all_checks = validation_results["checks"]
        passed_checks = sum(1 for check in all_checks.values() if check["status"] == "pass")
        total_checks = len(all_checks)

        validation_results["summary"] = {
            "total_checks": total_checks,
            "passed_checks": passed_checks,
            "failed_checks": total_checks - passed_checks,
            "success_rate": (passed_checks / total_checks) * 100 if total_checks > 0 else 0,
            "overall_status": "valid" if passed_checks == total_checks else "invalid"
        }

        # Save validation results
        validation_file = self.logs_dir / f"config_validation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(validation_file, 'w') as f:
            json.dump(validation_results, f, indent=2)

        logger.info(f"Configuration validation completed: {passed_checks}/{total_checks} checks passed")
        return validation_results

def setup_production_environment():
    """Complete production environment setup."""
    logger.info("🚀 Setting up FEDzk Production Environment")
    logger.info("=" * 60)

    # Initialize production configurator
    config = ProductionConfigurator("production")

    # Step 1: Generate secure API keys
    logger.info("\n🔐 Step 1: Generating secure API keys...")
    api_keys = config.generate_secure_api_keys(10)

    # Step 2: Setup TLS certificates
    logger.info("\n🔒 Step 2: Setting up TLS certificates...")
    tls_config = config.setup_tls_certificates()

    # Step 3: Create production environment
    logger.info("\n⚙️ Step 3: Creating production environment...")
    env_config = config.create_production_environment()

    # Step 4: Setup monitoring
    logger.info("\n📊 Step 4: Setting up monitoring...")
    monitoring_config = config.setup_monitoring()

    # Step 5: Create deployment manifests
    logger.info("\n🐳 Step 5: Creating deployment manifests...")
    manifests = config.create_deployment_manifests()

    # Step 6: Validate configuration
    logger.info("\n✅ Step 6: Validating configuration...")
    validation = config.validate_configuration()

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("🏆 PRODUCTION ENVIRONMENT SETUP COMPLETED")
    logger.info("=" * 60)

    logger.info("📋 Configuration Summary:")
    logger.info(f"   Environment: {config.environment}")
    logger.info(f"   API Keys Generated: {len(api_keys)}")
    logger.info(f"   TLS Certificates: Configured")
    logger.info(f"   Environment Variables: {len(env_config)}")
    logger.info(f"   Monitoring: {'Enabled' if monitoring_config['metrics']['enabled'] else 'Disabled'}")
    logger.info(f"   Deployment Manifests: {len(manifests)}")

    validation_summary = validation.get("summary", {})
    logger.info(f"   Configuration Validation: {validation_summary.get('passed_checks', 0)}/{validation_summary.get('total_checks', 0)} checks passed")

    if validation_summary.get("overall_status") == "valid":
        logger.info("✅ Production environment is ready for deployment!")
        return True
    else:
        logger.warning("⚠️ Production environment has configuration issues")
        logger.info("   Please review the validation results above")
        return False

def demonstrate_production_fedzk_usage():
    """Demonstrate production FEDzk usage with proper configuration."""
    logger.info("\n🔧 Demonstrating Production FEDzk Usage")
    logger.info("-" * 50)

    try:
        # Load production configuration
        config = FedZKConfig(
            environment="production",
            api_keys="prod_key_12345678901234567890123456789012",
            mpc_server_url="https://mpc.production.company.com:9000",
            coordinator_host="coordinator.production.company.com",
            coordinator_port=8443,
            tls_enabled=True
        )

        logger.info("✅ Production configuration loaded")

        # Validate ZK toolchain
        zk_validator = ZKValidator()
        toolchain_status = zk_validator.validate_toolchain()

        if toolchain_status["overall_status"] == "passed":
            logger.info("✅ ZK toolchain validation passed")
        else:
            logger.error("❌ ZK toolchain validation failed")
            return False

        # Initialize production clients
        mpc_client = MPCClient(
            server_url=config.mpc_server_url,
            api_key=config.api_keys.split(",")[0],
            tls_verify=True,
            timeout=60
        )

        coordinator = SecureCoordinator(
            f"https://{config.coordinator_host}:{config.coordinator_port}",
            tls_cert="/etc/ssl/certs/ca.crt"
        )

        logger.info("✅ Production clients initialized")

        # Demonstrate configuration usage
        logger.info("📊 Production Configuration Details:")
        logger.info(f"   Environment: {config.environment}")
        logger.info(f"   MPC Server: {config.mpc_server_url}")
        logger.info(f"   Coordinator: {config.coordinator_host}:{config.coordinator_port}")
        logger.info(f"   TLS Enabled: {config.tls_enabled}")
        logger.info(f"   ZK Toolchain: {toolchain_status['overall_status']}")

        return True

    except Exception as e:
        logger.error(f"❌ Production demonstration failed: {e}")
        return False

def main():
    """Main entry point."""
    try:
        # Setup production environment
        setup_success = setup_production_environment()

        # Demonstrate usage
        demo_success = demonstrate_production_fedzk_usage()

        if setup_success and demo_success:
            logger.info("\n🎉 FEDzk Production Configuration Complete!")
            exit(0)
        else:
            logger.warning("\n⚠️ FEDzk Production Configuration has issues")
            exit(1)

    except KeyboardInterrupt:
        logger.info("\n🛑 Configuration interrupted by user")
        exit(1)
    except Exception as e:
        logger.error(f"\n💥 Configuration failed: {e}")
        exit(1)

if __name__ == "__main__":
    main()

