# Functional Source License 1.1 with Apache-2.0 Future Grant (FSL-1.1-Apache-2.0)
# Copyright (c) 2025 Aaryan Guglani and FEDzk Contributors
# Licensed under FSL-1.1-Apache-2.0. See LICENSE for details.

"""
Production-grade configuration management for FEDzk.
Provides type-safe configuration with validation and environment variable support.
"""

import os
import logging
from typing import Optional, List, Dict, Any
from pathlib import Path
from pydantic import Field, validator
from pydantic_settings import BaseSettings
from enum import Enum

class Environment(str, Enum):
    """Supported deployment environments."""
    DEVELOPMENT = "development"
    TESTING = "testing"
    STAGING = "staging"
    PRODUCTION = "production"

class LogLevel(str, Enum):
    """Supported log levels."""
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

class FEDzkConfig(BaseSettings):
    """Production-grade configuration for FEDzk."""
    
    # Environment settings
    environment: Environment = Field(default=Environment.DEVELOPMENT, env="ENVIRONMENT")
    debug: bool = Field(default=False, env="DEBUG")
    log_level: LogLevel = Field(default=LogLevel.INFO, env="LOG_LEVEL")
    
    # Server settings
    host: str = Field(default="0.0.0.0", env="HOST")
    port: int = Field(default=8000, env="PORT")
    workers: int = Field(default=1, env="WORKERS")
    
    # Security settings
    api_keys: str = Field(default="", env="MPC_API_KEYS")
    allowed_origins: str = Field(default="*", env="ALLOWED_ORIGINS")
    trusted_hosts: str = Field(default="*", env="TRUSTED_HOSTS")
    rate_limit_requests: int = Field(default=100, env="RATE_LIMIT_REQUESTS")
    rate_limit_window: int = Field(default=60, env="RATE_LIMIT_WINDOW")
    max_failed_attempts: int = Field(default=10, env="MAX_FAILED_ATTEMPTS")
    
    # ZK Circuit paths
    zk_circuits_dir: str = Field(default="", env="ZK_CIRCUITS_DIR")
    std_wasm_path: str = Field(default="", env="MPC_STD_WASM_PATH")
    std_zkey_path: str = Field(default="", env="MPC_STD_ZKEY_PATH")
    std_vkey_path: str = Field(default="", env="MPC_STD_VER_KEY_PATH")
    sec_wasm_path: str = Field(default="", env="MPC_SEC_WASM_PATH")
    sec_zkey_path: str = Field(default="", env="MPC_SEC_ZKEY_PATH")
    sec_vkey_path: str = Field(default="", env="MPC_SEC_VER_KEY_PATH")
    
    # Performance settings
    enable_gpu: bool = Field(default=False, env="ENABLE_GPU")
    max_gradient_size: int = Field(default=10000, env="MAX_GRADIENT_SIZE")
    default_chunk_size: int = Field(default=2, env="DEFAULT_CHUNK_SIZE")
    request_timeout: int = Field(default=300, env="REQUEST_TIMEOUT")  # 5 minutes
    

    
    # Database settings (for future use)
    database_url: Optional[str] = Field(default=None, env="DATABASE_URL")
    redis_url: Optional[str] = Field(default=None, env="REDIS_URL")
    
    # Monitoring settings
    enable_metrics: bool = Field(default=True, env="ENABLE_METRICS")
    metrics_port: int = Field(default=9090, env="METRICS_PORT")
    health_check_interval: int = Field(default=30, env="HEALTH_CHECK_INTERVAL")
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
    
    @validator("api_keys")
    def validate_api_keys(cls, v, values):
        """Validate API keys."""
        # Allow empty API keys in development/testing, but warn
        environment = values.get("environment", Environment.DEVELOPMENT)

        if not v:
            if environment == Environment.PRODUCTION:
                raise ValueError("API keys are required in production environment")
            # For non-production environments, allow empty keys
            return v

        if v:
            keys = v.split(",")
            for key in keys:
                if len(key.strip()) < 32:
                    raise ValueError("All API keys must be at least 32 characters long")

        return v
    
    @validator("port")
    def validate_port(cls, v):
        """Validate port number."""
        if v < 1 or v > 65535:
            raise ValueError("Port must be between 1 and 65535")
        return v
    
    @validator("workers")
    def validate_workers(cls, v):
        """Validate worker count."""
        if v < 1:
            raise ValueError("Worker count must be at least 1")
        return v
    
    def get_api_key_list(self) -> List[str]:
        """Get list of API keys."""
        if not self.api_keys:
            return []
        return [key.strip() for key in self.api_keys.split(",") if key.strip()]
    
    def get_allowed_origins_list(self) -> List[str]:
        """Get list of allowed origins."""
        if self.allowed_origins == "*":
            return ["*"]
        return [origin.strip() for origin in self.allowed_origins.split(",") if origin.strip()]
    
    def get_trusted_hosts_list(self) -> List[str]:
        """Get list of trusted hosts."""
        if self.trusted_hosts == "*":
            return ["*"]
        return [host.strip() for host in self.trusted_hosts.split(",") if host.strip()]
    
    def get_zk_circuit_paths(self) -> Dict[str, str]:
        """Get ZK circuit file paths."""
        # Auto-detect paths if not explicitly set
        if not self.zk_circuits_dir:
            project_root = Path(__file__).resolve().parent.parent.parent.parent
            circuits_dir = project_root / "src" / "fedzk" / "zk" / "circuits"
        else:
            circuits_dir = Path(self.zk_circuits_dir)
        
        return {
            "std_wasm": self.std_wasm_path or str(circuits_dir / "build" / "model_update.wasm"),
            "std_zkey": self.std_zkey_path or str(circuits_dir / "proving_key.zkey"),
            "std_vkey": self.std_vkey_path or str(circuits_dir / "verification_key.json"),
            "sec_wasm": self.sec_wasm_path or str(circuits_dir / "build" / "model_update_secure.wasm"),
            "sec_zkey": self.sec_zkey_path or str(circuits_dir / "proving_key_secure.zkey"),
            "sec_vkey": self.sec_vkey_path or str(circuits_dir / "verification_key_secure.json"),
        }
    
    def setup_logging(self):
        """Setup production-grade logging."""
        log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        
        if self.environment == Environment.PRODUCTION:
            # JSON logging for production
            import json
            import sys
            
            class JSONFormatter(logging.Formatter):
                def format(self, record):
                    return json.dumps({
                        "timestamp": self.formatTime(record),
                        "level": record.levelname,
                        "logger": record.name,
                        "message": record.getMessage(),
                        "module": record.module,
                        "funcName": record.funcName,
                        "lineno": record.lineno
                    })
            
            handler = logging.StreamHandler(sys.stdout)
            handler.setFormatter(JSONFormatter())
        else:
            # Human-readable logging for development
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter(log_format))
        
        # Configure root logger
        logging.basicConfig(
            level=getattr(logging, self.log_level.upper()),
            handlers=[handler],
            force=True
        )
        
        # Set specific logger levels
        logging.getLogger("uvicorn").setLevel(logging.INFO)
        logging.getLogger("fastapi").setLevel(logging.INFO)
    
    def validate_setup(self) -> List[str]:
        """Validate the current configuration setup."""
        warnings = []

        # Always check ZK circuit files (no test mode bypass)
        circuit_paths = self.get_zk_circuit_paths()
        for name, path in circuit_paths.items():
            if not Path(path).exists():
                warnings.append(f"Missing ZK circuit file: {name} at {path}")

        # Check security settings for production
        if self.environment == Environment.PRODUCTION:
            if not self.api_keys:
                warnings.append("No API keys configured for production environment")

            if self.allowed_origins == "*":
                warnings.append("CORS allows all origins in production")

            if self.debug:
                warnings.append("Debug mode is enabled in production")

        return warnings

# Production configuration templates
PRODUCTION_CONFIG_TEMPLATE = {
    "environment": "production",
    "debug": False,
    "log_level": "warning",
    "host": "0.0.0.0",
    "port": 8443,
    "workers": 8,
    "api_keys": "prod_key_12345678901234567890123456789012,prod_key_2_12345678901234567890123456789012",  # Default production keys
    "allowed_origins": "https://your-domain.com",
    "trusted_hosts": "api.your-domain.com",
    "rate_limit_requests": 1000,
    "rate_limit_window": 3600,
    "max_failed_attempts": 5,
    "enable_gpu": True,
    "max_gradient_size": 100000,
    "default_chunk_size": 8,
    "request_timeout": 1800,  # 30 minutes
    "enable_metrics": True,
    "metrics_port": 9090,
    "health_check_interval": 30,
    "zk_circuits_dir": "/opt/fedzk/circuits",
}

DEVELOPMENT_CONFIG_TEMPLATE = {
    "environment": "development",
    "debug": True,
    "log_level": "debug",
    "host": "localhost",
    "port": 8000,
    "workers": 1,
    "api_keys": "",  # Not required in development
    "allowed_origins": "*",
    "trusted_hosts": "*",
    "rate_limit_requests": 10000,
    "rate_limit_window": 60,
    "max_failed_attempts": 10,
    "enable_gpu": False,
    "max_gradient_size": 10000,
    "default_chunk_size": 2,
    "request_timeout": 300,
    "enable_metrics": False,
    "metrics_port": 9090,
    "health_check_interval": 60,
    "zk_circuits_dir": "",  # Auto-detect
}

STAGING_CONFIG_TEMPLATE = {
    "environment": "staging",
    "debug": False,
    "log_level": "info",
    "host": "0.0.0.0",
    "port": 8080,
    "workers": 4,
    "api_keys": "",  # Must be set via environment variable
    "allowed_origins": "https://staging.your-domain.com",
    "trusted_hosts": "api.staging.your-domain.com",
    "rate_limit_requests": 5000,
    "rate_limit_window": 1800,
    "max_failed_attempts": 5,
    "enable_gpu": True,
    "max_gradient_size": 50000,
    "default_chunk_size": 4,
    "request_timeout": 900,  # 15 minutes
    "enable_metrics": True,
    "metrics_port": 9090,
    "health_check_interval": 30,
    "zk_circuits_dir": "/opt/fedzk/circuits",
}

def create_config_from_template(template_name: str) -> FEDzkConfig:
    """
    Create a FEDzkConfig instance from a predefined template.

    Args:
        template_name: Name of the template ("production", "development", "staging")

    Returns:
        FEDzkConfig: Configuration instance with template values

    Raises:
        ValueError: If template name is invalid
    """
    templates = {
        "production": PRODUCTION_CONFIG_TEMPLATE,
        "development": DEVELOPMENT_CONFIG_TEMPLATE,
        "staging": STAGING_CONFIG_TEMPLATE,
    }

    if template_name not in templates:
        raise ValueError(f"Unknown template: {template_name}. Available: {list(templates.keys())}")

    template = templates[template_name]

    # Create config with template values, allowing environment variables to override
    return FEDzkConfig(**template)

def generate_production_env_file(output_path: str = ".env.production"):
    """
    Generate a production environment file with all required settings.

    Args:
        output_path: Path where to write the .env file
    """
    env_content = """# FEDzk Production Environment Configuration
# Copy this file to .env and customize the values

# Environment
ENVIRONMENT=production
DEBUG=false
LOG_LEVEL=warning

# Server
HOST=0.0.0.0
PORT=8443
WORKERS=8

# Security (REQUIRED - Generate strong random keys)
MPC_API_KEYS=key1_$(openssl rand -hex 32),key2_$(openssl rand -hex 32)
ALLOWED_ORIGINS=https://your-production-domain.com
TRUSTED_HOSTS=api.your-production-domain.com

# Rate Limiting
RATE_LIMIT_REQUESTS=1000
RATE_LIMIT_WINDOW=3600
MAX_FAILED_ATTEMPTS=5

# Performance
ENABLE_GPU=true
MAX_GRADIENT_SIZE=100000
DEFAULT_CHUNK_SIZE=8
REQUEST_TIMEOUT=1800

# Monitoring
ENABLE_METRICS=true
METRICS_PORT=9090
HEALTH_CHECK_INTERVAL=30

# ZK Circuit Paths (update with your actual paths)
ZK_CIRCUITS_DIR=/opt/fedzk/circuits
MPC_STD_WASM_PATH=/opt/fedzk/circuits/build/model_update.wasm
MPC_STD_ZKEY_PATH=/opt/fedzk/circuits/proving_key.zkey
MPC_STD_VER_KEY_PATH=/opt/fedzk/circuits/verification_key.json
MPC_SEC_WASM_PATH=/opt/fedzk/circuits/build/model_update_secure.wasm
MPC_SEC_ZKEY_PATH=/opt/fedzk/circuits/proving_key_secure.zkey
MPC_SEC_VER_KEY_PATH=/opt/fedzk/circuits/verification_key_secure.json

# Database (optional - for future use)
# DATABASE_URL=postgresql://user:password@localhost/fedzk
# REDIS_URL=redis://localhost:6379
"""

    with open(output_path, "w") as f:
        f.write(env_content)

    print(f"✅ Production environment file generated: {output_path}")
    print("   📝 Remember to:")
    print("      1. Generate strong API keys using: openssl rand -hex 32")
    print("      2. Update domain names to match your deployment")
    print("      3. Set correct ZK circuit paths")
    print("      4. Review and adjust performance settings")

def generate_development_env_file(output_path: str = ".env.development"):
    """
    Generate a development environment file with permissive settings.

    Args:
        output_path: Path where to write the .env file
    """
    env_content = """# FEDzk Development Environment Configuration
# Safe defaults for local development

# Environment
ENVIRONMENT=development
DEBUG=true
LOG_LEVEL=debug

# Server
HOST=localhost
PORT=8000
WORKERS=1

# Security (optional in development)
MPC_API_KEYS=
ALLOWED_ORIGINS=*
TRUSTED_HOSTS=*

# Rate Limiting (permissive)
RATE_LIMIT_REQUESTS=10000
RATE_LIMIT_WINDOW=60
MAX_FAILED_ATTEMPTS=10

# Performance (conservative)
ENABLE_GPU=false
MAX_GRADIENT_SIZE=10000
DEFAULT_CHUNK_SIZE=2
REQUEST_TIMEOUT=300

# Monitoring (disabled)
ENABLE_METRICS=false
METRICS_PORT=9090
HEALTH_CHECK_INTERVAL=60

# ZK Circuit Paths (auto-detect)
ZK_CIRCUITS_DIR=
MPC_STD_WASM_PATH=
MPC_STD_ZKEY_PATH=
MPC_STD_VER_KEY_PATH=
MPC_SEC_WASM_PATH=
MPC_SEC_ZKEY_PATH=
MPC_SEC_VER_KEY_PATH=
"""

    with open(output_path, "w") as f:
        f.write(env_content)

    print(f"✅ Development environment file generated: {output_path}")

def validate_production_readiness() -> Dict[str, Any]:
    """
    Comprehensive validation for production readiness.

    Returns:
        Dict with validation results and recommendations
    """
    results = {
        "ready": True,
        "critical_issues": [],
        "warnings": [],
        "recommendations": [],
        "security_score": 100
    }

    config = FEDzkConfig()

    # Critical production checks
    if config.environment != Environment.PRODUCTION:
        results["critical_issues"].append("Not running in production environment")
        results["ready"] = False
        results["security_score"] -= 50

    if not config.api_keys:
        results["critical_issues"].append("No API keys configured")
        results["ready"] = False
        results["security_score"] -= 30

    if config.debug:
        results["critical_issues"].append("Debug mode enabled in production")
        results["ready"] = False
        results["security_score"] -= 20

    if config.allowed_origins == "*":
        results["critical_issues"].append("CORS allows all origins")
        results["ready"] = False
        results["security_score"] -= 15

    # Check ZK circuit files
    circuit_paths = config.get_zk_circuit_paths()
    missing_circuits = []
    for name, path in circuit_paths.items():
        if not Path(path).exists():
            missing_circuits.append(name)

    if missing_circuits:
        results["critical_issues"].append(f"Missing ZK circuits: {', '.join(missing_circuits)}")
        results["ready"] = False
        results["security_score"] -= 25

    # Warnings and recommendations
    if config.workers < 4:
        results["warnings"].append("Low worker count for production load")

    if config.max_gradient_size < 50000:
        results["warnings"].append("Low max gradient size for production")

    if not config.enable_gpu:
        results["warnings"].append("GPU not enabled - consider enabling for performance")

    if config.log_level == LogLevel.DEBUG:
        results["warnings"].append("Debug logging level in production")
        results["security_score"] -= 5

    # Recommendations
    if config.port < 1024:
        results["recommendations"].append("Consider using port >= 1024 for production")

    if config.request_timeout < 900:
        results["recommendations"].append("Consider longer request timeout for large proofs")

    results["security_score"] = max(0, results["security_score"])

    return results

# Global configuration instance
config = FEDzkConfig()

def get_config() -> FEDzkConfig:
    """Get the global configuration instance."""
    return config

def reload_config() -> FEDzkConfig:
    """Reload configuration from environment variables."""
    global config
    config = FEDzkConfig()
    return config
