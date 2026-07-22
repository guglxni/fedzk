#!/usr/bin/env python3
"""
Configuration Validation System
==============================

Advanced configuration validation with type checking, constraint validation,
and custom validation rules for FEDzk.
"""

import re
import ipaddress
import os
from pathlib import Path
from typing import Dict, Any, List, Optional, Callable, Union, Pattern
from dataclasses import dataclass
from enum import Enum
import logging


class ValidationSeverity(Enum):
    """Validation severity levels."""
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


@dataclass
class ValidationResult:
    """Result of a validation check."""
    field: str
    severity: ValidationSeverity
    message: str
    suggestion: Optional[str] = None
    value: Any = None


@dataclass
class ValidationRule:
    """Configuration validation rule."""
    name: str
    description: str
    field_type: type
    required: bool = False
    min_value: Optional[Union[int, float]] = None
    max_value: Optional[Union[int, float]] = None
    min_length: Optional[int] = None
    max_length: Optional[int] = None
    allowed_values: Optional[List[Any]] = None
    pattern: Optional[Union[str, Pattern]] = None
    custom_validator: Optional[Callable] = None
    depends_on: Optional[List[str]] = None
    severity: ValidationSeverity = ValidationSeverity.ERROR


class ConfigValidator:
    """Advanced configuration validator for FEDzk."""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.validation_rules = self._setup_validation_rules()

    def _setup_validation_rules(self) -> Dict[str, ValidationRule]:
        """Setup comprehensive validation rules for all configuration fields."""
        return {
            # Application Configuration
            'app_name': ValidationRule(
                name='app_name',
                description='Application name validation',
                field_type=str,
                required=True,
                min_length=1,
                max_length=50,
                pattern=r'^[a-zA-Z0-9_-]+$'
            ),

            'app_version': ValidationRule(
                name='app_version',
                description='Application version validation',
                field_type=str,
                required=True,
                pattern=r'^\d+\.\d+\.\d+$'
            ),

            'environment': ValidationRule(
                name='environment',
                description='Environment type validation',
                field_type=str,  # Will be handled specially for Environment enum
                required=True,
                allowed_values=['development', 'staging', 'production', 'testing']
            ),

            # Network Configuration
            'host': ValidationRule(
                name='host',
                description='Host address validation',
                field_type=str,
                required=True,
                custom_validator=self._validate_host
            ),

            'port': ValidationRule(
                name='port',
                description='Port number validation',
                field_type=int,
                required=True,
                min_value=1024,
                max_value=65535
            ),

            'coordinator_port': ValidationRule(
                name='coordinator_port',
                description='Coordinator port validation',
                field_type=int,
                required=True,
                min_value=1024,
                max_value=65535,
                custom_validator=self._validate_port_availability
            ),

            'mpc_port': ValidationRule(
                name='mpc_port',
                description='MPC server port validation',
                field_type=int,
                required=True,
                min_value=1024,
                max_value=65535,
                custom_validator=self._validate_port_availability
            ),

            # Database Configuration
            'postgresql_host': ValidationRule(
                name='postgresql_host',
                description='PostgreSQL host validation',
                field_type=str,
                custom_validator=self._validate_host,
                depends_on=['postgresql_enabled']
            ),

            'postgresql_port': ValidationRule(
                name='postgresql_port',
                description='PostgreSQL port validation',
                field_type=int,
                min_value=1024,
                max_value=65535,
                depends_on=['postgresql_enabled']
            ),

            'postgresql_database': ValidationRule(
                name='postgresql_database',
                description='PostgreSQL database name validation',
                field_type=str,
                min_length=1,
                max_length=63,
                pattern=r'^[a-zA-Z_][a-zA-Z0-9_]*$',
                depends_on=['postgresql_enabled']
            ),

            'postgresql_username': ValidationRule(
                name='postgresql_username',
                description='PostgreSQL username validation',
                field_type=str,
                min_length=1,
                max_length=63,
                depends_on=['postgresql_enabled']
            ),

            'postgresql_password': ValidationRule(
                name='postgresql_password',
                description='PostgreSQL password validation',
                field_type=str,
                min_length=8,
                depends_on=['postgresql_enabled']
            ),

            # Redis Configuration
            'redis_host': ValidationRule(
                name='redis_host',
                description='Redis host validation',
                field_type=str,
                custom_validator=self._validate_host,
                depends_on=['redis_enabled']
            ),

            'redis_port': ValidationRule(
                name='redis_port',
                description='Redis port validation',
                field_type=int,
                min_value=1024,
                max_value=65535,
                depends_on=['redis_enabled']
            ),

            'redis_password': ValidationRule(
                name='redis_password',
                description='Redis password validation',
                field_type=str,
                min_length=8,
                depends_on=['redis_enabled']
            ),

            # Security Configuration
            'jwt_secret_key': ValidationRule(
                name='jwt_secret_key',
                description='JWT secret key validation',
                field_type=str,
                required=True,
                min_length=32,
                severity=ValidationSeverity.ERROR
            ),

            'api_keys_min_length': ValidationRule(
                name='api_keys_min_length',
                description='API key minimum length validation',
                field_type=int,
                min_value=16,
                max_value=64,
                depends_on=['api_keys_enabled']
            ),

            'api_keys_max_length': ValidationRule(
                name='api_keys_max_length',
                description='API key maximum length validation',
                field_type=int,
                min_value=32,
                max_value=128,
                depends_on=['api_keys_enabled'],
                custom_validator=self._validate_api_key_lengths
            ),

            # TLS Configuration
            'tls_cert_path': ValidationRule(
                name='tls_cert_path',
                description='TLS certificate path validation',
                field_type=str,
                custom_validator=self._validate_file_path,
                depends_on=['tls_enabled']
            ),

            'tls_key_path': ValidationRule(
                name='tls_key_path',
                description='TLS private key path validation',
                field_type=str,
                custom_validator=self._validate_file_path,
                depends_on=['tls_enabled']
            ),

            # ZK Configuration
            'zk_circuit_path': ValidationRule(
                name='zk_circuit_path',
                description='ZK circuit path validation',
                field_type=str,
                custom_validator=self._validate_directory_path,
                depends_on=['zk_enabled']
            ),

            'zk_proving_key_path': ValidationRule(
                name='zk_proving_key_path',
                description='ZK proving key path validation',
                field_type=str,
                custom_validator=self._validate_file_path,
                depends_on=['zk_enabled']
            ),

            'zk_verification_key_path': ValidationRule(
                name='zk_verification_key_path',
                description='ZK verification key path validation',
                field_type=str,
                custom_validator=self._validate_file_path,
                depends_on=['zk_enabled']
            ),

            # Performance Configuration
            'max_concurrent_requests': ValidationRule(
                name='max_concurrent_requests',
                description='Maximum concurrent requests validation',
                field_type=int,
                min_value=1,
                max_value=10000
            ),

            'worker_processes': ValidationRule(
                name='worker_processes',
                description='Worker processes validation',
                field_type=int,
                min_value=1,
                max_value=100
            ),

            'request_timeout_seconds': ValidationRule(
                name='request_timeout_seconds',
                description='Request timeout validation',
                field_type=int,
                min_value=1,
                max_value=300
            ),

            # Logging Configuration
            'log_level': ValidationRule(
                name='log_level',
                description='Log level validation',
                field_type=str,
                allowed_values=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
            ),

            'log_max_size_mb': ValidationRule(
                name='log_max_size_mb',
                description='Log file size validation',
                field_type=int,
                min_value=1,
                max_value=1000
            ),

            'log_backup_count': ValidationRule(
                name='log_backup_count',
                description='Log backup count validation',
                field_type=int,
                min_value=1,
                max_value=100
            ),

            # Federated Learning Configuration
            'model_batch_size': ValidationRule(
                name='model_batch_size',
                description='Model batch size validation',
                field_type=int,
                min_value=1,
                max_value=1024
            ),

            'model_learning_rate': ValidationRule(
                name='model_learning_rate',
                description='Learning rate validation',
                field_type=float,
                min_value=0.0001,
                max_value=1.0
            ),

            'model_epochs': ValidationRule(
                name='model_epochs',
                description='Training epochs validation',
                field_type=int,
                min_value=1,
                max_value=1000
            ),

            'privacy_epsilon': ValidationRule(
                name='privacy_epsilon',
                description='Privacy epsilon validation',
                field_type=float,
                min_value=0.1,
                max_value=10.0
            ),

            'privacy_delta': ValidationRule(
                name='privacy_delta',
                description='Privacy delta validation',
                field_type=float,
                min_value=1e-10,
                max_value=1e-3
            )
        }

    def validate_config(self, config: Any) -> List[ValidationResult]:
        """Validate configuration object against all rules."""
        results = []

        # Get all config attributes
        if hasattr(config, '__dict__'):
            config_dict = config.__dict__
        elif isinstance(config, dict):
            config_dict = config
        else:
            results.append(ValidationResult(
                field='config',
                severity=ValidationSeverity.ERROR,
                message='Invalid configuration object type',
                suggestion='Configuration must be a dataclass or dictionary'
            ))
            return results

        # Validate each field
        for field_name, value in config_dict.items():
            if field_name.startswith('_'):  # Skip private attributes
                continue

            if field_name in self.validation_rules:
                rule_results = self._validate_field(field_name, value, config_dict)
                results.extend(rule_results)
            else:
                # Unknown field - this could be a warning
                results.append(ValidationResult(
                    field=field_name,
                    severity=ValidationSeverity.WARNING,
                    message=f'Unknown configuration field: {field_name}',
                    value=value
                ))

        # Cross-field validations
        cross_field_results = self._validate_cross_field_rules(config_dict)
        results.extend(cross_field_results)

        # Environment-specific validations
        env_results = self._validate_environment_specific(config_dict)
        results.extend(env_results)

        return results

    def _validate_field(self, field_name: str, value: Any, config_dict: Dict[str, Any]) -> List[ValidationResult]:
        """Validate a single field against its rule."""
        results = []
        rule = self.validation_rules[field_name]

        # Check dependencies
        if rule.depends_on:
            for dependency in rule.depends_on:
                if dependency in config_dict and not config_dict[dependency]:
                    return results  # Skip validation if dependency is not met

        # Check required fields
        if rule.required and (value is None or value == "" or value == []):
            results.append(ValidationResult(
                field=field_name,
                severity=rule.severity,
                message=f'Required field {field_name} is missing or empty',
                suggestion=f'Provide a value for {field_name}',
                value=value
            ))
            return results

        # Skip further validation if value is None and not required
        if value is None and not rule.required:
            return results

        # Type validation
        if not isinstance(value, rule.field_type):
            try:
                # Try to convert type
                if rule.field_type == int and isinstance(value, str):
                    value = int(value)
                elif rule.field_type == float and isinstance(value, str):
                    value = float(value)
                elif rule.field_type == bool and isinstance(value, str):
                    value = value.lower() in ('true', '1', 'yes', 'on')
                elif rule.field_type == str:
                    # Special handling for Environment enum
                    if (hasattr(value, 'value') and
                        hasattr(value, '__class__') and
                        'Environment' in value.__class__.__name__):
                        value = value.value  # Convert enum to its string value
                    else:
                        value = str(value)
                else:
                    raise ValueError(f"Cannot convert {type(value)} to {rule.field_type}")
            except (ValueError, TypeError):
                results.append(ValidationResult(
                    field=field_name,
                    severity=rule.severity,
                    message=f'Field {field_name} must be of type {rule.field_type.__name__}, got {type(value).__name__}',
                    suggestion=f'Convert {field_name} to {rule.field_type.__name__}',
                    value=value
                ))
                return results

        # Range validation for numbers
        if isinstance(value, (int, float)):
            if rule.min_value is not None and value < rule.min_value:
                results.append(ValidationResult(
                    field=field_name,
                    severity=rule.severity,
                    message=f'Field {field_name} must be at least {rule.min_value}, got {value}',
                    suggestion=f'Increase {field_name} to at least {rule.min_value}',
                    value=value
                ))

            if rule.max_value is not None and value > rule.max_value:
                results.append(ValidationResult(
                    field=field_name,
                    severity=rule.severity,
                    message=f'Field {field_name} must be at most {rule.max_value}, got {value}',
                    suggestion=f'Decrease {field_name} to at most {rule.max_value}',
                    value=value
                ))

        # Length validation for strings and lists
        if isinstance(value, (str, list)):
            length = len(value)

            if rule.min_length is not None and length < rule.min_length:
                results.append(ValidationResult(
                    field=field_name,
                    severity=rule.severity,
                    message=f'Field {field_name} length must be at least {rule.min_length}, got {length}',
                    suggestion=f'Increase {field_name} length to at least {rule.min_length}',
                    value=value
                ))

            if rule.max_length is not None and length > rule.max_length:
                results.append(ValidationResult(
                    field=field_name,
                    severity=rule.severity,
                    message=f'Field {field_name} length must be at most {rule.max_length}, got {length}',
                    suggestion=f'Decrease {field_name} length to at most {rule.max_length}',
                    value=value
                ))

        # Allowed values validation
        if rule.allowed_values:
            # Special handling for Environment enum
            from fedzk.config.environment import Environment  # Import here to avoid circular import

            # Handle Environment enum specially
            if (hasattr(value, 'value') and
                hasattr(value, '__class__') and
                'Environment' in value.__class__.__name__):
                # Convert Environment enum to its string value for validation
                value = value.value

            if value not in rule.allowed_values:
                results.append(ValidationResult(
                    field=field_name,
                    severity=rule.severity,
                    message=f'Field {field_name} must be one of: {rule.allowed_values}, got {value}',
                    suggestion=f'Change {field_name} to one of: {rule.allowed_values}',
                    value=value
                ))

        # Pattern validation
        if rule.pattern and isinstance(value, str):
            if isinstance(rule.pattern, str):
                pattern = re.compile(rule.pattern)
            else:
                pattern = rule.pattern

            if not pattern.match(value):
                results.append(ValidationResult(
                    field=field_name,
                    severity=rule.severity,
                    message=f'Field {field_name} does not match required pattern',
                    suggestion=f'Ensure {field_name} matches the required format',
                    value=value
                ))

        # Custom validation
        if rule.custom_validator:
            try:
                custom_result = rule.custom_validator(value, config_dict)
                if isinstance(custom_result, str):
                    results.append(ValidationResult(
                        field=field_name,
                        severity=rule.severity,
                        message=custom_result,
                        value=value
                    ))
                elif isinstance(custom_result, dict):
                    results.append(ValidationResult(
                        field=field_name,
                        severity=custom_result.get('severity', rule.severity),
                        message=custom_result.get('message', 'Custom validation failed'),
                        suggestion=custom_result.get('suggestion'),
                        value=value
                    ))
            except Exception as e:
                results.append(ValidationResult(
                    field=field_name,
                    severity=ValidationSeverity.ERROR,
                    message=f'Custom validation failed for {field_name}: {str(e)}',
                    value=value
                ))

        return results

    def _validate_cross_field_rules(self, config_dict: Dict[str, Any]) -> List[ValidationResult]:
        """Validate cross-field relationships and dependencies."""
        results = []

        # Port uniqueness validation
        ports = []
        port_fields = ['port', 'coordinator_port', 'mpc_port', 'postgresql_port', 'redis_port', 'metrics_port']
        for field in port_fields:
            if field in config_dict and config_dict[field]:
                port = config_dict[field]
                if port in ports:
                    results.append(ValidationResult(
                        field=field,
                        severity=ValidationSeverity.ERROR,
                        message=f'Port {port} is already used by another service',
                        suggestion='Choose a different port number'
                    ))
                else:
                    ports.append(port)

        # Database configuration validation
        if config_dict.get('postgresql_enabled') and config_dict.get('external_database_enabled'):
            results.append(ValidationResult(
                field='database',
                severity=ValidationSeverity.WARNING,
                message='Both PostgreSQL and external database are enabled',
                suggestion='Choose either PostgreSQL or external database, not both'
            ))

        # Redis configuration validation
        if config_dict.get('redis_enabled') and config_dict.get('external_redis_enabled'):
            results.append(ValidationResult(
                field='redis',
                severity=ValidationSeverity.WARNING,
                message='Both Redis and external Redis are enabled',
                suggestion='Choose either Redis or external Redis, not both'
            ))

        # Service dependency validation
        if config_dict.get('mpc_enabled') and not config_dict.get('coordinator_enabled'):
            results.append(ValidationResult(
                field='mpc_enabled',
                severity=ValidationSeverity.WARNING,
                message='MPC is enabled but Coordinator is disabled',
                suggestion='Enable Coordinator service for MPC to function properly'
            ))

        return results

    def _validate_environment_specific(self, config_dict: Dict[str, Any]) -> List[ValidationResult]:
        """Validate environment-specific configuration requirements."""
        results = []
        environment = config_dict.get('environment', 'development')

        if environment == 'production':
            # Production-specific validations
            required_prod_fields = [
                'jwt_secret_key',
                'tls_enabled',
                'log_level'
            ]

            for field in required_prod_fields:
                if not config_dict.get(field):
                    results.append(ValidationResult(
                        field=field,
                        severity=ValidationSeverity.ERROR,
                        message=f'Field {field} is required in production environment',
                        suggestion=f'Configure {field} for production deployment'
                    ))

            # Database requirements in production
            if not config_dict.get('postgresql_enabled') and not config_dict.get('external_database_enabled'):
                results.append(ValidationResult(
                    field='database',
                    severity=ValidationSeverity.ERROR,
                    message='Database configuration is required in production',
                    suggestion='Enable PostgreSQL or configure external database'
                ))

            # Security requirements in production
            if not config_dict.get('api_keys_enabled'):
                results.append(ValidationResult(
                    field='api_keys_enabled',
                    severity=ValidationSeverity.WARNING,
                    message='API key authentication should be enabled in production',
                    suggestion='Enable API key authentication for security'
                ))

        elif environment == 'development':
            # Development-specific warnings
            if config_dict.get('debug') == False:
                results.append(ValidationResult(
                    field='debug',
                    severity=ValidationSeverity.INFO,
                    message='Debug mode is disabled in development',
                    suggestion='Consider enabling debug mode for development'
                ))

        return results

    # Custom validation functions

    def _validate_host(self, value: str, config_dict: Dict[str, Any]) -> Optional[str]:
        """Validate host address."""
        if not value:
            return None

        # Allow localhost, hostnames, and IP addresses
        if value in ['localhost', '127.0.0.1', '0.0.0.0']:
            return None

        # Validate IP address
        try:
            ipaddress.ip_address(value)
            return None
        except ValueError:
            pass

        # Validate hostname pattern
        hostname_pattern = re.compile(r'^[a-zA-Z0-9]([a-zA-Z0-9\-]{0,61}[a-zA-Z0-9])?(\.[a-zA-Z0-9]([a-zA-Z0-9\-]{0,61}[a-zA-Z0-9])?)*$')
        if not hostname_pattern.match(value):
            return f"Invalid host format: {value}"

        return None

    def _validate_port_availability(self, value: int, config_dict: Dict[str, Any]) -> Optional[str]:
        """Validate port availability."""
        import socket

        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                result = s.connect_ex(('127.0.0.1', value))
                if result == 0:
                    return f"Port {value} is already in use"
        except Exception:
            pass  # If we can't check, assume it's okay

        return None

    def _validate_file_path(self, value: str, config_dict: Dict[str, Any]) -> Optional[str]:
        """Validate file path exists and is readable."""
        if not value:
            return None

        path = Path(value)
        if not path.exists():
            return f"File does not exist: {value}"

        if not path.is_file():
            return f"Path is not a file: {value}"

        try:
            with open(path, 'r'):
                pass
        except PermissionError:
            return f"File is not readable: {value}"

        return None

    def _validate_directory_path(self, value: str, config_dict: Dict[str, Any]) -> Optional[str]:
        """Validate directory path exists and is accessible."""
        if not value:
            return None

        path = Path(value)
        if not path.exists():
            return f"Directory does not exist: {value}"

        if not path.is_dir():
            return f"Path is not a directory: {value}"

        try:
            list(path.iterdir())
        except PermissionError:
            return f"Directory is not accessible: {value}"

        return None

    def _validate_api_key_lengths(self, max_length: int, config_dict: Dict[str, Any]) -> Optional[str]:
        """Validate API key length relationships."""
        min_length = config_dict.get('api_keys_min_length', 16)

        if max_length < min_length:
            return f"API key max length ({max_length}) must be greater than min length ({min_length})"

        return None

    def get_validation_summary(self, results: List[ValidationResult]) -> Dict[str, Any]:
        """Get a summary of validation results."""
        summary = {
            'total_issues': len(results),
            'errors': 0,
            'warnings': 0,
            'info': 0,
            'issues_by_field': {},
            'issues_by_severity': {
                'error': [],
                'warning': [],
                'info': []
            }
        }

        for result in results:
            summary[f"{result.severity.value}s"] += 1

            if result.field not in summary['issues_by_field']:
                summary['issues_by_field'][result.field] = []
            summary['issues_by_field'][result.field].append({
                'severity': result.severity.value,
                'message': result.message,
                'suggestion': result.suggestion
            })

            summary['issues_by_severity'][result.severity.value].append({
                'field': result.field,
                'message': result.message,
                'suggestion': result.suggestion
            })

        return summary

    def print_validation_report(self, results: List[ValidationResult]):
        """Print a formatted validation report."""
        if not results:
            print("✅ No validation issues found!")
            return

        summary = self.get_validation_summary(results)

        print("🔍 Configuration Validation Report")
        print("=" * 50)
        print(f"Total Issues: {summary['total_issues']}")
        print(f"Errors: {summary['errors']}")
        print(f"Warnings: {summary['warnings']}")
        print(f"Info: {summary['info']}")
        print()

        for result in results:
            severity_icon = {
                ValidationSeverity.ERROR: "❌",
                ValidationSeverity.WARNING: "⚠️",
                ValidationSeverity.INFO: "ℹ️"
            }[result.severity]

            print(f"{severity_icon} {result.field}: {result.message}")
            if result.suggestion:
                print(f"   💡 {result.suggestion}")
            print()


# Global validator instance
_validator = None

def get_config_validator() -> ConfigValidator:
    """Get the global configuration validator instance."""
    global _validator
    if _validator is None:
        _validator = ConfigValidator()
    return _validator

def validate_config(config: Any) -> List[ValidationResult]:
    """Validate a configuration object."""
    return get_config_validator().validate_config(config)

def print_validation_report(results: List[ValidationResult]):
    """Print a formatted validation report."""
    get_config_validator().print_validation_report(results)

