"""
FEDzk Structured Logging System
===============================

Comprehensive logging infrastructure with:
- Structured JSON logging
- Log aggregation capabilities
- Security and compliance features
- Audit logging for security events
"""

import json
import logging
import logging.handlers
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List, Union
import uuid
import hashlib
import socket
import os

try:
    import structlog
    STRUCTLOG_AVAILABLE = True
except ImportError:
    STRUCTLOG_AVAILABLE = False

try:
    from elasticsearch import Elasticsearch
    ELASTICSEARCH_AVAILABLE = True
except ImportError:
    ELASTICSEARCH_AVAILABLE = False


class FEDzkJSONFormatter(logging.Formatter):
    """
    Custom JSON formatter for FEDzk structured logging
    """

    def __init__(self, service_name: str = "fedzk", include_extra: bool = True):
        super().__init__()
        self.service_name = service_name
        self.include_extra = include_extra
        self.hostname = socket.gethostname()
        self.pid = os.getpid()

    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON"""

        # Base log entry
        log_entry = {
            "timestamp": datetime.now(datetime.UTC).isoformat() + "Z",
            "service": self.service_name,
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "hostname": self.hostname,
            "pid": self.pid,
            "thread": record.threadName if record.threadName else f"Thread-{record.thread}",
        }

        # Add exception info if present
        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)

        # Add structured data from record
        if hasattr(record, 'structured_data'):
            log_entry.update(record.structured_data)

        # Add extra fields if enabled
        if self.include_extra:
            for key, value in record.__dict__.items():
                if key not in {
                    'name', 'msg', 'args', 'levelname', 'levelno', 'pathname',
                    'filename', 'module', 'exc_info', 'exc_text', 'stack_info',
                    'lineno', 'funcName', 'created', 'msecs', 'relativeCreated',
                    'thread', 'threadName', 'processName', 'process', 'message',
                    'structured_data'
                }:
                    log_entry[f"extra_{key}"] = value

        # Add request ID if available
        if hasattr(record, 'request_id'):
            log_entry["request_id"] = record.request_id

        # Add correlation ID if available
        if hasattr(record, 'correlation_id'):
            log_entry["correlation_id"] = record.correlation_id

        return json.dumps(log_entry, default=str, ensure_ascii=False)


class FEDzkSecurityFormatter(FEDzkJSONFormatter):
    """
    Security-focused JSON formatter with compliance fields
    """

    def format(self, record: logging.LogRecord) -> str:
        """Format log record with security compliance fields"""

        # Get base JSON
        json_str = super().format(record)
        log_entry = json.loads(json_str)

        # Add security compliance fields
        log_entry.update({
            "compliance": {
                "pci_dss": self._is_pci_compliant(record),
                "gdpr": self._is_gdpr_compliant(record),
                "sox": self._is_sox_compliant(record),
                "hipaa": self._is_hipaa_compliant(record)
            },
            "security_level": self._get_security_level(record),
            "data_classification": self._get_data_classification(record),
            "audit_trail": self._generate_audit_trail(record)
        })

        return json.dumps(log_entry, default=str, ensure_ascii=False)

    def _is_pci_compliant(self, record: logging.LogRecord) -> bool:
        """Check PCI DSS compliance for log entry"""
        sensitive_keywords = ['card', 'payment', 'credit', 'cvv', 'pan']
        message = record.getMessage().lower()
        return not any(keyword in message for keyword in sensitive_keywords)

    def _is_gdpr_compliant(self, record: logging.LogRecord) -> bool:
        """Check GDPR compliance for log entry"""
        pii_keywords = ['email', 'phone', 'ssn', 'social', 'personal']
        message = record.getMessage().lower()
        return not any(keyword in message for keyword in pii_keywords)

    def _is_sox_compliant(self, record: logging.LogRecord) -> bool:
        """Check SOX compliance for log entry"""
        financial_keywords = ['financial', 'audit', 'transaction', 'ledger']
        message = record.getMessage().lower()
        return any(keyword in message for keyword in financial_keywords)

    def _is_hipaa_compliant(self, record: logging.LogRecord) -> bool:
        """Check HIPAA compliance for log entry"""
        phi_keywords = ['medical', 'health', 'patient', 'diagnosis', 'treatment']
        message = record.getMessage().lower()
        return not any(keyword in message for keyword in phi_keywords)

    def _get_security_level(self, record: logging.LogRecord) -> str:
        """Determine security level of log entry"""
        if record.levelno >= logging.CRITICAL:
            return "CRITICAL"
        elif record.levelno >= logging.ERROR:
            return "HIGH"
        elif record.levelno >= logging.WARNING:
            return "MEDIUM"
        else:
            return "LOW"

    def _get_data_classification(self, record: logging.LogRecord) -> str:
        """Determine data classification of log entry"""
        sensitive_keywords = ['password', 'secret', 'key', 'token', 'credential']
        message = record.getMessage().lower()

        if any(keyword in message for keyword in sensitive_keywords):
            return "RESTRICTED"
        elif any(keyword in message for keyword in ['email', 'phone', 'name']):
            return "CONFIDENTIAL"
        elif any(keyword in message for keyword in ['user', 'session', 'request']):
            return "INTERNAL"
        else:
            return "PUBLIC"

    def _generate_audit_trail(self, record: logging.LogRecord) -> Dict[str, Any]:
        """Generate audit trail information"""
        return {
            "event_id": str(uuid.uuid4()),
            "event_hash": hashlib.sha256(
                f"{record.created}:{record.getMessage()}:{record.levelname}".encode()
            ).hexdigest()[:16],
            "source_ip": getattr(record, 'source_ip', 'unknown'),
            "user_agent": getattr(record, 'user_agent', 'unknown'),
            "session_id": getattr(record, 'session_id', 'unknown'),
            "timestamp_epoch": record.created
        }


class FEDzkLogHandler(logging.Handler):
    """
    Custom log handler with advanced features
    """

    def __init__(self, level: int = logging.NOTSET):
        super().__init__(level)
        self.buffer = []
        self.buffer_size = 100
        self.flush_interval = 30  # seconds
        self.last_flush = time.time()
        self.lock = threading.Lock()

    def emit(self, record: logging.LogRecord) -> None:
        """Emit log record with buffering"""
        try:
            with self.lock:
                self.buffer.append(self.format(record))

                # Check if buffer should be flushed
                if (len(self.buffer) >= self.buffer_size or
                    time.time() - self.last_flush >= self.flush_interval):
                    self.flush()
        except Exception:
            self.handleError(record)

    def flush(self) -> None:
        """Flush buffered log entries"""
        with self.lock:
            if self.buffer:
                # Process buffered entries (implement in subclasses)
                self._process_buffer(self.buffer)
                self.buffer.clear()
                self.last_flush = time.time()

    def _process_buffer(self, entries: List[str]) -> None:
        """Process buffered log entries (override in subclasses)"""
        pass


class ElasticsearchHandler(FEDzkLogHandler):
    """
    Log handler that sends logs to Elasticsearch
    """

    def __init__(self, hosts: List[str], index_prefix: str = "fedzk-logs",
                 level: int = logging.NOTSET):
        super().__init__(level)
        self.hosts = hosts
        self.index_prefix = index_prefix
        self.es_client = None

        if ELASTICSEARCH_AVAILABLE:
            self.es_client = Elasticsearch(hosts=hosts)

    def _process_buffer(self, entries: List[str]) -> None:
        """Send buffered entries to Elasticsearch"""
        if not self.es_client:
            return

        actions = []
        for entry in entries:
            try:
                log_data = json.loads(entry)
                index_name = f"{self.index_prefix}-{datetime.now(datetime.UTC).strftime('%Y.%m.%d')}"

                actions.extend([
                    {"index": {"_index": index_name}},
                    log_data
                ])
            except json.JSONDecodeError:
                continue

        if actions:
            try:
                from elasticsearch.helpers import bulk
                bulk(self.es_client, actions)
            except Exception as e:
                print(f"Failed to send logs to Elasticsearch: {e}")


class FEDzkLogger:
    """
    Main FEDzk logging interface
    """

    def __init__(self, service_name: str = "fedzk",
                 log_level: str = "INFO",
                 enable_structlog: bool = True):
        self.service_name = service_name
        self.log_level = getattr(logging, log_level.upper(), logging.INFO)
        self.enable_structlog = enable_structlog and STRUCTLOG_AVAILABLE

        # Create logger
        self.logger = logging.getLogger(service_name)
        self.logger.setLevel(self.log_level)

        # Remove existing handlers to avoid duplicates
        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)

        # Add console handler with JSON formatter
        console_handler = logging.StreamHandler(sys.stdout)
        json_formatter = FEDzkJSONFormatter(service_name)
        console_handler.setFormatter(json_formatter)
        self.logger.addHandler(console_handler)

        # Add security handler for security events
        security_handler = logging.StreamHandler(sys.stderr)
        security_formatter = FEDzkSecurityFormatter(service_name)
        security_handler.setFormatter(security_formatter)
        security_handler.setLevel(logging.WARNING)
        self.logger.addHandler(security_handler)

        # Configure structlog if available
        if self.enable_structlog:
            self._configure_structlog()

        # Request context storage
        self._request_context = threading.local()

    def _configure_structlog(self):
        """Configure structlog for enhanced structured logging"""
        import structlog

        shared_processors = [
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
        ]

        structlog.configure(
            processors=shared_processors + [
                structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
            ],
            context_class=dict,
            logger_factory=structlog.stdlib.LoggerFactory(),
            wrapper_class=structlog.stdlib.BoundLogger,
            cache_logger_on_first_use=True,
        )

    def add_file_handler(self, log_file: str, max_bytes: int = 10*1024*1024,
                        backup_count: int = 5):
        """Add rotating file handler"""
        file_handler = logging.handlers.RotatingFileHandler(
            log_file, maxBytes=max_bytes, backupCount=backup_count
        )
        json_formatter = FEDzkJSONFormatter(self.service_name)
        file_handler.setFormatter(json_formatter)
        self.logger.addHandler(file_handler)

    def add_elasticsearch_handler(self, hosts: List[str],
                                index_prefix: str = "fedzk-logs"):
        """Add Elasticsearch handler for log aggregation"""
        if ELASTICSEARCH_AVAILABLE:
            es_handler = ElasticsearchHandler(hosts, index_prefix)
            es_handler.setLevel(logging.INFO)
            self.logger.addHandler(es_handler)
        else:
            self.logger.warning("Elasticsearch not available, skipping ES handler")

    def set_request_context(self, request_id: str = None,
                           correlation_id: str = None,
                           user_id: str = None,
                           source_ip: str = None):
        """Set request context for structured logging"""
        self._request_context.request_id = request_id or str(uuid.uuid4())
        self._request_context.correlation_id = correlation_id
        self._request_context.user_id = user_id
        self._request_context.source_ip = source_ip

    def clear_request_context(self):
        """Clear request context"""
        self._request_context.__dict__.clear()

    def log_structured(self, level: str, message: str,
                      structured_data: Dict[str, Any] = None,
                      **kwargs):
        """Log with structured data"""
        # Get request context
        extra = {}
        if hasattr(self._request_context, 'request_id'):
            extra['request_id'] = self._request_context.request_id
        if hasattr(self._request_context, 'correlation_id'):
            extra['correlation_id'] = self._request_context.correlation_id
        if hasattr(self._request_context, 'user_id'):
            extra['user_id'] = self._request_context.user_id
        if hasattr(self._request_context, 'source_ip'):
            extra['source_ip'] = self._request_context.source_ip

        # Add structured data
        if structured_data:
            extra['structured_data'] = structured_data

        # Add extra kwargs
        extra.update(kwargs)

        # Log with appropriate level
        log_method = getattr(self.logger, level.lower(), self.logger.info)
        log_method(message, extra=extra)

    def log_security_event(self, event_type: str, severity: str,
                          details: Dict[str, Any] = None,
                          user_id: str = None,
                          source_ip: str = None):
        """Log security event with compliance information"""
        structured_data = {
            "event_type": event_type,
            "severity": severity,
            "security_event": True,
            "compliance_required": True
        }

        if details:
            structured_data.update(details)

        if user_id:
            structured_data["user_id"] = user_id

        if source_ip:
            structured_data["source_ip"] = source_ip

        self.log_structured("warning", f"Security event: {event_type}",
                          structured_data, source_ip=source_ip)

    def log_audit_event(self, action: str, resource: str,
                       user_id: str = None, success: bool = True,
                       details: Dict[str, Any] = None):
        """Log audit event for compliance"""
        structured_data = {
            "audit_event": True,
            "action": action,
            "resource": resource,
            "success": success,
            "timestamp": datetime.now(datetime.UTC).isoformat(),
            "compliance": {
                "sox": True,  # Financial audit compliance
                "gdpr": True if not details else not any(
                    k in str(details) for k in ['email', 'phone', 'personal']
                )
            }
        }

        if details:
            structured_data.update(details)

        if user_id:
            structured_data["user_id"] = user_id

        level = "info" if success else "warning"
        self.log_structured(level, f"Audit: {action} on {resource}",
                          structured_data)

    def log_performance_metric(self, operation: str, duration: float,
                             success: bool = True, metadata: Dict[str, Any] = None):
        """Log performance metric"""
        structured_data = {
            "performance_metric": True,
            "operation": operation,
            "duration_seconds": duration,
            "success": success
        }

        if metadata:
            structured_data.update(metadata)

        level = "info" if success else "warning"
        status = "completed" if success else "failed"
        self.log_structured(level, f"Performance: {operation} {status} in {duration:.3f}s",
                          structured_data)


class FEDzkLogAggregator:
    """
    Log aggregation and analysis system
    """

    def __init__(self, retention_days: int = 30):
        self.retention_days = retention_days
        self.aggregated_logs = {}
        self.lock = threading.Lock()

    def aggregate_logs(self, log_entry: Dict[str, Any]):
        """Aggregate log entries for analysis"""
        with self.lock:
            # Aggregate by level
            level = log_entry.get('level', 'UNKNOWN')
            if level not in self.aggregated_logs:
                self.aggregated_logs[level] = []

            self.aggregated_logs[level].append(log_entry)

            # Keep only recent entries
            cutoff_time = time.time() - (self.retention_days * 24 * 60 * 60)
            self.aggregated_logs[level] = [
                entry for entry in self.aggregated_logs[level]
                if entry.get('timestamp_epoch', 0) > cutoff_time
            ]

    def get_log_statistics(self) -> Dict[str, Any]:
        """Get aggregated log statistics"""
        with self.lock:
            stats = {}
            for level, entries in self.aggregated_logs.items():
                stats[level] = {
                    "count": len(entries),
                    "recent_count": len([
                        e for e in entries
                        if time.time() - e.get('timestamp_epoch', 0) < 3600  # Last hour
                    ])
                }
            return stats

    def get_error_patterns(self) -> Dict[str, int]:
        """Analyze error patterns"""
        with self.lock:
            patterns = {}
            for entry in self.aggregated_logs.get('ERROR', []):
                message = entry.get('message', '')
                # Simple pattern extraction
                if 'exception' in message.lower():
                    patterns['exceptions'] = patterns.get('exceptions', 0) + 1
                elif 'timeout' in message.lower():
                    patterns['timeouts'] = patterns.get('timeouts', 0) + 1
                elif 'connection' in message.lower():
                    patterns['connection_errors'] = patterns.get('connection_errors', 0) + 1
                else:
                    patterns['other_errors'] = patterns.get('other_errors', 0) + 1

            return patterns

    def get_performance_insights(self) -> Dict[str, Any]:
        """Analyze performance patterns"""
        with self.lock:
            insights = {
                "slow_operations": [],
                "error_rates": {},
                "peak_usage_times": []
            }

            # Analyze performance metrics
            for entry in self.aggregated_logs.get('INFO', []):
                if entry.get('performance_metric'):
                    duration = entry.get('duration_seconds', 0)
                    if duration > 5.0:  # Slow operation threshold
                        insights["slow_operations"].append({
                            "operation": entry.get('operation'),
                            "duration": duration,
                            "timestamp": entry.get('timestamp')
                        })

            return insights


# Global instances
_logger = None
_aggregator = None


def get_logger(service_name: str = "fedzk") -> FEDzkLogger:
    """Get or create global logger instance"""
    global _logger
    if _logger is None:
        _logger = FEDzkLogger(service_name)
    return _logger


def get_aggregator() -> FEDzkLogAggregator:
    """Get or create global aggregator instance"""
    global _aggregator
    if _aggregator is None:
        _aggregator = FEDzkLogAggregator()
    return _aggregator


def initialize_logging(service_name: str = "fedzk",
                      log_level: str = "INFO",
                      log_file: str = None,
                      elasticsearch_hosts: List[str] = None,
                      enable_aggregation: bool = True) -> FEDzkLogger:
    """
    Initialize FEDzk logging system
    """
    global _logger, _aggregator

    # Create logger
    _logger = FEDzkLogger(service_name, log_level)

    # Add file handler if specified
    if log_file:
        _logger.add_file_handler(log_file)

    # Add Elasticsearch handler if configured
    if elasticsearch_hosts:
        _logger.add_elasticsearch_handler(elasticsearch_hosts)

    # Initialize aggregator
    if enable_aggregation:
        _aggregator = FEDzkLogAggregator()

    logger = _logger.logger
    logger.info(f"Initialized FEDzk logging system for {service_name}",
               extra={"structured_data": {"log_system_initialized": True}})

    return _logger


# Utility functions for easy integration
def log_zk_operation(operation: str, circuit_type: str, duration: float,
                    success: bool = True, proof_size: int = None):
    """Log ZK operation with structured data"""
    logger = get_logger()
    structured_data = {
        "zk_operation": True,
        "operation": operation,
        "circuit_type": circuit_type,
        "duration_seconds": duration,
        "success": success
    }

    if proof_size:
        structured_data["proof_size_bytes"] = proof_size

    logger.log_performance_metric(operation, duration, success, structured_data)


def log_security_violation(violation_type: str, user_id: str = None,
                          source_ip: str = None, details: Dict[str, Any] = None):
    """Log security violation"""
    logger = get_logger()
    logger.log_security_event(
        event_type=f"violation_{violation_type}",
        severity="HIGH",
        details=details,
        user_id=user_id,
        source_ip=source_ip
    )


def log_audit_trail(action: str, resource: str, user_id: str = None,
                   success: bool = True, changes: Dict[str, Any] = None):
    """Log audit trail entry"""
    logger = get_logger()
    audit_details = {"changes": changes} if changes else {}
    logger.log_audit_event(action, resource, user_id, success, audit_details)
