"""
FEDzk Security and Compliance Logging
=====================================

Advanced security logging with compliance support:
- GDPR, PCI DSS, SOX, HIPAA compliance tracking
- Security event correlation and analysis
- Audit trail generation and verification
- Data classification and handling
"""

import json
import logging
import hashlib
import hmac
import time
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Set
from dataclasses import dataclass, asdict
from enum import Enum
import re
import threading
from pathlib import Path

logger = logging.getLogger(__name__)


class ComplianceStandard(Enum):
    """Supported compliance standards"""
    GDPR = "gdpr"
    PCI_DSS = "pci_dss"
    SOX = "sox"
    HIPAA = "hipaa"
    ISO_27001 = "iso_27001"


class DataClassification(Enum):
    """Data classification levels"""
    PUBLIC = "public"
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"
    RESTRICTED = "restricted"


class SecurityEventType(Enum):
    """Security event types"""
    AUTHENTICATION_SUCCESS = "authentication_success"
    AUTHENTICATION_FAILURE = "authentication_failure"
    AUTHORIZATION_FAILURE = "authorization_failure"
    DATA_ACCESS = "data_access"
    DATA_MODIFICATION = "data_modification"
    SECURITY_VIOLATION = "security_violation"
    SUSPICIOUS_ACTIVITY = "suspicious_activity"
    BRUTE_FORCE_ATTACK = "brute_force_attack"
    SQL_INJECTION = "sql_injection"
    XSS_ATTACK = "xss_attack"
    CSRF_ATTACK = "csrf_attack"
    DATA_LEAKAGE = "data_leakage"
    PRIVILEGE_ESCALATION = "privilege_escalation"


@dataclass
class SecurityEvent:
    """Security event data structure"""
    event_id: str
    event_type: SecurityEventType
    timestamp: datetime
    user_id: Optional[str]
    session_id: Optional[str]
    source_ip: Optional[str]
    user_agent: Optional[str]
    resource: Optional[str]
    action: Optional[str]
    success: bool
    details: Dict[str, Any]
    compliance_flags: Dict[str, bool]
    data_classification: DataClassification
    severity: str
    correlation_id: Optional[str]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging"""
        data = asdict(self)
        data['event_type'] = self.event_type.value
        data['data_classification'] = self.data_classification.value
        data['timestamp'] = self.timestamp.isoformat()
        return data


@dataclass
class AuditEntry:
    """Audit trail entry"""
    audit_id: str
    timestamp: datetime
    action: str
    resource: str
    user_id: Optional[str]
    before_state: Optional[Dict[str, Any]]
    after_state: Optional[Dict[str, Any]]
    success: bool
    reason: Optional[str]
    compliance_standards: List[ComplianceStandard]
    data_classification: DataClassification
    checksum: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging"""
        data = asdict(self)
        data['compliance_standards'] = [std.value for std in self.compliance_standards]
        data['data_classification'] = self.data_classification.value
        data['timestamp'] = self.timestamp.isoformat()
        return data


class ComplianceChecker:
    """
    Compliance validation and checking system
    """

    def __init__(self):
        # Sensitive data patterns for compliance checking
        self.sensitive_patterns = {
            'gdpr': [
                r'\b\d{2}/\d{2}/\d{4}\b',  # Date of birth pattern
                r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',  # Email
                r'\b\d{10,15}\b',  # Phone numbers
                r'\b\d{3}-\d{2}-\d{4}\b',  # SSN pattern
            ],
            'pci_dss': [
                r'\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b',  # Credit card numbers
                r'\b\d{3,4}\b',  # CVV patterns
                r'\b\d{2}/\d{2,4}\b',  # Expiry dates
            ],
            'hipaa': [
                r'\b\d{3}-\d{3}-\d{4}\b',  # Medical record numbers
                r'\b[A-Z]{2}\d{6}\b',  # Medical IDs
                r'medical|diagnosis|treatment|patient',  # Medical keywords
            ]
        }

    def check_compliance(self, data: str, standards: List[ComplianceStandard]) -> Dict[str, bool]:
        """Check data against compliance standards"""
        results = {}

        for standard in standards:
            std_name = standard.value
            if std_name in self.sensitive_patterns:
                patterns = self.sensitive_patterns[std_name]
                compliant = not any(re.search(pattern, data, re.IGNORECASE) for pattern in patterns)
                results[std_name] = compliant
            else:
                results[std_name] = True  # Default to compliant for unknown standards

        return results

    def classify_data(self, data: str) -> DataClassification:
        """Classify data based on content analysis"""
        sensitive_keywords = {
            'restricted': ['password', 'secret', 'key', 'token', 'private', 'ssn', 'credit'],
            'confidential': ['email', 'phone', 'address', 'name', 'medical', 'financial'],
            'internal': ['user', 'session', 'request', 'response', 'log'],
        }

        data_lower = data.lower()

        for classification, keywords in sensitive_keywords.items():
            if any(keyword in data_lower for keyword in keywords):
                return DataClassification(classification.upper())

        return DataClassification.PUBLIC

    def validate_retention_policy(self, data_classification: DataClassification,
                                creation_date: datetime) -> bool:
        """Validate data retention policy compliance"""
        retention_periods = {
            DataClassification.RESTRICTED: 365,  # 1 year
            DataClassification.CONFIDENTIAL: 2555,  # 7 years
            DataClassification.INTERNAL: 365,  # 1 year
            DataClassification.PUBLIC: 90,  # 90 days
        }

        days_since_creation = (datetime.now(datetime.UTC) - creation_date).days
        max_retention = retention_periods.get(data_classification, 90)

        return days_since_creation <= max_retention


class SecurityEventLogger:
    """
    Specialized logger for security events
    """

    def __init__(self, service_name: str = "fedzk", compliance_checker: Optional[ComplianceChecker] = None):
        self.service_name = service_name
        self.compliance_checker = compliance_checker or ComplianceChecker()
        self.event_buffer = []
        self.correlation_tracker = {}
        self.lock = threading.Lock()

    def log_security_event(self, event_type: SecurityEventType,
                          user_id: Optional[str] = None,
                          session_id: Optional[str] = None,
                          source_ip: Optional[str] = None,
                          user_agent: Optional[str] = None,
                          resource: Optional[str] = None,
                          action: Optional[str] = None,
                          success: bool = True,
                          details: Optional[Dict[str, Any]] = None,
                          correlation_id: Optional[str] = None) -> str:
        """Log a security event with full compliance tracking"""

        event_id = self._generate_event_id()

        # Analyze data for compliance
        data_content = json.dumps(details or {})
        compliance_flags = self.compliance_checker.check_compliance(
            data_content,
            [ComplianceStandard.GDPR, ComplianceStandard.PCI_DSS, ComplianceStandard.HIPAA]
        )

        # Classify data
        data_classification = self.compliance_checker.classify_data(data_content)

        # Determine severity
        severity = self._calculate_severity(event_type, success, compliance_flags)

        # Create security event
        event = SecurityEvent(
            event_id=event_id,
            event_type=event_type,
            timestamp=datetime.now(datetime.UTC),
            user_id=user_id,
            session_id=session_id,
            source_ip=source_ip,
            user_agent=user_agent,
            resource=resource,
            action=action,
            success=success,
            details=details or {},
            compliance_flags=compliance_flags,
            data_classification=data_classification,
            severity=severity,
            correlation_id=correlation_id
        )

        # Buffer event
        with self.lock:
            self.event_buffer.append(event)

            # Track correlation
            if correlation_id:
                if correlation_id not in self.correlation_tracker:
                    self.correlation_tracker[correlation_id] = []
                self.correlation_tracker[correlation_id].append(event)

        # Log the event
        self._log_event(event)

        return event_id

    def _generate_event_id(self) -> str:
        """Generate unique event ID"""
        timestamp = str(int(time.time() * 1000000))
        random_part = hashlib.md5(str(time.time()).encode()).hexdigest()[:8]
        return f"sec_{timestamp}_{random_part}"

    def _calculate_severity(self, event_type: SecurityEventType, success: bool,
                           compliance_flags: Dict[str, bool]) -> str:
        """Calculate event severity"""

        # Base severity by event type
        severity_map = {
            SecurityEventType.AUTHENTICATION_FAILURE: "medium",
            SecurityEventType.AUTHORIZATION_FAILURE: "high",
            SecurityEventType.SECURITY_VIOLATION: "critical",
            SecurityEventType.BRUTE_FORCE_ATTACK: "critical",
            SecurityEventType.SQL_INJECTION: "critical",
            SecurityEventType.XSS_ATTACK: "high",
            SecurityEventType.DATA_LEAKAGE: "critical",
            SecurityEventType.PRIVILEGE_ESCALATION: "critical",
            SecurityEventType.SUSPICIOUS_ACTIVITY: "medium",
        }

        severity = severity_map.get(event_type, "low")

        # Increase severity for compliance violations
        if not all(compliance_flags.values()):
            if severity == "low":
                severity = "medium"
            elif severity == "medium":
                severity = "high"
            elif severity == "high":
                severity = "critical"

        # Decrease severity for successful events
        if success and severity == "low":
            severity = "info"

        return severity

    def _log_event(self, event: SecurityEvent):
        """Log security event using structured logger"""
        from .structured_logger import get_logger

        logger = get_logger()
        event_dict = event.to_dict()

        # Log with appropriate level based on severity
        level_map = {
            "info": "info",
            "low": "warning",
            "medium": "warning",
            "high": "error",
            "critical": "critical"
        }

        log_level = level_map.get(event.severity, "warning")

        logger.log_structured(
            log_level,
            f"Security event: {event.event_type.value}",
            {
                "security_event": True,
                "event_details": event_dict,
                "compliance_violations": [
                    std for std, compliant in event.compliance_flags.items()
                    if not compliant
                ]
            }
        )

    def get_events_by_correlation(self, correlation_id: str) -> List[SecurityEvent]:
        """Get all events related to a correlation ID"""
        with self.lock:
            return self.correlation_tracker.get(correlation_id, []).copy()

    def get_events_by_user(self, user_id: str, time_window_hours: int = 24) -> List[SecurityEvent]:
        """Get security events for a specific user within time window"""
        cutoff_time = datetime.now(datetime.UTC) - timedelta(hours=time_window_hours)

        with self.lock:
            return [
                event for event in self.event_buffer
                if event.user_id == user_id and event.timestamp > cutoff_time
            ]

    def detect_anomalous_activity(self, user_id: str, time_window_hours: int = 1) -> List[Dict[str, Any]]:
        """Detect anomalous security activity patterns"""
        events = self.get_events_by_user(user_id, time_window_hours)

        anomalies = []

        # Count event types
        event_counts = {}
        for event in events:
            event_type = event.event_type.value
            event_counts[event_type] = event_counts.get(event_type, 0) + 1

        # Detect brute force patterns
        auth_failures = event_counts.get(SecurityEventType.AUTHENTICATION_FAILURE.value, 0)
        if auth_failures >= 5:
            anomalies.append({
                "type": "brute_force_suspicion",
                "severity": "high",
                "description": f"{auth_failures} authentication failures in {time_window_hours} hours",
                "recommendation": "Implement account lockout or additional verification"
            })

        # Detect unusual access patterns
        suspicious_events = event_counts.get(SecurityEventType.SUSPICIOUS_ACTIVITY.value, 0)
        if suspicious_events >= 3:
            anomalies.append({
                "type": "unusual_activity",
                "severity": "medium",
                "description": f"{suspicious_events} suspicious activities detected",
                "recommendation": "Review user access patterns and behavior"
            })

        return anomalies

    def generate_compliance_report(self, time_window_days: int = 30) -> Dict[str, Any]:
        """Generate compliance report for the specified time window"""
        cutoff_date = datetime.now(datetime.UTC) - timedelta(days=time_window_days)

        with self.lock:
            relevant_events = [
                event for event in self.event_buffer
                if event.timestamp > cutoff_date
            ]

        report = {
            "report_period_days": time_window_days,
            "total_events": len(relevant_events),
            "compliance_summary": {},
            "severity_distribution": {},
            "top_event_types": {},
            "violations": []
        }

        # Analyze compliance
        for event in relevant_events:
            # Compliance summary
            for standard, compliant in event.compliance_flags.items():
                if standard not in report["compliance_summary"]:
                    report["compliance_summary"][standard] = {"compliant": 0, "violations": 0}

                if compliant:
                    report["compliance_summary"][standard]["compliant"] += 1
                else:
                    report["compliance_summary"][standard]["violations"] += 1
                    report["violations"].append({
                        "event_id": event.event_id,
                        "standard": standard,
                        "timestamp": event.timestamp.isoformat(),
                        "event_type": event.event_type.value
                    })

            # Severity distribution
            severity = event.severity
            report["severity_distribution"][severity] = report["severity_distribution"].get(severity, 0) + 1

            # Top event types
            event_type = event.event_type.value
            report["top_event_types"][event_type] = report["top_event_types"].get(event_type, 0) + 1

        return report


class AuditLogger:
    """
    Audit trail logger for compliance and accountability
    """

    def __init__(self, service_name: str = "fedzk",
                 enable_crypto_verification: bool = True,
                 key_file: Optional[str] = None):
        self.service_name = service_name
        self.enable_crypto_verification = enable_crypto_verification
        self.key_file = key_file or f"/etc/{service_name}/audit_key"
        self.audit_trail = []
        self.lock = threading.Lock()

        # Initialize cryptographic verification
        if self.enable_crypto_verification:
            self._init_crypto()

    def _init_crypto(self):
        """Initialize cryptographic verification for audit integrity"""
        try:
            # Generate or load audit key
            key_path = Path(self.key_file)
            if key_path.exists():
                with open(key_path, 'rb') as f:
                    self.audit_key = f.read()
            else:
                self.audit_key = self._generate_audit_key()
                key_path.parent.mkdir(parents=True, exist_ok=True)
                with open(key_path, 'wb') as f:
                    f.write(self.audit_key)
        except Exception as e:
            logger.warning(f"Failed to initialize audit crypto: {e}")
            self.enable_crypto_verification = False

    def _generate_audit_key(self) -> bytes:
        """Generate audit verification key"""
        return hashlib.sha256(f"{self.service_name}:{time.time()}".encode()).digest()

    def log_audit_entry(self, action: str, resource: str,
                       user_id: Optional[str] = None,
                       before_state: Optional[Dict[str, Any]] = None,
                       after_state: Optional[Dict[str, Any]] = None,
                       success: bool = True,
                       reason: Optional[str] = None,
                       compliance_standards: Optional[List[ComplianceStandard]] = None) -> str:
        """Log an audit entry with integrity verification"""

        audit_id = self._generate_audit_id()

        # Determine compliance standards
        if compliance_standards is None:
            compliance_standards = [ComplianceStandard.SOX]  # Default to SOX for audit

        # Classify data
        data_content = json.dumps({
            "before": before_state or {},
            "after": after_state or {}
        })
        data_classification = self._classify_audit_data(data_content)

        # Create audit entry
        entry = AuditEntry(
            audit_id=audit_id,
            timestamp=datetime.now(datetime.UTC),
            action=action,
            resource=resource,
            user_id=user_id,
            before_state=before_state,
            after_state=after_state,
            success=success,
            reason=reason,
            compliance_standards=compliance_standards,
            data_classification=data_classification,
            checksum=self._calculate_checksum(action, resource, user_id, before_state, after_state)
        )

        # Store entry
        with self.lock:
            self.audit_trail.append(entry)

        # Log the audit entry
        self._log_audit_entry(entry)

        return audit_id

    def _generate_audit_id(self) -> str:
        """Generate unique audit ID"""
        timestamp = str(int(time.time() * 1000000))
        random_part = hashlib.md5(str(time.time()).encode()).hexdigest()[:8]
        return f"audit_{timestamp}_{random_part}"

    def _classify_audit_data(self, data_content: str) -> DataClassification:
        """Classify audit data based on content"""
        checker = ComplianceChecker()
        return checker.classify_data(data_content)

    def _calculate_checksum(self, action: str, resource: str,
                           user_id: Optional[str], before_state: Optional[Dict],
                           after_state: Optional[Dict]) -> str:
        """Calculate cryptographic checksum for audit integrity"""
        data = f"{action}:{resource}:{user_id or ''}:{json.dumps(before_state or {})}:{json.dumps(after_state or {})}"

        if self.enable_crypto_verification and hasattr(self, 'audit_key'):
            return hmac.new(self.audit_key, data.encode(), hashlib.sha256).hexdigest()
        else:
            return hashlib.sha256(data.encode()).hexdigest()

    def _log_audit_entry(self, entry: AuditEntry):
        """Log audit entry using structured logger"""
        from .structured_logger import get_logger

        logger = get_logger()
        entry_dict = entry.to_dict()

        logger.log_structured(
            "info",
            f"Audit: {entry.action} on {entry.resource}",
            {
                "audit_event": True,
                "audit_details": entry_dict,
                "integrity_verified": self.enable_crypto_verification
            }
        )

    def verify_audit_integrity(self, audit_id: str) -> bool:
        """Verify the integrity of an audit entry"""
        with self.lock:
            for entry in self.audit_trail:
                if entry.audit_id == audit_id:
                    # Recalculate checksum
                    expected_checksum = self._calculate_checksum(
                        entry.action, entry.resource, entry.user_id,
                        entry.before_state, entry.after_state
                    )
                    return entry.checksum == expected_checksum

        return False

    def get_audit_trail(self, user_id: Optional[str] = None,
                       resource: Optional[str] = None,
                       time_window_days: int = 30) -> List[AuditEntry]:
        """Get audit trail with optional filtering"""
        cutoff_date = datetime.now(datetime.UTC) - timedelta(days=time_window_days)

        with self.lock:
            filtered_trail = [
                entry for entry in self.audit_trail
                if entry.timestamp > cutoff_date
            ]

            if user_id:
                filtered_trail = [entry for entry in filtered_trail if entry.user_id == user_id]

            if resource:
                filtered_trail = [entry for entry in filtered_trail if entry.resource == resource]

            return filtered_trail.copy()

    def generate_audit_report(self, time_window_days: int = 30) -> Dict[str, Any]:
        """Generate comprehensive audit report"""
        audit_trail = self.get_audit_trail(time_window_days=time_window_days)

        report = {
            "report_period_days": time_window_days,
            "total_entries": len(audit_trail),
            "successful_operations": len([e for e in audit_trail if e.success]),
            "failed_operations": len([e for e in audit_trail if not e.success]),
            "operations_by_user": {},
            "operations_by_resource": {},
            "operations_by_type": {},
            "compliance_coverage": {},
            "integrity_status": "verified" if self.enable_crypto_verification else "unverified"
        }

        # Analyze operations
        for entry in audit_trail:
            # By user
            user = entry.user_id or "system"
            report["operations_by_user"][user] = report["operations_by_user"].get(user, 0) + 1

            # By resource
            report["operations_by_resource"][entry.resource] = report["operations_by_resource"].get(entry.resource, 0) + 1

            # By type
            report["operations_by_type"][entry.action] = report["operations_by_type"].get(entry.action, 0) + 1

            # Compliance coverage
            for standard in entry.compliance_standards:
                std_name = standard.value
                report["compliance_coverage"][std_name] = report["compliance_coverage"].get(std_name, 0) + 1

        return report


# Global instances
_security_logger = None
_audit_logger = None


def get_security_logger() -> SecurityEventLogger:
    """Get or create global security logger"""
    global _security_logger
    if _security_logger is None:
        _security_logger = SecurityEventLogger()
    return _security_logger


def get_audit_logger() -> AuditLogger:
    """Get or create global audit logger"""
    global _audit_logger
    if _audit_logger is None:
        _audit_logger = AuditLogger()
    return _audit_logger


def initialize_security_logging(service_name: str = "fedzk",
                               enable_crypto_audit: bool = True) -> tuple:
    """
    Initialize security and audit logging
    Returns: (security_logger, audit_logger)
    """
    global _security_logger, _audit_logger

    _security_logger = SecurityEventLogger(service_name)
    _audit_logger = AuditLogger(service_name, enable_crypto_audit)

    logger.info(f"Initialized security and audit logging for {service_name}")

    return _security_logger, _audit_logger


# Utility functions for easy integration
def log_authentication_event(success: bool, user_id: str, source_ip: str,
                           user_agent: str = None, details: Dict[str, Any] = None):
    """Log authentication event"""
    security_logger = get_security_logger()

    event_type = (SecurityEventType.AUTHENTICATION_SUCCESS if success
                 else SecurityEventType.AUTHENTICATION_FAILURE)

    return security_logger.log_security_event(
        event_type=event_type,
        user_id=user_id,
        source_ip=source_ip,
        user_agent=user_agent,
        success=success,
        details=details
    )


def log_data_access(user_id: str, resource: str, action: str,
                   source_ip: str, success: bool = True):
    """Log data access event"""
    security_logger = get_security_logger()

    return security_logger.log_security_event(
        event_type=SecurityEventType.DATA_ACCESS,
        user_id=user_id,
        resource=resource,
        action=action,
        source_ip=source_ip,
        success=success
    )


def log_audit_change(action: str, resource: str, user_id: str,
                    before_state: Dict[str, Any] = None,
                    after_state: Dict[str, Any] = None,
                    success: bool = True):
    """Log audit change"""
    audit_logger = get_audit_logger()

    return audit_logger.log_audit_entry(
        action=action,
        resource=resource,
        user_id=user_id,
        before_state=before_state,
        after_state=after_state,
        success=success
    )


def check_compliance_violations(data: str) -> Dict[str, bool]:
    """Check data for compliance violations"""
    checker = ComplianceChecker()
    return checker.check_compliance(
        data,
        [ComplianceStandard.GDPR, ComplianceStandard.PCI_DSS, ComplianceStandard.HIPAA]
    )


def get_security_report(time_window_days: int = 7) -> Dict[str, Any]:
    """Generate comprehensive security report"""
    security_logger = get_security_logger()

    return {
        "security_events": security_logger.generate_compliance_report(time_window_days),
        "audit_trail": get_audit_logger().generate_audit_report(time_window_days)
    }
