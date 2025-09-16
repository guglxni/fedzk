#!/usr/bin/env python3
"""
FEDzk Logging Infrastructure Demo
==================================

Demonstrates the comprehensive logging infrastructure with:
- Structured JSON logging
- Log aggregation and analysis
- Security and compliance logging
- Audit trail generation
"""

import sys
import json
import time
import logging
from datetime import datetime
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

try:
    from fedzk.logging.structured_logger import (
        FEDzkLogger, FEDzkJSONFormatter, FEDzkSecurityFormatter,
        get_logger, initialize_logging
    )
    from fedzk.logging.log_aggregation import (
        LogAggregator, ElasticsearchLogShipper,
        get_log_pipeline, initialize_log_aggregation
    )
    from fedzk.logging.security_compliance import (
        SecurityEventLogger, AuditLogger, SecurityEventType,
        ComplianceStandard, DataClassification,
        get_security_logger, get_audit_logger,
        log_authentication_event, log_audit_change
    )
    LOGGING_AVAILABLE = True
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Some logging dependencies may not be available.")
    LOGGING_AVAILABLE = False


def demo_structured_logging():
    """Demonstrate structured JSON logging"""
    print("📝 DEMO: Structured JSON Logging")
    print("=" * 50)

    if not LOGGING_AVAILABLE:
        print("⚠️  Structured logging not available - using mock demonstration")
        demo_mock_structured_logging()
        return

    # Initialize logger
    logger = FEDzkLogger("fedzk-demo", "INFO")

    print("✅ Initialized FEDzk structured logger")

    # Log various types of messages
    logger.logger.info("Application started", extra={"structured_data": {"version": "1.0.0"}})

    # Log with request context
    logger.set_request_context(request_id="req-12345", user_id="user-67890")
    logger.log_structured("info", "Processing user request",
                         {"action": "login", "method": "POST"})

    # Log performance metrics
    logger.log_performance_metric("database_query", 0.245, True,
                                 {"query_type": "SELECT", "table": "users"})

    # Log security event
    logger.log_security_event("authentication_success", "low",
                            {"method": "jwt", "user_id": "user-67890"})

    # Log audit event
    logger.log_audit_event("USER_LOGIN", "user_sessions", "user-67890", True,
                          {"ip_address": "192.168.1.100"})

    print("📋 Logged various structured messages:")
    print("   • Application startup with version info")
    print("   • User request with context")
    print("   • Performance metrics")
    print("   • Security events")
    print("   • Audit trail entries")
    print()

    logger.clear_request_context()


def demo_mock_structured_logging():
    """Mock demonstration of structured logging"""
    print("📝 Mock Structured Logging Demo:")

    # Sample JSON log entries
    log_samples = [
        {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "service": "fedzk-demo",
            "level": "INFO",
            "message": "User login successful",
            "structured_data": {
                "user_id": "user-123",
                "action": "login",
                "method": "POST",
                "request_id": "req-abc123"
            }
        },
        {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "service": "fedzk-demo",
            "level": "WARNING",
            "message": "High memory usage detected",
            "structured_data": {
                "memory_percent": 85.5,
                "threshold": 80.0,
                "component": "coordinator"
            }
        },
        {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "service": "fedzk-demo",
            "level": "ERROR",
            "message": "Database connection failed",
            "exception": "ConnectionTimeout: Connection to database timed out",
            "structured_data": {
                "database_host": "db.internal",
                "timeout_seconds": 30,
                "retry_count": 3
            }
        }
    ]

    for log_entry in log_samples:
        print(f"   {json.dumps(log_entry, indent=2)[:200]}...")

    print()


def demo_log_aggregation():
    """Demonstrate log aggregation and analysis"""
    print("🔍 DEMO: Log Aggregation & Analysis")
    print("=" * 50)

    if not LOGGING_AVAILABLE:
        print("⚠️  Log aggregation not available - using mock demonstration")
        demo_mock_aggregation()
        return

    # Initialize aggregator
    aggregator = LogAggregator(retention_hours=1)

    print("✅ Initialized log aggregator")

    # Simulate log entries
    sample_logs = [
        {"level": "INFO", "message": "User login", "service": "auth", "timestamp_epoch": time.time()},
        {"level": "INFO", "message": "Database query", "service": "db", "timestamp_epoch": time.time()},
        {"level": "WARNING", "message": "High memory usage", "service": "coordinator", "timestamp_epoch": time.time()},
        {"level": "ERROR", "message": "Connection timeout", "service": "db", "timestamp_epoch": time.time()},
        {"level": "ERROR", "message": "Authentication failed", "service": "auth", "timestamp_epoch": time.time()},
        {"level": "INFO", "message": "ZK proof generated", "service": "zk", "timestamp_epoch": time.time()},
    ]

    print("📊 Aggregating sample log entries...")
    for log_entry in sample_logs:
        aggregator.aggregate_log(log_entry)
        time.sleep(0.01)  # Small delay for timestamps

    # Get statistics
    stats = aggregator.get_statistics(1)  # Last hour

    print("📈 Aggregation Statistics:")
    print(f"   • Total logs: {stats['total_logs']}")
    print("   • Logs by level:")
    for level, count in stats['logs_by_level'].items():
        print(f"     - {level}: {count}")
    print("   • Logs by service:")
    for service, count in stats['logs_by_service'].items():
        print(f"     - {service}: {count}")

    # Show error patterns
    if aggregator.error_patterns:
        print("   • Error patterns detected:")
        for pattern, count in aggregator.error_patterns.items():
            print(f"     - {pattern}: {count}")

    # Show anomalies
    anomalies = aggregator._detect_anomalies(time.time() - 3600)
    if anomalies:
        print(f"   • Anomalies detected: {len(anomalies)}")
        for anomaly in anomalies[:2]:  # Show first 2
            print(f"     - {anomaly['type']}: {anomaly['description']}")

    print()


def demo_mock_aggregation():
    """Mock demonstration of log aggregation"""
    print("🔍 Mock Log Aggregation Demo:")

    mock_stats = {
        "total_logs": 1250,
        "logs_by_level": {"INFO": 850, "WARNING": 150, "ERROR": 200, "CRITICAL": 50},
        "logs_by_service": {"coordinator": 400, "mpc": 350, "zk": 300, "auth": 200},
        "error_patterns": {"connection_error": 45, "timeout_error": 32, "auth_error": 28},
        "performance_insights": {
            "avg_response_time": 0.245,
            "error_rate": 0.032,
            "throughput": 125.5
        }
    }

    print(f"   📊 Total logs processed: {mock_stats['total_logs']}")
    print("   📋 Breakdown by level:")
    for level, count in mock_stats['logs_by_level'].items():
        print(f"      {level}: {count}")

    print("   🔧 Breakdown by service:")
    for service, count in mock_stats['logs_by_service'].items():
        print(f"      {service}: {count}")

    print("   ⚠️  Error patterns detected:")
    for pattern, count in mock_stats['error_patterns'].items():
        print(f"      {pattern}: {count}")

    print("   📈 Performance insights:")
    perf = mock_stats['performance_insights']
    print(f"      Avg response time: {perf['avg_response_time']:.3f}s")
    print(f"      Error rate: {perf['error_rate']:.1%}")
    print(f"      Throughput: {perf['throughput']:.1f} req/s")

    print()


def demo_security_compliance():
    """Demonstrate security and compliance logging"""
    print("🔒 DEMO: Security & Compliance Logging")
    print("=" * 50)

    if not LOGGING_AVAILABLE:
        print("⚠️  Security logging not available - using mock demonstration")
        demo_mock_security()
        return

    # Initialize security logger
    security_logger = SecurityEventLogger("fedzk-security-demo")
    audit_logger = AuditLogger("fedzk-security-demo")

    print("✅ Initialized security and audit loggers")

    # Log various security events
    print("🚨 Logging security events...")

    # Authentication events
    auth_event_1 = security_logger.log_security_event(
        event_type=SecurityEventType.AUTHENTICATION_SUCCESS,
        user_id="user-123",
        source_ip="192.168.1.100",
        user_agent="Mozilla/5.0...",
        success=True,
        details={"method": "jwt", "session_duration": 3600}
    )

    auth_event_2 = security_logger.log_security_event(
        event_type=SecurityEventType.AUTHENTICATION_FAILURE,
        user_id="user-456",
        source_ip="10.0.0.50",
        success=False,
        details={"method": "password", "reason": "invalid_credentials"}
    )

    # Data access events
    data_event = security_logger.log_security_event(
        event_type=SecurityEventType.DATA_ACCESS,
        user_id="user-123",
        resource="user_profiles",
        action="READ",
        source_ip="192.168.1.100",
        success=True,
        details={"record_count": 25, "query_type": "SELECT"}
    )

    # Security violations
    violation_event = security_logger.log_security_event(
        event_type=SecurityEventType.SECURITY_VIOLATION,
        user_id="user-789",
        source_ip="203.0.113.1",
        resource="admin_panel",
        action="ACCESS",
        success=False,
        details={"violation_type": "unauthorized_access", "severity": "high"}
    )

    print("🔐 Logged security events:")
    print(f"   • Authentication success: {auth_event_1}")
    print(f"   • Authentication failure: {auth_event_2}")
    print(f"   • Data access: {data_event}")
    print(f"   • Security violation: {violation_event}")

    # Log audit events
    print("\n📋 Logging audit events...")

    audit_1 = audit_logger.log_audit_entry(
        action="USER_LOGIN",
        resource="sessions",
        user_id="user-123",
        success=True,
        before_state=None,
        after_state={"status": "active", "login_time": datetime.utcnow().isoformat()}
    )

    audit_2 = audit_logger.log_audit_entry(
        action="PROFILE_UPDATE",
        resource="user_profiles",
        user_id="user-123",
        success=True,
        before_state={"email": "old@example.com"},
        after_state={"email": "new@example.com"},
        compliance_standards=[ComplianceStandard.GDPR]
    )

    print("📝 Logged audit events:")
    print(f"   • User login: {audit_1}")
    print(f"   • Profile update: {audit_2}")

    # Generate compliance report
    compliance_report = security_logger.generate_compliance_report(7)
    print("\n📊 Compliance Report (last 7 days):")
    print(f"   • Total events: {compliance_report['total_events']}")
    print(f"   • Compliance violations: {sum(std['violations'] for std in compliance_report['compliance_summary'].values())}")

    print()


def demo_mock_security():
    """Mock demonstration of security logging"""
    print("🔒 Mock Security & Compliance Demo:")

    mock_events = [
        {
            "event_type": "authentication_success",
            "user_id": "user-123",
            "source_ip": "192.168.1.100",
            "compliance": {"gdpr": True, "pci_dss": True, "hipaa": True},
            "severity": "info"
        },
        {
            "event_type": "authentication_failure",
            "user_id": "user-456",
            "source_ip": "10.0.0.50",
            "compliance": {"gdpr": True, "pci_dss": False, "hipaa": True},
            "severity": "medium"
        },
        {
            "event_type": "data_access",
            "user_id": "user-123",
            "resource": "user_profiles",
            "action": "READ",
            "severity": "low"
        },
        {
            "event_type": "security_violation",
            "user_id": "user-789",
            "violation_type": "unauthorized_access",
            "severity": "high"
        }
    ]

    mock_audit = [
        {
            "action": "USER_LOGIN",
            "resource": "sessions",
            "user_id": "user-123",
            "success": True,
            "compliance_standards": ["gdpr", "sox"]
        },
        {
            "action": "PROFILE_UPDATE",
            "resource": "user_profiles",
            "user_id": "user-123",
            "success": True,
            "changes": {"email": "old@example.com → new@example.com"},
            "compliance_standards": ["gdpr"]
        }
    ]

    print("   🚨 Security Events:")
    for event in mock_events:
        severity_icon = {"info": "ℹ️", "low": "🟢", "medium": "🟡", "high": "🔴"}.get(event.get("severity"), "❓")
        print(f"      {severity_icon} {event['event_type']}")

    print("   📋 Audit Events:")
    for audit in mock_audit:
        status_icon = "✅" if audit['success'] else "❌"
        print(f"      {status_icon} {audit['action']} on {audit['resource']}")

    print("   📊 Compliance Status:")
    print("      ✅ GDPR: 2/2 compliant")
    print("      ⚠️  PCI DSS: 1/2 compliant")
    print("      ✅ HIPAA: 2/2 compliant")

    print()


def demo_helm_configuration():
    """Demonstrate Helm logging configuration"""
    print("⚓ DEMO: Helm Logging Configuration")
    print("=" * 50)

    print("🔧 Sample Helm values for logging:")
    print("""
# Logging Configuration
logging:
  enabled: true
  level: "INFO"

  # Structured logging
  structured:
    enabled: true
    serviceName: "fedzk"

  # File logging
  file:
    enabled: true
    path: "/var/log/fedzk/app.log"
    maxSize: 10485760
    backupCount: 5

  # Security logging
  security:
    enabled: true
    auditTrail: true
    complianceCheck: true

  # Log aggregation
  aggregation:
    enabled: true
    bufferSize: 10000
    flushInterval: 30

  # External systems
  elasticsearch:
    enabled: true
    host: "elasticsearch-master"
    port: 9200
    username: "elastic"
    password: "changeme"
    ssl: true
    indexPrefix: "fedzk-logs"
    """)

    print("📋 Generated Kubernetes resources:")
    print("   • ConfigMap: fedzk-logging-config")
    print("   • Volume mounts for log files")
    print("   • Environment variables for logging")
    print("   • ServiceMonitor for log metrics")
    print()

    print("🚀 Deployment commands:")
    print("   # Deploy with logging enabled")
    print("   helm install fedzk ./helm/fedzk \\")
    print("     --set logging.enabled=true \\")
    print("     --set logging.elasticsearch.enabled=true")
    print()


def demo_integration_example():
    """Show complete integration example"""
    print("🔗 DEMO: Complete Logging Integration")
    print("=" * 50)

    print("📝 Example application integration:")

    integration_code = '''
# Initialize logging system
from fedzk.logging import (
    get_logger, get_security_logger, get_audit_logger,
    initialize_log_aggregation
)

# Initialize loggers
logger = get_logger("my-service")
security_logger = get_security_logger()
audit_logger = get_audit_logger()

# Initialize aggregation
pipeline = initialize_log_aggregation(
    elasticsearch_hosts=["elasticsearch:9200"]
)

# Log application events
logger.info("Application started", extra={
    "structured_data": {"version": "1.0.0"}
})

# Log with context
logger.set_request_context(
    request_id="req-123",
    user_id="user-456",
    source_ip="192.168.1.100"
)

logger.log_structured("info", "Processing request",
                     {"action": "process_data", "size": 1024})

# Security logging
security_logger.log_security_event(
    SecurityEventType.AUTHENTICATION_SUCCESS,
    user_id="user-456",
    source_ip="192.168.1.100",
    success=True
)

# Audit logging
audit_logger.log_audit_entry(
    action="DATA_PROCESSING",
    resource="user_data",
    user_id="user-456",
    success=True,
    compliance_standards=[ComplianceStandard.GDPR]
)

# Performance logging
logger.log_performance_metric(
    "data_processing", 0.245, True,
    {"records_processed": 1000}
)

# Clean up context
logger.clear_request_context()
'''

    print(integration_code)

    print("🎯 Integration Benefits:")
    print("   ✅ Structured JSON logs for easy parsing")
    print("   ✅ Security event tracking and compliance")
    print("   ✅ Audit trail for regulatory requirements")
    print("   ✅ Performance monitoring and alerting")
    print("   ✅ Elasticsearch integration for advanced analytics")
    print("   ✅ Kubernetes-native deployment configuration")
    print()


def main():
    """Run all logging infrastructure demos"""
    print("📋 FEDzk Logging Infrastructure Demo")
    print("=" * 70)
    print("This demo showcases the comprehensive logging system with:")
    print("• Structured JSON logging with metadata")
    print("• Log aggregation and analysis")
    print("• Security and compliance logging")
    print("• Audit trail generation")
    print("• Helm integration and deployment")
    print()

    try:
        # Run all demos
        demo_structured_logging()
        demo_log_aggregation()
        demo_security_compliance()
        demo_helm_configuration()
        demo_integration_example()

        print("=" * 70)
        print("🎉 LOGGING INFRASTRUCTURE DEMO COMPLETED!")
        print("=" * 70)
        print()
        print("📋 IMPLEMENTATION SUMMARY:")
        print("✅ Structured JSON logging with metadata")
        print("✅ Log aggregation and real-time analysis")
        print("✅ Security event logging and correlation")
        print("✅ Audit trail generation and verification")
        print("✅ Compliance checking (GDPR, PCI DSS, HIPAA, SOX)")
        print("✅ Helm chart integration and configuration")
        print("✅ Elasticsearch and Kibana integration")
        print("✅ Kubernetes-native deployment support")
        print()
        print("🔧 PRODUCTION FEATURES:")
        print("• Log retention and rotation policies")
        print("• Real-time anomaly detection")
        print("• Performance metrics and alerting")
        print("• Multi-level security classification")
        print("• Cryptographic audit trail verification")
        print("• ELK stack integration for advanced analytics")
        print()
        print("🚀 READY FOR ENTERPRISE LOGGING!")

    except KeyboardInterrupt:
        print("\n👋 Demo interrupted by user")
    except Exception as e:
        print(f"\n❌ Demo error: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
