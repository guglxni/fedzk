"""
Security Checklists for FEDZK

This module provides comprehensive security checklists and compliance
validation frameworks for audit preparation and ongoing compliance monitoring.
"""

import json
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path


class ChecklistStatus(Enum):
    """Status of checklist items"""
    NOT_STARTED = "not_started"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    NOT_APPLICABLE = "not_applicable"


class ChecklistCategory(Enum):
    """Categories of security checklists"""
    INFRASTRUCTURE = "infrastructure"
    APPLICATION = "application"
    DATA_SECURITY = "data_security"
    ACCESS_CONTROL = "access_control"
    MONITORING = "monitoring"
    COMPLIANCE = "compliance"
    INCIDENT_RESPONSE = "incident_response"


@dataclass
class ChecklistItem:
    """Represents a single checklist item"""
    id: str
    title: str
    description: str
    category: ChecklistCategory
    priority: str  # "Critical", "High", "Medium", "Low"
    status: ChecklistStatus = ChecklistStatus.NOT_STARTED
    evidence_required: bool = False
    evidence_path: Optional[str] = None
    assigned_to: Optional[str] = None
    due_date: Optional[datetime] = None
    completed_date: Optional[datetime] = None
    notes: str = ""
    verification_steps: List[str] = field(default_factory=list)
    references: List[str] = field(default_factory=list)


@dataclass
class SecurityChecklist:
    """Represents a complete security checklist"""
    name: str
    description: str
    version: str
    category: ChecklistCategory
    items: List[ChecklistItem] = field(default_factory=list)
    created_date: datetime = field(default_factory=datetime.now)
    last_updated: datetime = field(default_factory=datetime.now)
    overall_completion: float = 0.0


class SecurityChecklists:
    """
    Comprehensive security checklists for FEDZK audit preparation

    Provides checklists for various security domains and compliance requirements.
    """

    def __init__(self):
        self.checklists: Dict[str, SecurityChecklist] = {}
        self._initialize_checklists()

    def _initialize_checklists(self):
        """Initialize all security checklists"""
        self._create_infrastructure_security_checklist()
        self._create_application_security_checklist()
        self._create_data_security_checklist()
        self._create_access_control_checklist()
        self._create_monitoring_checklist()
        self._create_compliance_checklist()
        self._create_incident_response_checklist()

    def _create_infrastructure_security_checklist(self):
        """Create infrastructure security checklist"""
        checklist = SecurityChecklist(
            name="Infrastructure Security Checklist",
            description="Security controls for infrastructure components",
            version="1.0",
            category=ChecklistCategory.INFRASTRUCTURE
        )

        checklist.items = [
            ChecklistItem(
                id="INF001",
                title="Network Segmentation",
                description="Implement proper network segmentation between different security zones",
                category=ChecklistCategory.INFRASTRUCTURE,
                priority="High",
                evidence_required=True,
                verification_steps=[
                    "Review network architecture diagrams",
                    "Verify firewall rules and network policies",
                    "Test network isolation between zones",
                    "Document network segmentation strategy"
                ],
                references=["NIST SP 800-53", "ISO 27001 A.13.1.1"]
            ),
            ChecklistItem(
                id="INF002",
                title="Container Security",
                description="Implement security best practices for containerized applications",
                category=ChecklistCategory.INFRASTRUCTURE,
                priority="High",
                evidence_required=True,
                verification_steps=[
                    "Verify non-root user execution",
                    "Check for minimal base images",
                    "Validate security scanning results",
                    "Review resource limits and constraints"
                ],
                references=["NIST SP 800-190", "Docker Security Best Practices"]
            ),
            ChecklistItem(
                id="INF003",
                title="Infrastructure as Code Security",
                description="Secure infrastructure provisioning and configuration management",
                category=ChecklistCategory.INFRASTRUCTURE,
                priority="Medium",
                evidence_required=True,
                verification_steps=[
                    "Review IaC templates for security issues",
                    "Verify secrets management in IaC",
                    "Check for secure defaults",
                    "Validate configuration drift prevention"
                ],
                references=["CIS Benchmarks", "Infrastructure Security Best Practices"]
            ),
            ChecklistItem(
                id="INF004",
                title="Secure Configuration Management",
                description="Implement secure configuration management practices",
                category=ChecklistCategory.INFRASTRUCTURE,
                priority="High",
                evidence_required=True,
                verification_steps=[
                    "Review system hardening configurations",
                    "Verify secure service configurations",
                    "Check for unnecessary services disabled",
                    "Validate configuration management processes"
                ],
                references=["CIS Benchmarks", "NIST SP 800-53 SC-2"]
            ),
            ChecklistItem(
                id="INF005",
                title="Backup and Recovery Security",
                description="Secure backup and disaster recovery procedures",
                category=ChecklistCategory.INFRASTRUCTURE,
                priority="Medium",
                evidence_required=True,
                verification_steps=[
                    "Verify encrypted backups",
                    "Review backup integrity verification",
                    "Check secure backup storage",
                    "Validate recovery testing procedures"
                ],
                references=["NIST SP 800-53 RE-2", "ISO 27001 A.12.3.1"]
            )
        ]

        self.checklists["infrastructure_security"] = checklist

    def _create_application_security_checklist(self):
        """Create application security checklist"""
        checklist = SecurityChecklist(
            name="Application Security Checklist",
            description="Security controls for application development and deployment",
            version="1.0",
            category=ChecklistCategory.APPLICATION
        )

        checklist.items = [
            ChecklistItem(
                id="APP001",
                title="Input Validation and Sanitization",
                description="Implement comprehensive input validation and sanitization",
                category=ChecklistCategory.APPLICATION,
                priority="Critical",
                evidence_required=True,
                verification_steps=[
                    "Review input validation functions",
                    "Test boundary conditions",
                    "Verify sanitization of user inputs",
                    "Check for injection vulnerabilities"
                ],
                references=["OWASP Input Validation Cheat Sheet", "NIST SP 800-53 SI-10"]
            ),
            ChecklistItem(
                id="APP002",
                title="Authentication and Session Management",
                description="Implement secure authentication and session management",
                category=ChecklistCategory.APPLICATION,
                priority="Critical",
                evidence_required=True,
                verification_steps=[
                    "Review authentication mechanisms",
                    "Verify session timeout settings",
                    "Check secure cookie configuration",
                    "Validate password policies"
                ],
                references=["OWASP Authentication Cheat Sheet", "NIST SP 800-63"]
            ),
            ChecklistItem(
                id="APP003",
                title="Authorization Controls",
                description="Implement proper authorization and access controls",
                category=ChecklistCategory.APPLICATION,
                priority="High",
                evidence_required=True,
                verification_steps=[
                    "Review role-based access controls",
                    "Verify authorization checks",
                    "Test privilege escalation scenarios",
                    "Validate resource permissions"
                ],
                references=["OWASP Access Control Cheat Sheet", "NIST SP 800-53 AC-2"]
            ),
            ChecklistItem(
                id="APP004",
                title="Secure Error Handling",
                description="Implement secure error handling without information leakage",
                category=ChecklistCategory.APPLICATION,
                priority="Medium",
                evidence_required=True,
                verification_steps=[
                    "Review error handling code",
                    "Test error message content",
                    "Verify no sensitive data in errors",
                    "Check error logging security"
                ],
                references=["OWASP Error Handling Cheat Sheet", "NIST SP 800-53 SI-11"]
            ),
            ChecklistItem(
                id="APP005",
                title="Cryptographic Implementation",
                description="Implement secure cryptographic functions",
                category=ChecklistCategory.APPLICATION,
                priority="High",
                evidence_required=True,
                verification_steps=[
                    "Review cryptographic algorithm usage",
                    "Verify key management practices",
                    "Check for hardcoded secrets",
                    "Validate secure random generation"
                ],
                references=["NIST SP 800-57", "OWASP Cryptographic Storage Cheat Sheet"]
            ),
            ChecklistItem(
                id="APP006",
                title="Secure Dependencies",
                description="Manage third-party dependencies securely",
                category=ChecklistCategory.APPLICATION,
                priority="High",
                evidence_required=True,
                verification_steps=[
                    "Review dependency vulnerability scans",
                    "Verify dependency update process",
                    "Check for unused dependencies",
                    "Validate dependency integrity"
                ],
                references=["OWASP Dependency Check", "NIST SP 800-53 SA-8"]
            )
        ]

        self.checklists["application_security"] = checklist

    def _create_data_security_checklist(self):
        """Create data security checklist"""
        checklist = SecurityChecklist(
            name="Data Security Checklist",
            description="Security controls for data protection and privacy",
            version="1.0",
            category=ChecklistCategory.DATA_SECURITY
        )

        checklist.items = [
            ChecklistItem(
                id="DAT001",
                title="Data Classification",
                description="Implement data classification and handling procedures",
                category=ChecklistCategory.DATA_SECURITY,
                priority="High",
                evidence_required=True,
                verification_steps=[
                    "Review data classification policy",
                    "Verify data labeling procedures",
                    "Check classification accuracy",
                    "Validate handling procedures"
                ],
                references=["NIST SP 800-60", "ISO 27001 A.8.2.1"]
            ),
            ChecklistItem(
                id="DAT002",
                title="Encryption at Rest",
                description="Implement encryption for data at rest",
                category=ChecklistCategory.DATA_SECURITY,
                priority="Critical",
                evidence_required=True,
                verification_steps=[
                    "Review encryption algorithms used",
                    "Verify key management processes",
                    "Test encryption/decryption functionality",
                    "Validate backup encryption"
                ],
                references=["NIST SP 800-57", "ISO 27001 A.10.1.1"]
            ),
            ChecklistItem(
                id="DAT003",
                title="Encryption in Transit",
                description="Implement encryption for data in transit",
                category=ChecklistCategory.DATA_SECURITY,
                priority="Critical",
                evidence_required=True,
                verification_steps=[
                    "Review TLS configuration",
                    "Verify certificate management",
                    "Test secure communication channels",
                    "Validate protocol security"
                ],
                references=["NIST SP 800-52", "ISO 27001 A.10.1.2"]
            ),
            ChecklistItem(
                id="DAT004",
                title="Data Retention and Disposal",
                description="Implement secure data retention and disposal procedures",
                category=ChecklistCategory.DATA_SECURITY,
                priority="Medium",
                evidence_required=True,
                verification_steps=[
                    "Review data retention policies",
                    "Verify disposal procedures",
                    "Test secure deletion methods",
                    "Validate compliance with regulations"
                ],
                references=["NIST SP 800-88", "GDPR Article 17"]
            ),
            ChecklistItem(
                id="DAT005",
                title="Data Loss Prevention",
                description="Implement data loss prevention controls",
                category=ChecklistCategory.DATA_SECURITY,
                priority="High",
                evidence_required=True,
                verification_steps=[
                    "Review DLP policies and rules",
                    "Test data exfiltration prevention",
                    "Verify monitoring and alerting",
                    "Validate incident response procedures"
                ],
                references=["NIST SP 800-53 SC-4", "ISO 27001 A.13.2.1"]
            )
        ]

        self.checklists["data_security"] = checklist

    def _create_access_control_checklist(self):
        """Create access control checklist"""
        checklist = SecurityChecklist(
            name="Access Control Checklist",
            description="Security controls for access management and authorization",
            version="1.0",
            category=ChecklistCategory.ACCESS_CONTROL
        )

        checklist.items = [
            ChecklistItem(
                id="ACC001",
                title="Multi-Factor Authentication",
                description="Implement multi-factor authentication for all users",
                category=ChecklistCategory.ACCESS_CONTROL,
                priority="High",
                evidence_required=True,
                verification_steps=[
                    "Review MFA implementation",
                    "Test MFA functionality",
                    "Verify MFA enforcement",
                    "Check MFA bypass scenarios"
                ],
                references=["NIST SP 800-63", "ISO 27001 A.9.2.1"]
            ),
            ChecklistItem(
                id="ACC002",
                title="Role-Based Access Control",
                description="Implement role-based access control system",
                category=ChecklistCategory.ACCESS_CONTROL,
                priority="High",
                evidence_required=True,
                verification_steps=[
                    "Review RBAC implementation",
                    "Verify role definitions",
                    "Test access control enforcement",
                    "Validate least privilege principle"
                ],
                references=["NIST SP 800-53 AC-2", "ISO 27001 A.9.2.2"]
            ),
            ChecklistItem(
                id="ACC003",
                title="Access Review Process",
                description="Implement regular access review and cleanup procedures",
                category=ChecklistCategory.ACCESS_CONTROL,
                priority="Medium",
                evidence_required=True,
                verification_steps=[
                    "Review access review procedures",
                    "Verify review frequency",
                    "Check access revocation processes",
                    "Validate review documentation"
                ],
                references=["NIST SP 800-53 AC-2", "ISO 27001 A.9.2.5"]
            ),
            ChecklistItem(
                id="ACC004",
                title="Privileged Access Management",
                description="Secure management of privileged accounts and access",
                category=ChecklistCategory.ACCESS_CONTROL,
                priority="Critical",
                evidence_required=True,
                verification_steps=[
                    "Review privileged account management",
                    "Verify just-in-time access",
                    "Test privilege escalation controls",
                    "Validate audit logging"
                ],
                references=["NIST SP 800-53 AC-6", "ISO 27001 A.9.2.3"]
            ),
            ChecklistItem(
                id="ACC005",
                title="Account Lifecycle Management",
                description="Implement secure account lifecycle management",
                category=ChecklistCategory.ACCESS_CONTROL,
                priority="Medium",
                evidence_required=True,
                verification_steps=[
                    "Review account creation procedures",
                    "Verify account deactivation processes",
                    "Check automated account cleanup",
                    "Validate account audit trails"
                ],
                references=["NIST SP 800-53 AC-2", "ISO 27001 A.9.2.4"]
            )
        ]

        self.checklists["access_control"] = checklist

    def _create_monitoring_checklist(self):
        """Create monitoring and logging checklist"""
        checklist = SecurityChecklist(
            name="Monitoring and Logging Checklist",
            description="Security monitoring and audit logging controls",
            version="1.0",
            category=ChecklistCategory.MONITORING
        )

        checklist.items = [
            ChecklistItem(
                id="MON001",
                title="Security Event Monitoring",
                description="Implement comprehensive security event monitoring",
                category=ChecklistCategory.MONITORING,
                priority="High",
                evidence_required=True,
                verification_steps=[
                    "Review monitoring systems",
                    "Verify security event detection",
                    "Test alerting mechanisms",
                    "Validate event correlation"
                ],
                references=["NIST SP 800-53 SI-4", "ISO 27001 A.12.4.1"]
            ),
            ChecklistItem(
                id="MON002",
                title="Audit Logging",
                description="Implement comprehensive audit logging",
                category=ChecklistCategory.MONITORING,
                priority="Critical",
                evidence_required=True,
                verification_steps=[
                    "Review audit log configuration",
                    "Verify log integrity protection",
                    "Test log collection and analysis",
                    "Validate log retention policies"
                ],
                references=["NIST SP 800-53 AU-2", "ISO 27001 A.12.4.2"]
            ),
            ChecklistItem(
                id="MON003",
                title="Intrusion Detection",
                description="Implement intrusion detection and prevention systems",
                category=ChecklistCategory.MONITORING,
                priority="High",
                evidence_required=True,
                verification_steps=[
                    "Review IDS/IPS implementation",
                    "Test intrusion detection capabilities",
                    "Verify false positive rates",
                    "Validate incident response integration"
                ],
                references=["NIST SP 800-53 SI-4", "ISO 27001 A.12.4.3"]
            ),
            ChecklistItem(
                id="MON004",
                title="Performance Monitoring",
                description="Implement performance monitoring and alerting",
                category=ChecklistCategory.MONITORING,
                priority="Medium",
                evidence_required=True,
                verification_steps=[
                    "Review performance monitoring setup",
                    "Verify threshold configurations",
                    "Test alerting mechanisms",
                    "Validate monitoring coverage"
                ],
                references=["NIST SP 800-53 PE-6", "ISO 27001 A.12.1.3"]
            ),
            ChecklistItem(
                id="MON005",
                title="Log Analysis and Reporting",
                description="Implement log analysis and security reporting",
                category=ChecklistCategory.MONITORING,
                priority="Medium",
                evidence_required=True,
                verification_steps=[
                    "Review log analysis procedures",
                    "Verify reporting capabilities",
                    "Test automated report generation",
                    "Validate compliance reporting"
                ],
                references=["NIST SP 800-53 AU-6", "ISO 27001 A.12.4.2"]
            )
        ]

        self.checklists["monitoring"] = checklist

    def _create_compliance_checklist(self):
        """Create compliance checklist"""
        checklist = SecurityChecklist(
            name="Compliance Checklist",
            description="Regulatory and industry compliance requirements",
            version="1.0",
            category=ChecklistCategory.COMPLIANCE
        )

        checklist.items = [
            ChecklistItem(
                id="COM001",
                title="GDPR Compliance",
                description="Ensure compliance with GDPR requirements",
                category=ChecklistCategory.COMPLIANCE,
                priority="Critical",
                evidence_required=True,
                verification_steps=[
                    "Review data processing activities",
                    "Verify privacy notices",
                    "Test data subject rights",
                    "Validate consent mechanisms"
                ],
                references=["GDPR Articles 12-23", "ICO Guidelines"]
            ),
            ChecklistItem(
                id="COM002",
                title="CCPA Compliance",
                description="Ensure compliance with CCPA requirements",
                category=ChecklistCategory.COMPLIANCE,
                priority="High",
                evidence_required=True,
                verification_steps=[
                    "Review personal information handling",
                    "Verify privacy policy compliance",
                    "Test consumer rights implementation",
                    "Validate data minimization practices"
                ],
                references=["California Civil Code §1798.100 et seq."]
            ),
            ChecklistItem(
                id="COM003",
                title="NIST Cybersecurity Framework",
                description="Implement NIST CSF controls",
                category=ChecklistCategory.COMPLIANCE,
                priority="High",
                evidence_required=True,
                verification_steps=[
                    "Review NIST CSF implementation",
                    "Verify identify function",
                    "Test protect function",
                    "Validate detect and respond functions"
                ],
                references=["NIST Cybersecurity Framework v1.1"]
            ),
            ChecklistItem(
                id="COM004",
                title="ISO 27001 Alignment",
                description="Align with ISO 27001 requirements",
                category=ChecklistCategory.COMPLIANCE,
                priority="Medium",
                evidence_required=True,
                verification_steps=[
                    "Review ISO 27001 controls",
                    "Verify information security policies",
                    "Test risk management processes",
                    "Validate continuous improvement"
                ],
                references=["ISO 27001:2022"]
            ),
            ChecklistItem(
                id="COM005",
                title="SOC 2 Readiness",
                description="Prepare for SOC 2 Type II audit",
                category=ChecklistCategory.COMPLIANCE,
                priority="Medium",
                evidence_required=True,
                verification_steps=[
                    "Review trust service criteria",
                    "Verify control activities",
                    "Test monitoring procedures",
                    "Validate evidence collection"
                ],
                references=["AICPA TSP Section 100"]
            )
        ]

        self.checklists["compliance"] = checklist

    def _create_incident_response_checklist(self):
        """Create incident response checklist"""
        checklist = SecurityChecklist(
            name="Incident Response Checklist",
            description="Incident response and business continuity controls",
            version="1.0",
            category=ChecklistCategory.INCIDENT_RESPONSE
        )

        checklist.items = [
            ChecklistItem(
                id="INC001",
                title="Incident Response Plan",
                description="Develop and maintain incident response plan",
                category=ChecklistCategory.INCIDENT_RESPONSE,
                priority="Critical",
                evidence_required=True,
                verification_steps=[
                    "Review incident response plan",
                    "Verify contact information",
                    "Test communication procedures",
                    "Validate plan currency"
                ],
                references=["NIST SP 800-61", "ISO 27001 A.16.1.5"]
            ),
            ChecklistItem(
                id="INC002",
                title="Incident Detection and Analysis",
                description="Implement incident detection and analysis capabilities",
                category=ChecklistCategory.INCIDENT_RESPONSE,
                priority="High",
                evidence_required=True,
                verification_steps=[
                    "Review detection mechanisms",
                    "Verify analysis procedures",
                    "Test incident classification",
                    "Validate escalation processes"
                ],
                references=["NIST SP 800-61", "ISO 27001 A.16.1.2"]
            ),
            ChecklistItem(
                id="INC003",
                title="Incident Containment and Eradication",
                description="Implement incident containment and eradication procedures",
                category=ChecklistCategory.INCIDENT_RESPONSE,
                priority="High",
                evidence_required=True,
                verification_steps=[
                    "Review containment strategies",
                    "Verify eradication procedures",
                    "Test recovery processes",
                    "Validate evidence preservation"
                ],
                references=["NIST SP 800-61", "ISO 27001 A.16.1.3"]
            ),
            ChecklistItem(
                id="INC004",
                title="Business Continuity Planning",
                description="Develop business continuity and disaster recovery plans",
                category=ChecklistCategory.INCIDENT_RESPONSE,
                priority="Medium",
                evidence_required=True,
                verification_steps=[
                    "Review business impact analysis",
                    "Verify recovery strategies",
                    "Test backup procedures",
                    "Validate plan testing frequency"
                ],
                references=["NIST SP 800-34", "ISO 27001 A.17.1.1"]
            ),
            ChecklistItem(
                id="INC005",
                title="Post-Incident Analysis",
                description="Implement post-incident analysis and improvement processes",
                category=ChecklistCategory.INCIDENT_RESPONSE,
                priority="Medium",
                evidence_required=True,
                verification_steps=[
                    "Review incident analysis procedures",
                    "Verify lessons learned process",
                    "Test plan update mechanisms",
                    "Validate continuous improvement"
                ],
                references=["NIST SP 800-61", "ISO 27001 A.16.1.6"]
            )
        ]

        self.checklists["incident_response"] = checklist

    def get_checklist(self, checklist_id: str) -> Optional[SecurityChecklist]:
        """Get a specific checklist by ID"""
        return self.checklists.get(checklist_id)

    def get_all_checklists(self) -> Dict[str, SecurityChecklist]:
        """Get all available checklists"""
        return self.checklists.copy()

    def update_checklist_item(self, checklist_id: str, item_id: str,
                            status: ChecklistStatus, notes: str = ""):
        """Update the status of a checklist item"""
        if checklist_id in self.checklists:
            checklist = self.checklists[checklist_id]
            for item in checklist.items:
                if item.id == item_id:
                    item.status = status
                    item.notes = notes
                    if status == ChecklistStatus.COMPLETED:
                        item.completed_date = datetime.now()
                    checklist.last_updated = datetime.now()
                    self._update_checklist_completion(checklist)
                    break

    def _update_checklist_completion(self, checklist: SecurityChecklist):
        """Update the completion percentage of a checklist"""
        if not checklist.items:
            checklist.overall_completion = 100.0
            return

        completed_items = sum(1 for item in checklist.items
                            if item.status == ChecklistStatus.COMPLETED)
        checklist.overall_completion = (completed_items / len(checklist.items)) * 100

    def get_completion_summary(self) -> Dict[str, Any]:
        """Get overall completion summary across all checklists"""
        total_items = 0
        completed_items = 0
        category_summary = {}

        for checklist in self.checklists.values():
            total_items += len(checklist.items)
            completed_items += sum(1 for item in checklist.items
                                 if item.status == ChecklistStatus.COMPLETED)

            category = checklist.category.value
            if category not in category_summary:
                category_summary[category] = {'total': 0, 'completed': 0}
            category_summary[category]['total'] += len(checklist.items)
            category_summary[category]['completed'] += sum(
                1 for item in checklist.items
                if item.status == ChecklistStatus.COMPLETED
            )

        overall_completion = (completed_items / total_items * 100) if total_items > 0 else 0

        return {
            'overall_completion': overall_completion,
            'total_items': total_items,
            'completed_items': completed_items,
            'pending_items': total_items - completed_items,
            'category_summary': category_summary
        }

    def export_checklists(self, output_format: str = 'json') -> str:
        """Export all checklists in specified format"""
        checklists_data = {}
        for checklist_id, checklist in self.checklists.items():
            checklists_data[checklist_id] = {
                'name': checklist.name,
                'description': checklist.description,
                'version': checklist.version,
                'category': checklist.category.value,
                'overall_completion': checklist.overall_completion,
                'items': [
                    {
                        'id': item.id,
                        'title': item.title,
                        'description': item.description,
                        'category': item.category.value,
                        'priority': item.priority,
                        'status': item.status.value,
                        'evidence_required': item.evidence_required,
                        'verification_steps': item.verification_steps,
                        'references': item.references,
                        'notes': item.notes
                    }
                    for item in checklist.items
                ]
            }

        if output_format == 'json':
            return json.dumps(checklists_data, indent=2, default=str)
        else:
            raise ValueError(f"Unsupported format: {output_format}")

    def get_high_priority_items(self) -> List[Dict[str, Any]]:
        """Get all high priority checklist items that are not completed"""
        high_priority_items = []

        for checklist in self.checklists.values():
            for item in checklist.items:
                if (item.priority in ['Critical', 'High'] and
                    item.status != ChecklistStatus.COMPLETED):
                    high_priority_items.append({
                        'checklist': checklist.name,
                        'item_id': item.id,
                        'title': item.title,
                        'priority': item.priority,
                        'status': item.status.value,
                        'description': item.description
                    })

        return high_priority_items

    def generate_compliance_report(self) -> Dict[str, Any]:
        """Generate a comprehensive compliance report"""
        summary = self.get_completion_summary()
        high_priority_items = self.get_high_priority_items()

        return {
            'report_generated': datetime.now().isoformat(),
            'compliance_overview': summary,
            'high_priority_action_items': high_priority_items,
            'recommendations': self._generate_compliance_recommendations(summary, high_priority_items),
            'next_steps': [
                "Complete high priority security items",
                "Conduct regular compliance reviews",
                "Maintain detailed audit documentation",
                "Implement continuous compliance monitoring",
                "Prepare for external audits and assessments"
            ]
        }

    def _generate_compliance_recommendations(self, summary: Dict[str, Any],
                                           high_priority_items: List[Dict[str, Any]]) -> List[str]:
        """Generate compliance recommendations based on current status"""
        recommendations = []

        overall_completion = summary['overall_completion']

        if overall_completion < 50:
            recommendations.append("CRITICAL: Immediate action required - overall compliance below 50%")
        elif overall_completion < 75:
            recommendations.append("HIGH: Improve compliance - focus on remaining critical items")
        elif overall_completion < 90:
            recommendations.append("MEDIUM: Good progress - complete remaining items for full compliance")

        if len(high_priority_items) > 10:
            recommendations.append("Address high priority items immediately - excessive backlog identified")
        elif len(high_priority_items) > 5:
            recommendations.append("Focus on high priority items to improve compliance posture")

        # Category-specific recommendations
        category_completion = summary.get('category_summary', {})
        for category, stats in category_completion.items():
            completion_rate = (stats['completed'] / stats['total'] * 100) if stats['total'] > 0 else 0
            if completion_rate < 70:
                recommendations.append(f"Improve {category} category compliance (currently {completion_rate:.1f}%)")

        return recommendations
