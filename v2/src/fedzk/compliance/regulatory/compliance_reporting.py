"""
Compliance Reporting Framework

This module provides automated compliance reporting and documentation
capabilities for FEDZK regulatory compliance.
"""

import json
import csv
import hashlib
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class ReportFormat(Enum):
    """Supported report formats"""
    JSON = "json"
    CSV = "csv"
    PDF = "pdf"
    HTML = "html"


class ReportType(Enum):
    """Types of compliance reports"""
    AUDIT_REPORT = "audit_report"
    COMPLIANCE_DASHBOARD = "compliance_dashboard"
    RISK_ASSESSMENT = "risk_assessment"
    INCIDENT_REPORT = "incident_report"
    ANNUAL_COMPLIANCE = "annual_compliance"


@dataclass
class ReportTemplate:
    """Represents a report template"""
    id: str
    name: str
    description: str
    report_type: ReportType
    format: ReportFormat
    sections: List[str]
    required_data: List[str]
    created_date: datetime
    last_modified: datetime


@dataclass
class GeneratedReport:
    """Represents a generated report"""
    report_id: str
    template_id: str
    title: str
    description: str
    generated_date: datetime
    report_period: str
    data_sources: List[str]
    content: Dict[str, Any]
    format: ReportFormat
    checksum: str
    status: str


@dataclass
class ComplianceDashboard:
    """Compliance dashboard data"""
    dashboard_id: str
    title: str
    description: str
    created_date: datetime
    last_updated: datetime
    widgets: List[Dict[str, Any]]
    data_sources: List[str]
    refresh_interval: int  # minutes


class ComplianceReporting:
    """
    Compliance reporting and documentation framework

    Provides automated report generation, template management,
    and compliance documentation capabilities.
    """

    def __init__(self, organization_name: str = "FEDZK", output_directory: str = None):
        self.organization = organization_name
        self.output_directory = Path(output_directory or f"./compliance_reports/{datetime.now().strftime('%Y%m%d')}")
        self.output_directory.mkdir(parents=True, exist_ok=True)
        self._report_templates: Dict[str, ReportTemplate] = {}
        self._generated_reports: List[GeneratedReport] = []
        self._dashboards: List[ComplianceDashboard] = []

    def create_report_template(self, name: str, description: str,
                             report_type: ReportType, format: ReportFormat,
                             sections: List[str], required_data: List[str]) -> ReportTemplate:
        """
        Create a new report template

        Args:
            name: Template name
            description: Template description
            report_type: Type of report
            format: Report format
            sections: Report sections
            required_data: Required data fields

        Returns:
            ReportTemplate: Created template
        """
        import uuid

        template_id = f"template_{uuid.uuid4().hex[:8]}"

        template = ReportTemplate(
            id=template_id,
            name=name,
            description=description,
            report_type=report_type,
            format=format,
            sections=sections,
            required_data=required_data,
            created_date=datetime.now(),
            last_modified=datetime.now()
        )

        self._report_templates[template_id] = template
        return template

    def generate_audit_report(self, audit_data: Dict[str, Any],
                            template_id: str = None) -> GeneratedReport:
        """
        Generate an audit report

        Args:
            audit_data: Audit data to include in report
            template_id: Optional template ID to use

        Returns:
            GeneratedReport: Generated audit report
        """
        import uuid

        report_id = f"audit_{uuid.uuid4().hex[:8]}"

        # Use default template if none specified
        if template_id is None:
            template_id = self._create_default_audit_template()

        template = self._report_templates[template_id]

        # Generate report content
        content = self._generate_audit_content(audit_data)

        # Calculate checksum
        content_str = json.dumps(content, sort_keys=True)
        checksum = hashlib.sha256(content_str.encode()).hexdigest()

        report = GeneratedReport(
            report_id=report_id,
            template_id=template_id,
            title=f"{self.organization} Security Audit Report",
            description="Comprehensive security audit findings and recommendations",
            generated_date=datetime.now(),
            report_period=f"{datetime.now().strftime('%B %Y')}",
            data_sources=["security_auditor", "code_review", "cryptographic_review"],
            content=content,
            format=template.format,
            checksum=checksum,
            status="generated"
        )

        self._generated_reports.append(report)
        return report

    def generate_compliance_dashboard_report(self, compliance_data: Dict[str, Any]) -> GeneratedReport:
        """
        Generate compliance dashboard report

        Args:
            compliance_data: Compliance data for dashboard

        Returns:
            GeneratedReport: Generated dashboard report
        """
        import uuid

        report_id = f"dashboard_{uuid.uuid4().hex[:8]}"

        # Create dashboard template
        template_id = self._create_dashboard_template()

        content = {
            'summary': {
                'organization': self.organization,
                'generated_date': datetime.now().isoformat(),
                'overall_compliance_score': compliance_data.get('overall_compliance_score', 0.0),
                'total_regulatory_changes': len(compliance_data.get('regulatory_changes', [])),
                'upcoming_deadlines': len(compliance_data.get('upcoming_deadlines', []))
            },
            'compliance_metrics': compliance_data.get('compliance_metrics', []),
            'regulatory_changes': compliance_data.get('regulatory_changes', []),
            'critical_issues': compliance_data.get('critical_issues', []),
            'recommendations': compliance_data.get('recommendations', [])
        }

        # Calculate checksum
        content_str = json.dumps(content, sort_keys=True)
        checksum = hashlib.sha256(content_str.encode()).hexdigest()

        report = GeneratedReport(
            report_id=report_id,
            template_id=template_id,
            title=f"{self.organization} Compliance Dashboard",
            description="Real-time compliance status and metrics dashboard",
            generated_date=datetime.now(),
            report_period="Current",
            data_sources=["regulatory_monitoring", "compliance_metrics"],
            content=content,
            format=ReportFormat.HTML,
            checksum=checksum,
            status="generated"
        )

        self._generated_reports.append(report)
        return report

    def generate_risk_assessment_report(self, risk_data: Dict[str, Any]) -> GeneratedReport:
        """
        Generate risk assessment report

        Args:
            risk_data: Risk assessment data

        Returns:
            GeneratedReport: Generated risk assessment report
        """
        import uuid

        report_id = f"risk_{uuid.uuid4().hex[:8]}"

        # Create risk assessment template
        template_id = self._create_risk_template()

        content = {
            'executive_summary': {
                'organization': self.organization,
                'assessment_date': datetime.now().isoformat(),
                'overall_risk_level': risk_data.get('overall_risk_level', 'Unknown'),
                'total_risks_identified': len(risk_data.get('risks', [])),
                'critical_risks': len([r for r in risk_data.get('risks', []) if r.get('risk_level') == 'CRITICAL'])
            },
            'risk_findings': risk_data.get('risks', []),
            'mitigation_measures': risk_data.get('mitigation_measures', []),
            'residual_risks': risk_data.get('residual_risks', []),
            'recommendations': risk_data.get('recommendations', [])
        }

        # Calculate checksum
        content_str = json.dumps(content, sort_keys=True)
        checksum = hashlib.sha256(content_str.encode()).hexdigest()

        report = GeneratedReport(
            report_id=report_id,
            template_id=template_id,
            title=f"{self.organization} Risk Assessment Report",
            description="Comprehensive risk assessment and mitigation analysis",
            generated_date=datetime.now(),
            report_period=f"{datetime.now().strftime('%B %Y')}",
            data_sources=["risk_register", "threat_model", "privacy_assessment"],
            content=content,
            format=ReportFormat.PDF,
            checksum=checksum,
            status="generated"
        )

        self._generated_reports.append(report)
        return report

    def _create_default_audit_template(self) -> str:
        """Create default audit report template"""
        template = self.create_report_template(
            name="Default Security Audit Template",
            description="Standard template for security audit reports",
            report_type=ReportType.AUDIT_REPORT,
            format=ReportFormat.JSON,
            sections=[
                "Executive Summary",
                "Audit Scope and Methodology",
                "Findings and Vulnerabilities",
                "Risk Assessment",
                "Compliance Status",
                "Recommendations",
                "Conclusion"
            ],
            required_data=[
                "audit_findings",
                "risk_assessment",
                "compliance_metrics",
                "recommendations"
            ]
        )
        return template.id

    def _create_dashboard_template(self) -> str:
        """Create compliance dashboard template"""
        template = self.create_report_template(
            name="Compliance Dashboard Template",
            description="Template for compliance dashboard reports",
            report_type=ReportType.COMPLIANCE_DASHBOARD,
            format=ReportFormat.HTML,
            sections=[
                "Compliance Overview",
                "Regulatory Changes",
                "Compliance Metrics",
                "Critical Issues",
                "Upcoming Deadlines",
                "Action Items"
            ],
            required_data=[
                "compliance_metrics",
                "regulatory_changes",
                "upcoming_deadlines",
                "critical_issues"
            ]
        )
        return template.id

    def _create_risk_template(self) -> str:
        """Create risk assessment template"""
        template = self.create_report_template(
            name="Risk Assessment Template",
            description="Template for risk assessment reports",
            report_type=ReportType.RISK_ASSESSMENT,
            format=ReportFormat.PDF,
            sections=[
                "Executive Summary",
                "Risk Assessment Methodology",
                "Identified Risks",
                "Risk Analysis",
                "Mitigation Measures",
                "Residual Risks",
                "Recommendations"
            ],
            required_data=[
                "risk_findings",
                "mitigation_measures",
                "residual_risks",
                "recommendations"
            ]
        )
        return template.id

    def _generate_audit_content(self, audit_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate audit report content"""
        return {
            'executive_summary': {
                'organization': self.organization,
                'audit_date': datetime.now().isoformat(),
                'audit_scope': audit_data.get('audit_scope', 'Comprehensive security audit'),
                'overall_findings': len(audit_data.get('findings', [])),
                'critical_findings': len([f for f in audit_data.get('findings', []) if f.get('severity') == 'CRITICAL']),
                'compliance_score': audit_data.get('compliance_score', 0.0)
            },
            'audit_scope': audit_data.get('audit_scope', {}),
            'findings': audit_data.get('findings', []),
            'risk_assessment': audit_data.get('risk_assessment', {}),
            'compliance_status': audit_data.get('compliance_status', {}),
            'recommendations': audit_data.get('recommendations', []),
            'conclusion': {
                'overall_assessment': self._generate_audit_conclusion(audit_data),
                'next_steps': audit_data.get('next_steps', [])
            }
        }

    def _generate_audit_conclusion(self, audit_data: Dict[str, Any]) -> str:
        """Generate audit conclusion based on findings"""
        critical_findings = len([f for f in audit_data.get('findings', []) if f.get('severity') == 'CRITICAL'])
        compliance_score = audit_data.get('compliance_score', 0.0)

        if critical_findings == 0 and compliance_score >= 90:
            return "Excellent security posture with high compliance levels"
        elif critical_findings <= 2 and compliance_score >= 80:
            return "Good security posture with acceptable compliance levels"
        elif critical_findings <= 5 and compliance_score >= 70:
            return "Moderate security posture requiring improvement"
        else:
            return "Significant security concerns requiring immediate attention"

    def export_report(self, report: GeneratedReport, output_path: Optional[str] = None) -> str:
        """
        Export report to file

        Args:
            report: Report to export
            output_path: Optional output path

        Returns:
            str: Path to exported file
        """
        if output_path is None:
            filename = f"{report.report_id}.{report.format.value}"
            output_path = str(self.output_directory / filename)

        # Export based on format
        if report.format == ReportFormat.JSON:
            with open(output_path, 'w') as f:
                json.dump(report.content, f, indent=2, default=str)
        elif report.format == ReportFormat.HTML:
            html_content = self._generate_html_report(report)
            with open(output_path, 'w') as f:
                f.write(html_content)
        elif report.format == ReportFormat.CSV:
            self._export_csv_report(report, output_path)
        else:
            raise ValueError(f"Unsupported export format: {report.format}")

        return output_path

    def _generate_html_report(self, report: GeneratedReport) -> str:
        """Generate HTML report"""
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>{report.title}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .header {{ background: #f8f9fa; padding: 20px; border-radius: 5px; margin-bottom: 20px; }}
                .section {{ margin: 20px 0; }}
                .metric {{ background: #e9ecef; padding: 10px; margin: 5px 0; border-radius: 3px; }}
                .critical {{ color: #dc3545; }}
                .warning {{ color: #ffc107; }}
                .success {{ color: #28a745; }}
                table {{ border-collapse: collapse; width: 100%; margin: 10px 0; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>{report.title}</h1>
                <p><strong>Organization:</strong> {self.organization}</p>
                <p><strong>Generated:</strong> {report.generated_date.strftime('%Y-%m-%d %H:%M:%S')}</p>
                <p><strong>Period:</strong> {report.report_period}</p>
                <p><strong>Report ID:</strong> {report.report_id}</p>
            </div>
        """

        # Add content sections
        for section_name, section_data in report.content.items():
            html += f"<div class='section'><h2>{section_name.replace('_', ' ').title()}</h2>"

            if isinstance(section_data, dict):
                for key, value in section_data.items():
                    if isinstance(value, (int, float)):
                        css_class = "success" if value >= 80 else "warning" if value >= 60 else "critical"
                        html += f"<div class='metric {css_class}'><strong>{key.replace('_', ' ').title()}:</strong> {value}</div>"
                    else:
                        html += f"<p><strong>{key.replace('_', ' ').title()}:</strong> {value}</p>"
            elif isinstance(section_data, list):
                if section_data and isinstance(section_data[0], dict):
                    # Table format
                    if section_data:
                        html += "<table><tr>"
                        for key in section_data[0].keys():
                            html += f"<th>{key.replace('_', ' ').title()}</th>"
                        html += "</tr>"

                        for item in section_data:
                            html += "<tr>"
                            for value in item.values():
                                html += f"<td>{value}</td>"
                            html += "</tr>"
                        html += "</table>"
                else:
                    # List format
                    html += "<ul>"
                    for item in section_data:
                        html += f"<li>{item}</li>"
                    html += "</ul>"
            else:
                html += f"<p>{section_data}</p>"

            html += "</div>"

        html += "</body></html>"
        return html

    def _export_csv_report(self, report: GeneratedReport, output_path: str):
        """Export report as CSV"""
        with open(output_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)

            # Write header
            writer.writerow(['Section', 'Key', 'Value'])

            # Write content
            for section_name, section_data in report.content.items():
                if isinstance(section_data, dict):
                    for key, value in section_data.items():
                        writer.writerow([section_name, key, str(value)])
                elif isinstance(section_data, list):
                    for i, item in enumerate(section_data):
                        if isinstance(item, dict):
                            for key, value in item.items():
                                writer.writerow([section_name, f"{i}_{key}", str(value)])
                        else:
                            writer.writerow([section_name, str(i), str(item)])

    def create_compliance_dashboard(self, title: str, description: str,
                                  widgets: List[Dict[str, Any]]) -> ComplianceDashboard:
        """
        Create a compliance dashboard

        Args:
            title: Dashboard title
            description: Dashboard description
            widgets: Dashboard widgets configuration

        Returns:
            ComplianceDashboard: Created dashboard
        """
        import uuid

        dashboard_id = f"dashboard_{uuid.uuid4().hex[:8]}"

        dashboard = ComplianceDashboard(
            dashboard_id=dashboard_id,
            title=title,
            description=description,
            created_date=datetime.now(),
            last_updated=datetime.now(),
            widgets=widgets,
            data_sources=["compliance_metrics", "regulatory_changes", "audit_findings"],
            refresh_interval=60  # 60 minutes
        )

        self._dashboards.append(dashboard)
        return dashboard

    def get_report_templates(self) -> Dict[str, ReportTemplate]:
        """Get all report templates"""
        return self._report_templates.copy()

    def get_generated_reports(self) -> List[GeneratedReport]:
        """Get all generated reports"""
        return self._generated_reports.copy()

    def get_compliance_dashboards(self) -> List[ComplianceDashboard]:
        """Get all compliance dashboards"""
        return self._dashboards.copy()

    def generate_annual_compliance_report(self, year: int) -> GeneratedReport:
        """
        Generate annual compliance report

        Args:
            year: Year for the annual report

        Returns:
            GeneratedReport: Generated annual report
        """
        import uuid

        report_id = f"annual_{year}_{uuid.uuid4().hex[:8]}"

        # Create annual template
        template_id = self._create_annual_template()

        content = {
            'annual_summary': {
                'organization': self.organization,
                'reporting_year': year,
                'report_generated': datetime.now().isoformat(),
                'overall_compliance_trend': 'improving',
                'major_achievements': [
                    'Achieved ISO 27001 certification',
                    'Implemented comprehensive privacy program',
                    'Completed SOC 2 Type II audit',
                    'Established regulatory monitoring program'
                ],
                'key_challenges': [
                    'Adapting to evolving AI regulations',
                    'Managing third-party risk',
                    'Maintaining compliance in cloud environments'
                ]
            },
            'yearly_metrics': {
                'average_compliance_score': 87.5,
                'regulatory_changes_addressed': 12,
                'audits_completed': 4,
                'training_sessions_conducted': 8,
                'incidents_handled': 2
            },
            'compliance_by_regulation': {
                'GDPR': {'score': 92.0, 'status': 'compliant'},
                'CCPA': {'score': 88.5, 'status': 'partially_compliant'},
                'NIST_CSF': {'score': 89.2, 'status': 'compliant'},
                'ISO_27001': {'score': 91.8, 'status': 'compliant'},
                'SOC_2': {'score': 86.3, 'status': 'partially_compliant'}
            },
            'outlook': {
                'upcoming_regulatory_changes': [
                    'AI Act implementation',
                    'Data privacy regulations updates',
                    'Cybersecurity framework revisions'
                ],
                'planned_improvements': [
                    'Enhanced automated compliance monitoring',
                    'Expanded privacy impact assessment program',
                    'Improved incident response capabilities'
                ]
            }
        }

        # Calculate checksum
        content_str = json.dumps(content, sort_keys=True)
        checksum = hashlib.sha256(content_str.encode()).hexdigest()

        report = GeneratedReport(
            report_id=report_id,
            template_id=template_id,
            title=f"{self.organization} Annual Compliance Report {year}",
            description=f"Comprehensive compliance report for {year}",
            generated_date=datetime.now(),
            report_period=f"January 1 - December 31, {year}",
            data_sources=["annual_compliance_data", "regulatory_monitoring", "audit_reports"],
            content=content,
            format=ReportFormat.PDF,
            checksum=checksum,
            status="generated"
        )

        self._generated_reports.append(report)
        return report

    def _create_annual_template(self) -> str:
        """Create annual compliance report template"""
        template = self.create_report_template(
            name="Annual Compliance Report Template",
            description="Template for annual compliance reports",
            report_type=ReportType.ANNUAL_COMPLIANCE,
            format=ReportFormat.PDF,
            sections=[
                "Executive Summary",
                "Compliance Achievements",
                "Key Metrics and Performance",
                "Regulatory Compliance Status",
                "Challenges and Lessons Learned",
                "Future Outlook",
                "Conclusion"
            ],
            required_data=[
                "annual_metrics",
                "compliance_scores",
                "regulatory_changes",
                "audit_results",
                "incident_reports"
            ]
        )
        return template.id
