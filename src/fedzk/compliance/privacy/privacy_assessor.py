"""
Privacy Impact Assessment Framework

This module provides comprehensive privacy impact assessment capabilities
for evaluating privacy risks and implementing mitigation measures.
"""

import json
import hashlib
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class PrivacyRiskLevel(Enum):
    """Privacy risk levels"""
    VERY_HIGH = "VERY_HIGH"
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"
    VERY_LOW = "VERY_LOW"


class PrivacyImpactType(Enum):
    """Types of privacy impacts"""
    IDENTIFICATION = "identification"
    SENSITIVE_DATA_EXPOSURE = "sensitive_data_exposure"
    TRACKING = "tracking"
    PROFILING = "profiling"
    DISCRIMINATION = "discrimination"
    REPUTATIONAL_HARM = "reputational_harm"
    FINANCIAL_LOSS = "financial_loss"
    PHYSICAL_HARM = "physical_harm"


class DataProcessingScale(Enum):
    """Scale of data processing"""
    SMALL = "small"  # < 1000 individuals
    MEDIUM = "medium"  # 1000-10000 individuals
    LARGE = "large"  # 10000-100000 individuals
    VERY_LARGE = "very_large"  # > 100000 individuals


@dataclass
class PrivacyRisk:
    """Represents a privacy risk"""
    id: str
    title: str
    description: str
    risk_level: PrivacyRiskLevel
    impact_type: PrivacyImpactType
    likelihood: float  # 0.0 to 1.0
    impact_score: float  # 0.0 to 1.0
    affected_data_subjects: List[str]
    data_categories: List[str]
    mitigation_measures: List[str]
    residual_risk: float
    risk_owner: str
    review_date: datetime


@dataclass
class PrivacyImpactAssessment:
    """Privacy Impact Assessment result"""
    assessment_id: str
    project_name: str
    assessment_date: datetime
    assessor: str
    data_processing_description: str
    processing_scale: DataProcessingScale
    data_subjects_affected: List[str]
    data_categories: List[str]
    processing_purposes: List[str]
    privacy_risks: List[PrivacyRisk]
    mitigation_measures: List[str]
    residual_risks: List[str]
    recommendations: List[str]
    approval_required: bool
    approval_status: str
    next_review_date: datetime
    approval_date: Optional[datetime] = None


@dataclass
class DataFlow:
    """Represents a data flow in the system"""
    id: str
    source: str
    destination: str
    data_categories: List[str]
    processing_purpose: str
    security_measures: List[str]
    retention_period: str
    cross_border_transfer: bool
    third_party_involvement: bool


class PrivacyImpactAssessor:
    """
    Privacy Impact Assessment framework for FEDZK

    Provides comprehensive PIA capabilities for evaluating privacy risks
    and implementing appropriate mitigation measures.
    """

    def __init__(self, organization_name: str = "FEDZK"):
        self.organization = organization_name
        self._assessments: List[PrivacyImpactAssessment] = []
        self._data_flows: List[DataFlow] = []

    def perform_privacy_impact_assessment(self, project_name: str,
                                        data_processing_description: str,
                                        processing_scale: DataProcessingScale,
                                        data_subjects: List[str],
                                        data_categories: List[str],
                                        processing_purposes: List[str]) -> PrivacyImpactAssessment:
        """
        Perform comprehensive privacy impact assessment

        Args:
            project_name: Name of the project/activity
            data_processing_description: Description of data processing
            processing_scale: Scale of data processing
            data_subjects: Types of data subjects affected
            data_categories: Categories of personal data
            processing_purposes: Purposes of processing

        Returns:
            PrivacyImpactAssessment: Detailed PIA result
        """
        import uuid

        assessment_id = f"pia_{uuid.uuid4().hex[:8]}"

        # Identify privacy risks
        privacy_risks = self._identify_privacy_risks(
            data_categories, processing_purposes, processing_scale
        )

        # Develop mitigation measures
        mitigation_measures = self._develop_mitigation_measures(privacy_risks)

        # Assess residual risks
        residual_risks = self._assess_residual_risks(privacy_risks, mitigation_measures)

        # Generate recommendations
        recommendations = self._generate_pia_recommendations(privacy_risks, residual_risks)

        # Determine if approval is required
        approval_required = self._determine_approval_requirement(
            privacy_risks, processing_scale
        )

        assessment = PrivacyImpactAssessment(
            assessment_id=assessment_id,
            project_name=project_name,
            assessment_date=datetime.now(),
            assessor="Privacy Officer",
            data_processing_description=data_processing_description,
            processing_scale=processing_scale,
            data_subjects_affected=data_subjects,
            data_categories=data_categories,
            processing_purposes=processing_purposes,
            privacy_risks=privacy_risks,
            mitigation_measures=mitigation_measures,
            residual_risks=residual_risks,
            recommendations=recommendations,
            approval_required=approval_required,
            approval_status="pending" if approval_required else "approved",
            next_review_date=datetime.now() + timedelta(days=365)
        )

        self._assessments.append(assessment)
        return assessment

    def _identify_privacy_risks(self, data_categories: List[str],
                              processing_purposes: List[str],
                              processing_scale: DataProcessingScale) -> List[PrivacyRisk]:
        """Identify privacy risks based on data processing characteristics"""
        risks = []

        # Risk 1: Identification risk
        if "identifiers" in data_categories or "behavioral" in data_categories:
            risk_level = PrivacyRiskLevel.HIGH if processing_scale in [DataProcessingScale.LARGE, DataProcessingScale.VERY_LARGE] else PrivacyRiskLevel.MEDIUM
            risk = PrivacyRisk(
                id=f"risk_id_{hashlib.md5(str(datetime.now()).encode()).hexdigest()[:8]}",
                title="Individual Identification Risk",
                description="Risk of identifying individuals through data correlation",
                risk_level=risk_level,
                impact_type=PrivacyImpactType.IDENTIFICATION,
                likelihood=0.7 if processing_scale == DataProcessingScale.VERY_LARGE else 0.5,
                impact_score=0.8,
                affected_data_subjects=["end_users", "participants"],
                data_categories=data_categories,
                mitigation_measures=[
                    "Implement data pseudonymization",
                    "Use differential privacy techniques",
                    "Limit data retention periods",
                    "Implement data minimization"
                ],
                residual_risk=0.3,
                risk_owner="Privacy Officer",
                review_date=datetime.now() + timedelta(days=180)
            )
            risks.append(risk)

        # Risk 2: Sensitive data exposure
        if "sensitive" in data_categories or "health" in data_categories:
            risk = PrivacyRisk(
                id=f"risk_sensitive_{hashlib.md5(str(datetime.now()).encode()).hexdigest()[:8]}",
                title="Sensitive Data Exposure Risk",
                description="Risk of exposing sensitive personal information",
                risk_level=PrivacyRiskLevel.VERY_HIGH,
                impact_type=PrivacyImpactType.SENSITIVE_DATA_EXPOSURE,
                likelihood=0.4,
                impact_score=0.9,
                affected_data_subjects=["end_users"],
                data_categories=data_categories,
                mitigation_measures=[
                    "Implement strong encryption (AES-256)",
                    "Use secure key management",
                    "Implement strict access controls",
                    "Regular security audits"
                ],
                residual_risk=0.2,
                risk_owner="Security Officer",
                review_date=datetime.now() + timedelta(days=90)
            )
            risks.append(risk)

        # Risk 3: Profiling and automated decision making
        if "profiling" in processing_purposes or "automated_decisions" in processing_purposes:
            risk = PrivacyRisk(
                id=f"risk_profiling_{hashlib.md5(str(datetime.now()).encode()).hexdigest()[:8]}",
                title="Profiling and Automated Decision Risk",
                description="Risk of discriminatory profiling and biased automated decisions",
                risk_level=PrivacyRiskLevel.HIGH,
                impact_type=PrivacyImpactType.PROFILING,
                likelihood=0.6,
                impact_score=0.7,
                affected_data_subjects=["end_users"],
                data_categories=data_categories,
                mitigation_measures=[
                    "Implement fairness-aware algorithms",
                    "Regular bias audits and testing",
                    "Transparent decision-making processes",
                    "Human oversight of automated decisions"
                ],
                residual_risk=0.4,
                risk_owner="AI Ethics Officer",
                review_date=datetime.now() + timedelta(days=180)
            )
            risks.append(risk)

        # Risk 4: Data breach impact
        risk = PrivacyRisk(
            id=f"risk_breach_{hashlib.md5(str(datetime.now()).encode()).hexdigest()[:8]}",
            title="Data Breach Impact Risk",
            description="Risk of data breach leading to privacy violations",
            risk_level=PrivacyRiskLevel.HIGH,
            impact_type=PrivacyImpactType.REPUTATIONAL_HARM,
            likelihood=0.3,
            impact_score=0.8,
            affected_data_subjects=["all_data_subjects"],
            data_categories=data_categories,
            mitigation_measures=[
                "Implement comprehensive security controls",
                "Regular security testing and penetration testing",
                "Incident response planning",
                "Data encryption and access controls"
            ],
            residual_risk=0.3,
            risk_owner="Security Officer",
            review_date=datetime.now() + timedelta(days=180)
        )
        risks.append(risk)

        return risks

    def _develop_mitigation_measures(self, risks: List[PrivacyRisk]) -> List[str]:
        """Develop mitigation measures for identified risks"""
        mitigation_measures = []

        # Collect all unique mitigation measures
        all_measures = set()
        for risk in risks:
            all_measures.update(risk.mitigation_measures)

        mitigation_measures.extend(all_measures)

        # Add general mitigation measures
        mitigation_measures.extend([
            "Conduct regular privacy risk assessments",
            "Implement comprehensive data protection measures",
            "Establish privacy training programs",
            "Maintain detailed privacy documentation",
            "Implement privacy by design principles",
            "Establish data subject rights procedures",
            "Regular compliance monitoring and auditing"
        ])

        return list(set(mitigation_measures))  # Remove duplicates

    def _assess_residual_risks(self, risks: List[PrivacyRisk],
                             mitigation_measures: List[str]) -> List[str]:
        """Assess residual risks after mitigation"""
        residual_risks = []

        # Evaluate each risk considering mitigation measures
        for risk in risks:
            # Calculate residual risk based on mitigation effectiveness
            mitigation_effectiveness = self._calculate_mitigation_effectiveness(
                risk, mitigation_measures
            )

            residual_risk_level = risk.likelihood * risk.impact_score * (1 - mitigation_effectiveness)

            if residual_risk_level > 0.6:
                residual_risks.append(f"High residual risk for {risk.title}")
            elif residual_risk_level > 0.3:
                residual_risks.append(f"Medium residual risk for {risk.title}")

        # Add general residual risks
        residual_risks.extend([
            "Advanced privacy attacks and de-anonymization techniques",
            "Supply chain privacy risks from third-party providers",
            "Emerging privacy regulations and compliance changes",
            "Technological changes affecting privacy controls"
        ])

        return residual_risks

    def _calculate_mitigation_effectiveness(self, risk: PrivacyRisk,
                                         mitigation_measures: List[str]) -> float:
        """Calculate effectiveness of mitigation measures for a risk"""
        # Simple effectiveness calculation based on mitigation coverage
        risk_mitigation_keywords = {
            PrivacyRiskLevel.VERY_HIGH: ['encryption', 'access_controls', 'audits', 'oversight'],
            PrivacyRiskLevel.HIGH: ['privacy', 'security', 'monitoring', 'training'],
            PrivacyRiskLevel.MEDIUM: ['policies', 'procedures', 'documentation'],
            PrivacyRiskLevel.LOW: ['awareness', 'basic_controls']
        }

        relevant_keywords = risk_mitigation_keywords.get(risk.risk_level, [])
        matching_measures = 0

        for measure in mitigation_measures:
            measure_lower = measure.lower()
            if any(keyword in measure_lower for keyword in relevant_keywords):
                matching_measures += 1

        effectiveness = min(1.0, matching_measures / len(relevant_keywords)) if relevant_keywords else 0.5
        return effectiveness

    def _generate_pia_recommendations(self, risks: List[PrivacyRisk],
                                    residual_risks: List[str]) -> List[str]:
        """Generate PIA recommendations"""
        recommendations = []

        # Risk-based recommendations
        high_risks = [r for r in risks if r.risk_level in [PrivacyRiskLevel.HIGH, PrivacyRiskLevel.VERY_HIGH]]
        if high_risks:
            recommendations.append("Implement enhanced privacy controls for high-risk processing activities")

        # General recommendations
        recommendations.extend([
            "Conduct regular privacy impact assessments",
            "Implement privacy by design principles",
            "Establish comprehensive data protection measures",
            "Develop privacy training programs for staff",
            "Implement data subject rights procedures",
            "Maintain detailed privacy documentation",
            "Regular compliance monitoring and auditing",
            "Establish privacy incident response procedures"
        ])

        # Residual risk recommendations
        if residual_risks:
            recommendations.append("Develop strategies to address identified residual risks")

        return recommendations

    def _determine_approval_requirement(self, risks: List[PrivacyRisk],
                                      processing_scale: DataProcessingScale) -> bool:
        """Determine if executive approval is required"""
        # Approval required for:
        # - Very high risk activities
        # - Large or very large scale processing
        # - Sensitive data categories

        has_very_high_risk = any(r.risk_level == PrivacyRiskLevel.VERY_HIGH for r in risks)
        large_scale = processing_scale in [DataProcessingScale.LARGE, DataProcessingScale.VERY_LARGE]

        return has_very_high_risk or large_scale

    def analyze_data_flows(self) -> List[DataFlow]:
        """Analyze data flows in the system for privacy compliance"""
        # Define FEDZK data flows
        data_flows = [
            DataFlow(
                id="df_001",
                source="User Application",
                destination="Federated Learning Coordinator",
                data_categories=["behavioral", "technical"],
                processing_purpose="Federated model training",
                security_measures=[
                    "TLS 1.3 encryption",
                    "Zero-knowledge proofs",
                    "Differential privacy"
                ],
                retention_period="Training data: 30 days, Aggregated models: indefinite",
                cross_border_transfer=False,
                third_party_involvement=False
            ),
            DataFlow(
                id="df_002",
                source="Federated Learning Coordinator",
                destination="ZK Proof Generator",
                data_categories=["technical", "behavioral"],
                processing_purpose="Generate privacy-preserving proofs",
                security_measures=[
                    "Homomorphic encryption",
                    "Secure multi-party computation",
                    "Cryptographic proof validation"
                ],
                retention_period="Proof data: 90 days",
                cross_border_transfer=False,
                third_party_involvement=False
            ),
            DataFlow(
                id="df_003",
                source="ZK Proof Generator",
                destination="Model Registry",
                data_categories=["technical"],
                processing_purpose="Store verified models",
                security_measures=[
                    "AES-256 encryption",
                    "Access control lists",
                    "Integrity verification"
                ],
                retention_period="Models: indefinite",
                cross_border_transfer=False,
                third_party_involvement=False
            )
        ]

        self._data_flows.extend(data_flows)
        return data_flows

    def export_assessment(self, assessment: PrivacyImpactAssessment,
                         output_format: str = 'json') -> str:
        """Export privacy impact assessment"""
        if output_format == 'json':
            return json.dumps(self._assessment_to_dict(assessment), indent=2, default=str)
        else:
            raise ValueError(f"Unsupported format: {output_format}")

    def _assessment_to_dict(self, assessment: PrivacyImpactAssessment) -> Dict[str, Any]:
        """Convert assessment to dictionary"""
        return {
            'assessment_id': assessment.assessment_id,
            'project_name': assessment.project_name,
            'assessment_date': assessment.assessment_date.isoformat(),
            'assessor': assessment.assessor,
            'data_processing_description': assessment.data_processing_description,
            'processing_scale': assessment.processing_scale.value,
            'data_subjects_affected': assessment.data_subjects_affected,
            'data_categories': assessment.data_categories,
            'processing_purposes': assessment.processing_purposes,
            'privacy_risks': [
                {
                    'id': r.id,
                    'title': r.title,
                    'description': r.description,
                    'risk_level': r.risk_level.value,
                    'impact_type': r.impact_type.value,
                    'likelihood': r.likelihood,
                    'impact_score': r.impact_score,
                    'affected_data_subjects': r.affected_data_subjects,
                    'data_categories': r.data_categories,
                    'mitigation_measures': r.mitigation_measures,
                    'residual_risk': r.residual_risk,
                    'risk_owner': r.risk_owner,
                    'review_date': r.review_date.isoformat()
                }
                for r in assessment.privacy_risks
            ],
            'mitigation_measures': assessment.mitigation_measures,
            'residual_risks': assessment.residual_risks,
            'recommendations': assessment.recommendations,
            'approval_required': assessment.approval_required,
            'approval_status': assessment.approval_status,
            'next_review_date': assessment.next_review_date.isoformat()
        }
