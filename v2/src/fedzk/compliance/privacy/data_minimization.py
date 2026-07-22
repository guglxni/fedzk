"""
Data Minimization Framework

This module provides data minimization capabilities for implementing
privacy by design principles and reducing privacy risks.
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


class DataMinimizationStrategy(Enum):
    """Data minimization strategies"""
    DATA_REDUCTION = "data_reduction"
    PURPOSE_LIMITATION = "purpose_limitation"
    STORAGE_LIMITATION = "storage_limitation"
    PSEUDONYMIZATION = "pseudonymization"
    AGGREGATION = "aggregation"
    DIFFERENTIAL_PRIVACY = "differential_privacy"


class DataRetentionPolicy(Enum):
    """Data retention policy types"""
    BUSINESS_NEED = "business_need"
    LEGAL_REQUIREMENT = "legal_requirement"
    CONSENT_DURATION = "consent_duration"
    CONTRACT_DURATION = "contract_duration"
    FIXED_PERIOD = "fixed_period"


@dataclass
class DataField:
    """Represents a data field in the system"""
    id: str
    name: str
    category: str
    sensitivity_level: str  # public, internal, confidential, restricted
    purpose: str
    retention_period: str
    legal_basis: str
    minimization_applied: bool
    minimization_strategy: Optional[DataMinimizationStrategy]
    usage_frequency: str  # high, medium, low


@dataclass
class DataMinimizationRule:
    """Represents a data minimization rule"""
    id: str
    name: str
    description: str
    data_categories: List[str]
    strategy: DataMinimizationStrategy
    conditions: Dict[str, Any]
    implementation_status: str
    effectiveness_score: float
    automated: bool


@dataclass
class DataMinimizationAssessment:
    """Assessment of data minimization implementation"""
    assessment_id: str
    assessment_date: datetime
    data_fields_analyzed: int
    minimization_rules_applied: int
    data_reduction_percentage: float
    privacy_risk_reduction: float
    recommendations: List[str]
    implementation_gaps: List[str]
    next_review_date: datetime


class DataMinimization:
    """
    Data Minimization framework for FEDZK

    Implements privacy by design principles through data minimization
    techniques and automated data lifecycle management.
    """

    def __init__(self, organization_name: str = "FEDZK"):
        self.organization = organization_name
        self._data_fields: List[DataField] = []
        self._minimization_rules: List[DataMinimizationRule] = []
        self._assessments: List[DataMinimizationAssessment] = []

    def define_data_fields(self) -> List[DataField]:
        """Define data fields used in FEDZK system"""
        data_fields = [
            DataField(
                id="df_001",
                name="user_behavior_data",
                category="behavioral",
                sensitivity_level="confidential",
                purpose="Federated learning model training",
                retention_period="30 days",
                legal_basis="Legitimate interest",
                minimization_applied=True,
                minimization_strategy=DataMinimizationStrategy.DIFFERENTIAL_PRIVACY,
                usage_frequency="high"
            ),
            DataField(
                id="df_002",
                name="model_parameters",
                category="technical",
                sensitivity_level="internal",
                purpose="Model aggregation and updates",
                retention_period="indefinite",
                legal_basis="Contract performance",
                minimization_applied=True,
                minimization_strategy=DataMinimizationStrategy.AGGREGATION,
                usage_frequency="high"
            ),
            DataField(
                id="df_003",
                name="user_identifiers",
                category="identifiers",
                sensitivity_level="restricted",
                purpose="User authentication and authorization",
                retention_period="7 years",
                legal_basis="Legal obligation",
                minimization_applied=True,
                minimization_strategy=DataMinimizationStrategy.PSEUDONYMIZATION,
                usage_frequency="medium"
            ),
            DataField(
                id="df_004",
                name="training_metadata",
                category="technical",
                sensitivity_level="internal",
                purpose="Training process monitoring",
                retention_period="90 days",
                legal_basis="Business necessity",
                minimization_applied=True,
                minimization_strategy=DataMinimizationStrategy.DATA_REDUCTION,
                usage_frequency="medium"
            ),
            DataField(
                id="df_005",
                name="audit_logs",
                category="technical",
                sensitivity_level="confidential",
                purpose="Security monitoring and compliance",
                retention_period="7 years",
                legal_basis="Legal obligation",
                minimization_applied=True,
                minimization_strategy=DataMinimizationStrategy.PURPOSE_LIMITATION,
                usage_frequency="low"
            )
        ]

        self._data_fields.extend(data_fields)
        return data_fields

    def define_minimization_rules(self) -> List[DataMinimizationRule]:
        """Define data minimization rules"""
        rules = [
            DataMinimizationRule(
                id="rule_001",
                name="Differential Privacy for Behavioral Data",
                description="Apply differential privacy to behavioral data before processing",
                data_categories=["behavioral"],
                strategy=DataMinimizationStrategy.DIFFERENTIAL_PRIVACY,
                conditions={
                    "sensitivity_level": "confidential",
                    "data_volume": "high"
                },
                implementation_status="implemented",
                effectiveness_score=0.85,
                automated=True
            ),
            DataMinimizationRule(
                id="rule_002",
                name="Data Aggregation for Model Parameters",
                description="Aggregate model parameters to reduce individual-level data",
                data_categories=["technical"],
                strategy=DataMinimizationStrategy.AGGREGATION,
                conditions={
                    "purpose": "model_training",
                    "data_type": "parameters"
                },
                implementation_status="implemented",
                effectiveness_score=0.95,
                automated=True
            ),
            DataMinimizationRule(
                id="rule_003",
                name="Pseudonymization of Identifiers",
                description="Replace direct identifiers with pseudonyms",
                data_categories=["identifiers"],
                strategy=DataMinimizationStrategy.PSEUDONYMIZATION,
                conditions={
                    "sensitivity_level": "restricted",
                    "processing_type": "storage"
                },
                implementation_status="implemented",
                effectiveness_score=0.90,
                automated=True
            ),
            DataMinimizationRule(
                id="rule_004",
                name="Purpose Limitation for Audit Data",
                description="Limit audit data usage to security purposes only",
                data_categories=["technical"],
                strategy=DataMinimizationStrategy.PURPOSE_LIMITATION,
                conditions={
                    "purpose": "audit",
                    "data_category": "logs"
                },
                implementation_status="implemented",
                effectiveness_score=0.80,
                automated=False
            ),
            DataMinimizationRule(
                id="rule_005",
                name="Storage Limitation for Training Data",
                description="Automatically delete training data after retention period",
                data_categories=["behavioral", "technical"],
                strategy=DataMinimizationStrategy.STORAGE_LIMITATION,
                conditions={
                    "retention_expired": True
                },
                implementation_status="implemented",
                effectiveness_score=0.95,
                automated=True
            ),
            DataMinimizationRule(
                id="rule_006",
                name="Data Reduction for Metadata",
                description="Reduce metadata collection to essential fields only",
                data_categories=["technical"],
                strategy=DataMinimizationStrategy.DATA_REDUCTION,
                conditions={
                    "data_type": "metadata",
                    "usage_frequency": "low"
                },
                implementation_status="planned",
                effectiveness_score=0.70,
                automated=False
            )
        ]

        self._minimization_rules.extend(rules)
        return rules

    def apply_data_minimization(self, data_field: DataField) -> Dict[str, Any]:
        """
        Apply appropriate data minimization techniques to a data field

        Args:
            data_field: The data field to minimize

        Returns:
            Dict containing minimization results
        """
        applicable_rules = self._find_applicable_rules(data_field)

        if not applicable_rules:
            return {
                'minimized': False,
                'reason': 'No applicable minimization rules found',
                'recommendations': ['Define minimization strategy for this data field']
            }

        # Apply the most effective rule
        best_rule = max(applicable_rules, key=lambda r: r.effectiveness_score)

        minimization_result = {
            'minimized': True,
            'strategy_applied': best_rule.strategy.value,
            'rule_id': best_rule.id,
            'effectiveness_score': best_rule.effectiveness_score,
            'automated': best_rule.automated,
            'data_reduction_estimate': self._estimate_data_reduction(best_rule.strategy, data_field),
            'privacy_risk_reduction': self._estimate_privacy_risk_reduction(best_rule.strategy)
        }

        return minimization_result

    def _find_applicable_rules(self, data_field: DataField) -> List[DataMinimizationRule]:
        """Find applicable minimization rules for a data field"""
        applicable_rules = []

        for rule in self._minimization_rules:
            if data_field.category in rule.data_categories:
                # Check conditions
                conditions_met = True
                for condition_key, condition_value in rule.conditions.items():
                    if hasattr(data_field, condition_key):
                        field_value = getattr(data_field, condition_key)
                        if field_value != condition_value:
                            conditions_met = False
                            break

                if conditions_met:
                    applicable_rules.append(rule)

        return applicable_rules

    def _estimate_data_reduction(self, strategy: DataMinimizationStrategy,
                               data_field: DataField) -> float:
        """Estimate data reduction percentage for a minimization strategy"""
        reduction_estimates = {
            DataMinimizationStrategy.DATA_REDUCTION: 0.60,  # 60% reduction
            DataMinimizationStrategy.PSEUDONYMIZATION: 0.30,  # 30% reduction (linkage removal)
            DataMinimizationStrategy.AGGREGATION: 0.80,  # 80% reduction (individual data loss)
            DataMinimizationStrategy.DIFFERENTIAL_PRIVACY: 0.20,  # 20% reduction (noise addition)
            DataMinimizationStrategy.STORAGE_LIMITATION: 0.90,  # 90% reduction (deletion)
            DataMinimizationStrategy.PURPOSE_LIMITATION: 0.10  # 10% reduction (usage restriction)
        }

        base_reduction = reduction_estimates.get(strategy, 0.0)

        # Adjust based on data field characteristics
        if data_field.sensitivity_level == "restricted":
            base_reduction *= 1.2  # Higher reduction for sensitive data
        elif data_field.usage_frequency == "low":
            base_reduction *= 1.1  # Higher reduction for infrequently used data

        return min(0.95, base_reduction)  # Cap at 95%

    def _estimate_privacy_risk_reduction(self, strategy: DataMinimizationStrategy) -> float:
        """Estimate privacy risk reduction for a minimization strategy"""
        risk_reduction_estimates = {
            DataMinimizationStrategy.DATA_REDUCTION: 0.40,
            DataMinimizationStrategy.PSEUDONYMIZATION: 0.70,
            DataMinimizationStrategy.AGGREGATION: 0.85,
            DataMinimizationStrategy.DIFFERENTIAL_PRIVACY: 0.60,
            DataMinimizationStrategy.STORAGE_LIMITATION: 0.95,
            DataMinimizationStrategy.PURPOSE_LIMITATION: 0.30
        }

        return risk_reduction_estimates.get(strategy, 0.0)

    def perform_minimization_assessment(self) -> DataMinimizationAssessment:
        """
        Perform comprehensive data minimization assessment

        Returns:
            DataMinimizationAssessment: Assessment results
        """
        import uuid

        assessment_id = f"minimization_{uuid.uuid4().hex[:8]}"

        # Analyze all data fields
        total_fields = len(self._data_fields)
        minimized_fields = sum(1 for field in self._data_fields if field.minimization_applied)
        minimization_percentage = (minimized_fields / total_fields * 100) if total_fields > 0 else 0

        # Calculate data reduction
        total_data_reduction = 0.0
        total_privacy_risk_reduction = 0.0

        for field in self._data_fields:
            if field.minimization_applied and field.minimization_strategy:
                total_data_reduction += self._estimate_data_reduction(field.minimization_strategy, field)
                total_privacy_risk_reduction += self._estimate_privacy_risk_reduction(field.minimization_strategy)

        avg_data_reduction = float(total_data_reduction / total_fields) if total_fields > 0 else 0.0
        avg_privacy_risk_reduction = float(total_privacy_risk_reduction / total_fields) if total_fields > 0 else 0.0

        # Generate recommendations
        recommendations = self._generate_minimization_recommendations()

        # Identify implementation gaps
        implementation_gaps = self._identify_implementation_gaps()

        assessment = DataMinimizationAssessment(
            assessment_id=assessment_id,
            assessment_date=datetime.now(),
            data_fields_analyzed=total_fields,
            minimization_rules_applied=len([r for r in self._minimization_rules if r.implementation_status == "implemented"]),
            data_reduction_percentage=avg_data_reduction,
            privacy_risk_reduction=avg_privacy_risk_reduction,
            recommendations=recommendations,
            implementation_gaps=implementation_gaps,
            next_review_date=datetime.now() + timedelta(days=180)
        )

        self._assessments.append(assessment)
        return assessment

    def _generate_minimization_recommendations(self) -> List[str]:
        """Generate data minimization recommendations"""
        recommendations = []

        # Analyze current implementation
        implemented_rules = [r for r in self._minimization_rules if r.implementation_status == "implemented"]
        planned_rules = [r for r in self._minimization_rules if r.implementation_status == "planned"]

        if len(implemented_rules) < len(self._minimization_rules) * 0.8:
            recommendations.append("Increase implementation of data minimization rules")

        # Check effectiveness
        low_effectiveness_rules = [r for r in implemented_rules if r.effectiveness_score < 0.7]
        if low_effectiveness_rules:
            recommendations.append("Review and improve low-effectiveness minimization rules")

        # General recommendations
        recommendations.extend([
            "Conduct regular data minimization assessments",
            "Implement automated data deletion processes",
            "Review data retention policies annually",
            "Train staff on data minimization principles",
            "Implement data minimization by design",
            "Regular audit of data processing activities",
            "Establish data minimization metrics and KPIs"
        ])

        return recommendations

    def _identify_implementation_gaps(self) -> List[str]:
        """Identify gaps in data minimization implementation"""
        gaps = []

        # Check for unminimized fields
        unminimized_fields = [f for f in self._data_fields if not f.minimization_applied]
        if unminimized_fields:
            gaps.append(f"{len(unminimized_fields)} data fields lack minimization strategies")

        # Check for unimplemented rules
        unimplemented_rules = [r for r in self._minimization_rules if r.implementation_status != "implemented"]
        if unimplemented_rules:
            gaps.append(f"{len(unimplemented_rules)} minimization rules not fully implemented")

        # Check automation level
        manual_rules = [r for r in self._minimization_rules if not r.automated]
        if len(manual_rules) > len(self._minimization_rules) * 0.5:
            gaps.append("High proportion of manual minimization processes")

        return gaps

    def implement_automated_minimization(self, data_field: DataField,
                                        strategy: DataMinimizationStrategy) -> Dict[str, Any]:
        """
        Implement automated data minimization for a data field

        Args:
            data_field: The data field to minimize
            strategy: The minimization strategy to apply

        Returns:
            Dict containing implementation results
        """
        implementation_result = {
            'field_id': data_field.id,
            'strategy': strategy.value,
            'automated': True,
            'implementation_status': 'success',
            'data_reduction_achieved': self._estimate_data_reduction(strategy, data_field),
            'privacy_benefits': self._estimate_privacy_risk_reduction(strategy),
            'monitoring_enabled': True,
            'audit_trail': True
        }

        # Update the data field
        data_field.minimization_applied = True
        data_field.minimization_strategy = strategy

        return implementation_result

    def monitor_minimization_effectiveness(self) -> Dict[str, Any]:
        """Monitor the effectiveness of data minimization measures"""
        monitoring_results = {
            'monitoring_date': datetime.now().isoformat(),
            'overall_effectiveness': 0.0,
            'rule_effectiveness': {},
            'data_reduction_metrics': {},
            'privacy_risk_metrics': {},
            'recommendations': []
        }

        # Calculate overall effectiveness
        if self._minimization_rules:
            total_effectiveness = sum(r.effectiveness_score for r in self._minimization_rules)
            monitoring_results['overall_effectiveness'] = total_effectiveness / len(self._minimization_rules)

        # Rule-specific effectiveness
        for rule in self._minimization_rules:
            monitoring_results['rule_effectiveness'][rule.id] = {
                'name': rule.name,
                'effectiveness': rule.effectiveness_score,
                'status': rule.implementation_status,
                'automated': rule.automated
            }

        # Generate monitoring recommendations
        if monitoring_results['overall_effectiveness'] < 0.7:
            monitoring_results['recommendations'].append("Improve overall minimization effectiveness")
        if any(r.effectiveness_score < 0.6 for r in self._minimization_rules):
            monitoring_results['recommendations'].append("Review low-effectiveness minimization rules")

        return monitoring_results

    def export_minimization_report(self, assessment: DataMinimizationAssessment,
                                 output_format: str = 'json') -> str:
        """Export data minimization assessment report"""
        if output_format == 'json':
            return json.dumps(self._assessment_to_dict(assessment), indent=2, default=str)
        else:
            raise ValueError(f"Unsupported format: {output_format}")

    def _assessment_to_dict(self, assessment: DataMinimizationAssessment) -> Dict[str, Any]:
        """Convert assessment to dictionary"""
        return {
            'assessment_id': assessment.assessment_id,
            'assessment_date': assessment.assessment_date.isoformat(),
            'data_fields_analyzed': assessment.data_fields_analyzed,
            'minimization_rules_applied': assessment.minimization_rules_applied,
            'data_reduction_percentage': assessment.data_reduction_percentage,
            'privacy_risk_reduction': assessment.privacy_risk_reduction,
            'recommendations': assessment.recommendations,
            'implementation_gaps': assessment.implementation_gaps,
            'next_review_date': assessment.next_review_date.isoformat()
        }
