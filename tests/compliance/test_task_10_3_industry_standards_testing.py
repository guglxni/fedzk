#!/usr/bin/env python3
"""
FEDZK Task 10.3.3: Industry Standards Testing

Comprehensive testing suite for Task 10.2.2 Industry Standards components:
- IndustryStandardsCompliance
- NISTCompliance
- ISO27001Compliance
- SOC2Compliance
"""

import unittest
import json
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta

# Add src to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))

from fedzk.compliance.regulatory.industry_standards import (
    IndustryStandardsCompliance, NISTCompliance, ISO27001Compliance, SOC2Compliance,
    ComplianceAssessment, ComplianceControl, ComplianceStatus
)
from fedzk.compliance.regulatory.industry_standards import (
    NISTFunction, ISO27001Control
)


class TestIndustryStandardsCompliance(unittest.TestCase):
    """Test suite for IndustryStandardsCompliance component"""

    def setUp(self):
        """Set up test fixtures"""
        self.standards_compliance = IndustryStandardsCompliance("FEDZK")

    def test_industry_standards_compliance_initialization(self):
        """Test IndustryStandardsCompliance initialization"""
        self.assertIsNotNone(self.standards_compliance)
        self.assertEqual(self.standards_compliance.organization, "FEDZK")

    def test_nist_compliance_assessment(self):
        """Test NIST compliance assessment access"""
        nist_compliance = self.standards_compliance.assess_nist_compliance()

        self.assertIsInstance(nist_compliance, NISTCompliance)
        self.assertEqual(nist_compliance.organization, "FEDZK")

    def test_iso27001_compliance_assessment(self):
        """Test ISO 27001 compliance assessment access"""
        iso27001_compliance = self.standards_compliance.assess_iso27001_compliance()

        self.assertIsInstance(iso27001_compliance, ISO27001Compliance)
        self.assertEqual(iso27001_compliance.organization, "FEDZK")

    def test_soc2_compliance_assessment(self):
        """Test SOC 2 compliance assessment access"""
        soc2_compliance = self.standards_compliance.assess_soc2_compliance()

        self.assertIsInstance(soc2_compliance, SOC2Compliance)
        self.assertEqual(soc2_compliance.organization, "FEDZK")

    def test_comprehensive_assessment_execution(self):
        """Test comprehensive compliance assessment execution"""
        assessments = self.standards_compliance.perform_comprehensive_assessment()

        self.assertIsInstance(assessments, dict)
        self.assertIn('NIST', assessments)
        self.assertIn('ISO27001', assessments)
        self.assertIn('SOC2', assessments)

        # Verify assessment structure
        for framework, assessment in assessments.items():
            self.assertIsInstance(assessment, ComplianceAssessment)
            # Framework name might be more detailed than the key
            if framework == 'NIST':
                self.assertEqual(assessment.framework, "NIST Cybersecurity Framework")
            elif framework == 'ISO27001':
                self.assertEqual(assessment.framework, "ISO 27001")
            elif framework == 'SOC2':
                self.assertEqual(assessment.framework, "SOC 2")
            self.assertIsInstance(assessment.overall_compliance, float)
            self.assertIsInstance(assessment.controls_assessed, list)

    def test_compliance_report_generation(self):
        """Test compliance report generation"""
        report = self.standards_compliance.generate_compliance_report()

        self.assertIsInstance(report, dict)
        self.assertIn('report_id', report)
        self.assertIn('organization', report)
        self.assertIn('assessment_date', report)
        self.assertIn('framework_assessments', report)
        self.assertIn('overall_compliance_scores', report)
        self.assertIn('critical_gaps', report)
        self.assertIn('consolidated_recommendations', report)


class TestNISTCompliance(unittest.TestCase):
    """Test suite for NISTCompliance component"""

    def setUp(self):
        """Set up test fixtures"""
        self.nist_compliance = NISTCompliance("FEDZK")

    def test_nist_compliance_initialization(self):
        """Test NISTCompliance initialization"""
        self.assertIsNotNone(self.nist_compliance)
        self.assertEqual(self.nist_compliance.organization, "FEDZK")

    def test_nist_assessment_execution(self):
        """Test NIST assessment execution"""
        assessment = self.nist_compliance.perform_assessment()

        self.assertIsInstance(assessment, ComplianceAssessment)
        self.assertEqual(assessment.framework, "NIST Cybersecurity Framework")
        self.assertEqual(assessment.assessor, "Compliance Team")
        self.assertIsInstance(assessment.overall_compliance, float)
        self.assertGreaterEqual(assessment.overall_compliance, 0.0)
        self.assertLessEqual(assessment.overall_compliance, 100.0)

    def test_nist_controls_structure(self):
        """Test NIST controls structure"""
        assessment = self.nist_compliance.perform_assessment()

        self.assertIsInstance(assessment.controls_assessed, list)
        self.assertGreater(len(assessment.controls_assessed), 0)

        # Check control structure
        for control in assessment.controls_assessed:
            self.assertIsInstance(control, ComplianceControl)
            self.assertIsNotNone(control.id)
            self.assertIsNotNone(control.title)
            self.assertEqual(control.framework, "NIST CSF")
            self.assertIn(control.category, [f.value for f in NISTFunction])

    def test_nist_compliance_scoring(self):
        """Test NIST compliance scoring logic"""
        assessment = self.nist_compliance.perform_assessment()

        # Verify scoring is reasonable
        self.assertIsInstance(assessment.overall_compliance, float)
        self.assertGreaterEqual(assessment.overall_compliance, 0.0)
        self.assertLessEqual(assessment.overall_compliance, 100.0)

        # Check that controls contribute to scoring
        total_controls = len(assessment.controls_assessed)
        compliant_controls = sum(1 for c in assessment.controls_assessed
                                if c.status == ComplianceStatus.COMPLIANT)

        # Calculate partial count
        partial_count = sum(1 for c in assessment.controls_assessed
                           if c.status == ComplianceStatus.PARTIALLY_COMPLIANT)

        # Allow for some flexibility in scoring calculation
        expected_score = ((compliant_controls * 1.0 + partial_count * 0.5) / total_controls) * 100
        self.assertAlmostEqual(assessment.overall_compliance, expected_score, delta=5.0)

    def test_nist_critical_gaps_identification(self):
        """Test NIST critical gaps identification"""
        assessment = self.nist_compliance.perform_assessment()

        self.assertIsInstance(assessment.critical_gaps, list)
        self.assertIsInstance(assessment.recommendations, list)

        # Should have some recommendations
        self.assertGreater(len(assessment.recommendations), 0)


class TestISO27001Compliance(unittest.TestCase):
    """Test suite for ISO27001Compliance component"""

    def setUp(self):
        """Set up test fixtures"""
        self.iso27001_compliance = ISO27001Compliance("FEDZK")

    def test_iso27001_compliance_initialization(self):
        """Test ISO27001Compliance initialization"""
        self.assertIsNotNone(self.iso27001_compliance)
        self.assertEqual(self.iso27001_compliance.organization, "FEDZK")

    def test_iso27001_assessment_execution(self):
        """Test ISO 27001 assessment execution"""
        assessment = self.iso27001_compliance.perform_assessment()

        self.assertIsInstance(assessment, ComplianceAssessment)
        self.assertEqual(assessment.framework, "ISO 27001")
        self.assertEqual(assessment.assessor, "ISMS Team")
        self.assertIsInstance(assessment.overall_compliance, float)

    def test_iso27001_controls_structure(self):
        """Test ISO 27001 controls structure"""
        assessment = self.iso27001_compliance.perform_assessment()

        self.assertIsInstance(assessment.controls_assessed, list)
        self.assertGreater(len(assessment.controls_assessed), 0)

        # Check control structure
        for control in assessment.controls_assessed:
            self.assertIsInstance(control, ComplianceControl)
            self.assertIsNotNone(control.id)
            self.assertIsNotNone(control.title)
            self.assertEqual(control.framework, "ISO 27001")
            self.assertTrue(control.category.startswith("A."))

    def test_iso27001_compliance_scoring(self):
        """Test ISO 27001 compliance scoring"""
        assessment = self.iso27001_compliance.perform_assessment()

        self.assertIsInstance(assessment.overall_compliance, float)
        self.assertGreaterEqual(assessment.overall_compliance, 0.0)
        self.assertLessEqual(assessment.overall_compliance, 100.0)

    def test_iso27001_control_categories(self):
        """Test ISO 27001 control categories"""
        assessment = self.iso27001_compliance.perform_assessment()

        categories = set()
        for control in assessment.controls_assessed:
            categories.add(control.category)

        # Should have multiple control categories
        self.assertGreater(len(categories), 1)

        # All categories should be valid ISO 27001 controls
        valid_categories = [c.value for c in ISO27001Control]
        for category in categories:
            self.assertIn(category, valid_categories)


class TestSOC2Compliance(unittest.TestCase):
    """Test suite for SOC2Compliance component"""

    def setUp(self):
        """Set up test fixtures"""
        self.soc2_compliance = SOC2Compliance("FEDZK")

    def test_soc2_compliance_initialization(self):
        """Test SOC2Compliance initialization"""
        self.assertIsNotNone(self.soc2_compliance)
        self.assertEqual(self.soc2_compliance.organization, "FEDZK")

    def test_soc2_assessment_execution(self):
        """Test SOC 2 assessment execution"""
        assessment = self.soc2_compliance.perform_assessment()

        self.assertIsInstance(assessment, ComplianceAssessment)
        self.assertEqual(assessment.framework, "SOC 2")
        self.assertEqual(assessment.assessor, "Audit Team")
        self.assertIsInstance(assessment.overall_compliance, float)

    def test_soc2_controls_structure(self):
        """Test SOC 2 controls structure"""
        assessment = self.soc2_compliance.perform_assessment()

        self.assertIsInstance(assessment.controls_assessed, list)
        self.assertGreater(len(assessment.controls_assessed), 0)

        # Check control structure
        for control in assessment.controls_assessed:
            self.assertIsInstance(control, ComplianceControl)
            self.assertIsNotNone(control.id)
            self.assertIsNotNone(control.title)
            self.assertEqual(control.framework, "SOC 2")

    def test_soc2_compliance_scoring(self):
        """Test SOC 2 compliance scoring"""
        assessment = self.soc2_compliance.perform_assessment()

        self.assertIsInstance(assessment.overall_compliance, float)
        self.assertGreaterEqual(assessment.overall_compliance, 0.0)
        self.assertLessEqual(assessment.overall_compliance, 100.0)

    def test_soc2_trust_service_categories(self):
        """Test SOC 2 trust service categories"""
        assessment = self.soc2_compliance.perform_assessment()

        categories = set()
        for control in assessment.controls_assessed:
            categories.add(control.category)

        # Should include main trust service categories
        expected_categories = {"Security", "Availability", "Processing Integrity", "Confidentiality", "Privacy"}
        for category in categories:
            self.assertIn(category, expected_categories)


class TestComplianceControl(unittest.TestCase):
    """Test suite for ComplianceControl data structure"""

    def test_compliance_control_creation(self):
        """Test ComplianceControl creation"""
        control = ComplianceControl(
            id="TEST_001",
            title="Test Control",
            description="Test control description",
            framework="Test Framework",
            category="Test Category",
            status=ComplianceStatus.COMPLIANT,
            evidence_required=True,
            evidence_path="/test/evidence.json",
            implementation_notes="Test implementation",
            last_assessed=datetime.now(),
            next_assessment=datetime.now() + timedelta(days=365),
            remediation_required=False,
            remediation_plan=""
        )

        self.assertEqual(control.id, "TEST_001")
        self.assertEqual(control.title, "Test Control")
        self.assertEqual(control.framework, "Test Framework")
        self.assertEqual(control.status, ComplianceStatus.COMPLIANT)
        self.assertTrue(control.evidence_required)
        self.assertFalse(control.remediation_required)

    def test_compliance_control_status_values(self):
        """Test ComplianceControl status values"""
        for status in ComplianceStatus:
            control = ComplianceControl(
                id=f"TEST_{status.value}",
                title=f"Test {status.value}",
                description="Test description",
                framework="Test Framework",
                category="Test Category",
                status=status,
                evidence_required=False,
                evidence_path=None,
                implementation_notes="",
                last_assessed=datetime.now(),
                next_assessment=datetime.now() + timedelta(days=365),
                remediation_required=False,
                remediation_plan=""
            )
            self.assertEqual(control.status, status)


class TestComplianceAssessment(unittest.TestCase):
    """Test suite for ComplianceAssessment data structure"""

    def test_compliance_assessment_creation(self):
        """Test ComplianceAssessment creation"""
        controls = [
            ComplianceControl(
                id="CTRL_001",
                title="Test Control 1",
                description="Test control 1",
                framework="Test Framework",
                category="Test Category",
                status=ComplianceStatus.COMPLIANT,
                evidence_required=False,
                evidence_path=None,
                implementation_notes="",
                last_assessed=datetime.now(),
                next_assessment=datetime.now() + timedelta(days=365),
                remediation_required=False,
                remediation_plan=""
            ),
            ComplianceControl(
                id="CTRL_002",
                title="Test Control 2",
                description="Test control 2",
                framework="Test Framework",
                category="Test Category",
                status=ComplianceStatus.PARTIALLY_COMPLIANT,
                evidence_required=False,
                evidence_path=None,
                implementation_notes="",
                last_assessed=datetime.now(),
                next_assessment=datetime.now() + timedelta(days=365),
                remediation_required=True,
                remediation_plan="Fix implementation"
            )
        ]

        assessment = ComplianceAssessment(
            assessment_id="ASSESS_001",
            framework="Test Framework",
            assessment_date=datetime.now(),
            assessor="Test Assessor",
            overall_compliance=75.0,
            controls_assessed=controls,
            critical_gaps=["Gap 1", "Gap 2"],
            recommendations=["Rec 1", "Rec 2"],
            next_assessment_date=datetime.now() + timedelta(days=365)
        )

        self.assertEqual(assessment.assessment_id, "ASSESS_001")
        self.assertEqual(assessment.framework, "Test Framework")
        self.assertEqual(assessment.overall_compliance, 75.0)
        self.assertEqual(len(assessment.controls_assessed), 2)
        self.assertEqual(len(assessment.critical_gaps), 2)
        self.assertEqual(len(assessment.recommendations), 2)


class TestIndustryStandardsIntegration(unittest.TestCase):
    """Integration tests for industry standards components"""

    def setUp(self):
        """Set up test fixtures"""
        self.standards_compliance = IndustryStandardsCompliance("FEDZK")

    def test_cross_framework_consistency(self):
        """Test consistency across different frameworks"""
        assessments = self.standards_compliance.perform_comprehensive_assessment()

        # All assessments should have consistent structure
        for framework, assessment in assessments.items():
            self.assertIsInstance(assessment.overall_compliance, float)
            self.assertIsInstance(assessment.controls_assessed, list)
            self.assertGreater(len(assessment.controls_assessed), 0)
            self.assertIsInstance(assessment.critical_gaps, list)
            self.assertIsInstance(assessment.recommendations, list)

    def test_compliance_score_calculation_consistency(self):
        """Test compliance score calculation consistency"""
        assessments = self.standards_compliance.perform_comprehensive_assessment()

        for framework, assessment in assessments.items():
            score = assessment.overall_compliance
            self.assertGreaterEqual(score, 0.0)
            self.assertLessEqual(score, 100.0)

            # Score should be based on control status
            total_controls = len(assessment.controls_assessed)
            compliant_count = sum(1 for c in assessment.controls_assessed
                                 if c.status == ComplianceStatus.COMPLIANT)
            partial_count = sum(1 for c in assessment.controls_assessed
                               if c.status == ComplianceStatus.PARTIALLY_COMPLIANT)

            expected_score = ((compliant_count * 1.0 + partial_count * 0.5) / total_controls) * 100
            self.assertAlmostEqual(score, expected_score, places=1)

    def test_framework_specific_requirements(self):
        """Test framework-specific requirements"""
        assessments = self.standards_compliance.perform_comprehensive_assessment()

        # NIST should have function categories
        nist_assessment = assessments['NIST']
        nist_categories = set(c.category for c in nist_assessment.controls_assessed)
        expected_nist_functions = {f.value for f in NISTFunction}
        self.assertTrue(nist_categories.issubset(expected_nist_functions))

        # ISO 27001 should have control categories starting with A.
        iso_assessment = assessments['ISO27001']
        iso_categories = set(c.category for c in iso_assessment.controls_assessed)
        self.assertTrue(all(cat.startswith('A.') for cat in iso_categories))

        # SOC 2 should have trust service categories
        soc2_assessment = assessments['SOC2']
        soc2_categories = set(c.category for c in soc2_assessment.controls_assessed)
        expected_soc2_categories = {"Security", "Availability", "Processing Integrity", "Confidentiality", "Privacy"}
        self.assertTrue(soc2_categories.issubset(expected_soc2_categories))

    def test_comprehensive_report_integration(self):
        """Test comprehensive compliance report integration"""
        report = self.standards_compliance.generate_compliance_report()

        self.assertIn('report_id', report)
        self.assertIn('organization', report)
        self.assertIn('framework_assessments', report)
        self.assertIn('overall_compliance_scores', report)

        # Verify all frameworks are included
        framework_assessments = report['framework_assessments']
        self.assertIn('NIST', framework_assessments)
        self.assertIn('ISO27001', framework_assessments)
        self.assertIn('SOC2', framework_assessments)

        # Verify overall scores are calculated
        overall_scores = report['overall_compliance_scores']
        self.assertIn('NIST', overall_scores)
        self.assertIn('ISO27001', overall_scores)
        self.assertIn('SOC2', overall_scores)

        # Verify consolidated data
        self.assertIn('critical_gaps', report)
        self.assertIn('consolidated_recommendations', report)
        self.assertIn('summary', report)


if __name__ == '__main__':
    unittest.main()
