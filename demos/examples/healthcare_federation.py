#!/usr/bin/env python3
"""
FEDZK Healthcare Federation Example

This example demonstrates how to create a healthcare-focused federated learning
federation for collaborative medical research while maintaining patient privacy.

Use Case: Multiple hospitals collaborate to train a diagnostic model for detecting
pneumonia from chest X-rays without sharing patient data.

Key Features Demonstrated:
- Healthcare-specific privacy configurations
- Multi-institutional collaboration
- Medical data handling
- Compliance with healthcare regulations (HIPAA, GDPR)
- Differential privacy for medical data
"""

import asyncio
import logging
from pathlib import Path
from typing import Dict, List, Any

# FEDZK imports
from fedzk.core import FederatedLearning, ModelConfig, PrivacyConfig
from fedzk.privacy import DifferentialPrivacy, ZeroKnowledgeProofs
from fedzk.compliance import HIPAACompliance, GDPRCompliance
from fedzk.monitoring import HealthcareMetrics

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class HealthcareFederation:
    """
    Healthcare-focused federated learning federation.

    This class provides healthcare-specific configurations and compliance
    measures for medical federated learning scenarios.
    """

    def __init__(self, federation_name: str):
        self.federation_name = federation_name
        self.federation = None
        self.participants = []
        self.metrics = HealthcareMetrics()

        # Healthcare-specific privacy requirements
        self.privacy_requirements = {
            'epsilon': 0.1,  # Very strict privacy for medical data
            'delta': 1e-6,
            'hipaa_compliant': True,
            'gdpr_compliant': True,
            'data_retention_days': 365,
            'audit_trail_enabled': True
        }

    def create_healthcare_federation(self) -> FederatedLearning:
        """
        Create a healthcare-focused federated learning federation.

        Returns:
            Configured FederatedLearning instance
        """

        # Healthcare-specific model configuration
        model_config = ModelConfig(
            model_type='convolutional_neural_network',
            input_shape=(224, 224, 3),  # Standard medical image size
            output_shape=(2,),  # Binary classification: pneumonia/no pneumonia
            architecture={
                'layers': [
                    {'type': 'conv2d', 'filters': 32, 'kernel_size': (3, 3)},
                    {'type': 'maxpool2d', 'pool_size': (2, 2)},
                    {'type': 'conv2d', 'filters': 64, 'kernel_size': (3, 3)},
                    {'type': 'maxpool2d', 'pool_size': (2, 2)},
                    {'type': 'conv2d', 'filters': 128, 'kernel_size': (3, 3)},
                    {'type': 'maxpool2d', 'pool_size': (2, 2)},
                    {'type': 'flatten'},
                    {'type': 'dense', 'units': 512, 'activation': 'relu'},
                    {'type': 'dropout', 'rate': 0.5},
                    {'type': 'dense', 'units': 2, 'activation': 'softmax'}
                ],
                'optimizer': 'adam',
                'loss': 'categorical_crossentropy',
                'metrics': ['accuracy', 'precision', 'recall', 'f1_score']
            }
        )

        # Healthcare-specific privacy configuration
        privacy_config = PrivacyConfig(
            enable_zk_proofs=True,
            differential_privacy={
                'epsilon': self.privacy_requirements['epsilon'],
                'delta': self.privacy_requirements['delta'],
                'noise_mechanism': 'gaussian',
                'clipping_norm': 1.0,
                'adaptive_noise': True
            },
            encryption='AES-256-GCM',
            secure_aggregation=True,
            audit_trail_enabled=True,
            compliance_frameworks=['HIPAA', 'GDPR']
        )

        # Create the federation
        self.federation = FederatedLearning.create_federation(
            name=self.federation_name,
            description="Collaborative pneumonia detection from chest X-rays",
            model_config=model_config,
            privacy_config=privacy_config,
            min_participants=5,  # Require at least 5 hospitals
            max_participants=20,
            domain='healthcare',
            compliance_requirements=['HIPAA', 'GDPR']
        )

        logger.info(f"Created healthcare federation: {self.federation.id}")
        return self.federation

    def add_hospital_participant(self, hospital_info: Dict[str, Any]) -> str:
        """
        Add a hospital as a participant in the federation.

        Args:
            hospital_info: Dictionary containing hospital information

        Returns:
            Participant ID
        """

        required_fields = ['name', 'location', 'data_size', 'specialties']
        for field in required_fields:
            if field not in hospital_info:
                raise ValueError(f"Missing required field: {field}")

        # Generate cryptographic keys for the hospital
        key_manager = self.federation.get_key_manager()
        public_key, private_key = key_manager.generate_healthcare_keys()

        # Create participant with healthcare-specific metadata
        participant_data = {
            'participant_id': f"hospital_{hospital_info['name'].lower().replace(' ', '_')}",
            'organization_type': 'hospital',
            'public_key': public_key,
            'metadata': {
                'hospital_name': hospital_info['name'],
                'location': hospital_info['location'],
                'data_size': hospital_info['data_size'],
                'specialties': hospital_info['specialties'],
                'hipaa_compliant': True,
                'irb_approved': True,
                'data_retention_policy': '7_years',
                'anonymization_level': 'full'
            }
        }

        # Join the federation
        participant = self.federation.join(**participant_data)
        self.participants.append(participant)

        logger.info(f"Added hospital participant: {participant.id}")
        return participant.id

    def configure_medical_data_processing(self, data_config: Dict[str, Any]):
        """
        Configure data processing for medical images and records.

        Args:
            data_config: Configuration for medical data processing
        """

        # Set up medical image preprocessing
        self.federation.configure_data_processing({
            'data_type': 'medical_images',
            'preprocessing': {
                'image_normalization': True,
                'clahe_enhancement': True,  # Contrast Limited Adaptive Histogram Equalization
                'noise_reduction': True,
                'standardization': {
                    'target_size': (224, 224),
                    'color_mode': 'rgb',
                    'rescale_factor': 1.0/255.0
                }
            },
            'privacy_filters': {
                'phi_detection': True,  # Protected Health Information detection
                'face_detection': True,  # Remove faces from X-rays
                'text_removal': True,   # Remove any text overlays
                'metadata_stripping': True
            },
            'quality_checks': {
                'image_quality_assessment': True,
                'artifact_detection': True,
                'completeness_check': True
            }
        })

        # Configure differential privacy for medical data
        dp_config = DifferentialPrivacy.configure_for_healthcare(
            epsilon=self.privacy_requirements['epsilon'],
            sensitivity_bounds={
                'pixel_intensity': [0, 255],
                'diagnostic_confidence': [0, 1]
            }
        )

        self.federation.set_differential_privacy_config(dp_config)

    def setup_healthcare_compliance(self):
        """Set up healthcare-specific compliance measures."""

        # Configure HIPAA compliance
        hipaa_compliance = HIPAACompliance()
        hipaa_config = {
            'business_associate_agreement': True,
            'data_encryption': 'AES-256-GCM',
            'access_controls': 'role_based',
            'audit_logging': True,
            'breach_notification': True,
            'data_retention': '7_years'
        }

        hipaa_compliance.configure(hipaa_config)
        self.federation.register_compliance_framework(hipaa_compliance)

        # Configure GDPR compliance for medical data
        gdpr_compliance = GDPRCompliance()
        gdpr_config = {
            'data_processing_grounds': 'consent_and_legitimate_interest',
            'data_minimization': True,
            'purpose_limitation': True,
            'storage_limitation': '7_years',
            'data_subject_rights': ['access', 'rectification', 'erasure'],
            'automated_decision_making': False
        }

        gdpr_compliance.configure(gdpr_config)
        self.federation.register_compliance_framework(gdpr_compliance)

        logger.info("Healthcare compliance frameworks configured")

    def start_medical_federated_training(self, training_config: Dict[str, Any]):
        """
        Start federated training for medical diagnosis.

        Args:
            training_config: Training configuration parameters
        """

        # Validate training configuration for medical use case
        self._validate_medical_training_config(training_config)

        # Set up healthcare-specific training parameters
        medical_training_config = {
            **training_config,
            'medical_validation': {
                'cross_validation_folds': 5,
                'clinical_validation_required': True,
                'bias_fairness_checks': True,
                'interpretability_requirements': True
            },
            'privacy_preserving_techniques': {
                'secure_aggregation': True,
                'zk_proof_validation': True,
                'differential_privacy': True,
                'homomorphic_encryption': False  # Can be enabled for additional security
            }
        }

        # Start the training session
        training_session = self.federation.start_training(
            training_config=medical_training_config
        )

        logger.info(f"Started medical federated training: {training_session.id}")
        return training_session

    def _validate_medical_training_config(self, config: Dict[str, Any]):
        """Validate training configuration for medical use case."""

        required_fields = ['epochs', 'batch_size', 'learning_rate']
        for field in required_fields:
            if field not in config:
                raise ValueError(f"Missing required training parameter: {field}")

        # Medical-specific validations
        if config.get('batch_size', 0) < 16:
            raise ValueError("Batch size too small for medical imaging tasks")

        if config.get('learning_rate', 0) > 0.01:
            logger.warning("Learning rate may be too high for medical imaging convergence")

        # Check for medical-specific requirements
        if not config.get('data_augmentation', False):
            logger.warning("Data augmentation recommended for medical imaging robustness")

    def monitor_healthcare_training(self, training_session):
        """Monitor training progress with healthcare-specific metrics."""

        def medical_progress_callback(progress_data):
            """Callback for monitoring medical training progress."""

            # Extract medical-specific metrics
            metrics = {
                'epoch': progress_data.get('epoch'),
                'accuracy': progress_data.get('accuracy'),
                'precision': progress_data.get('precision'),
                'recall': progress_data.get('recall'),
                'f1_score': progress_data.get('f1_score'),
                'privacy_budget_remaining': progress_data.get('privacy_budget'),
                'zk_proofs_verified': progress_data.get('zk_proofs_verified')
            }

            # Healthcare-specific monitoring
            if metrics['accuracy'] > 0.95:
                logger.info("High accuracy achieved - consider early stopping")
            elif metrics['accuracy'] < 0.7:
                logger.warning("Low accuracy detected - check data quality or model architecture")

            # Privacy budget monitoring
            if metrics['privacy_budget_remaining'] < 0.1:
                logger.warning("Privacy budget running low - consider stopping training")

            # Log metrics to healthcare monitoring system
            self.metrics.record_healthcare_metrics(metrics)

        # Set up progress monitoring
        training_session.set_progress_callback(medical_progress_callback)

        return training_session

    def generate_healthcare_report(self, training_session) -> Dict[str, Any]:
        """Generate comprehensive healthcare training report."""

        # Get training results
        results = training_session.get_final_results()

        # Generate compliance report
        compliance_report = self.federation.generate_compliance_report()

        # Create healthcare-specific report
        healthcare_report = {
            'federation_name': self.federation_name,
            'training_session_id': training_session.id,
            'model_performance': {
                'accuracy': results.get('accuracy'),
                'precision': results.get('precision'),
                'recall': results.get('recall'),
                'f1_score': results.get('f1_score'),
                'auc_roc': results.get('auc_roc')
            },
            'privacy_metrics': {
                'epsilon_used': results.get('epsilon_used'),
                'privacy_budget_remaining': results.get('privacy_budget_remaining'),
                'zk_proofs_generated': results.get('zk_proofs_generated'),
                'zk_proofs_verified': results.get('zk_proofs_verified')
            },
            'compliance_status': {
                'hipaa_compliant': compliance_report.get('hipaa_compliant'),
                'gdpr_compliant': compliance_report.get('gdpr_compliant'),
                'data_minimization_applied': compliance_report.get('data_minimization_applied')
            },
            'participant_contributions': results.get('participant_contributions', []),
            'training_metadata': {
                'total_epochs': results.get('total_epochs'),
                'total_participants': len(self.participants),
                'training_duration': results.get('training_duration'),
                'model_size_mb': results.get('model_size_mb')
            },
            'clinical_validation': {
                'cross_validation_score': results.get('cross_validation_score'),
                'clinical_trial_ready': results.get('clinical_trial_ready', False),
                'bias_assessment_completed': results.get('bias_assessment_completed', False)
            }
        }

        return healthcare_report


async def main():
    """Main example execution."""

    print("🏥 FEDZK Healthcare Federation Example")
    print("=" * 50)

    # Create healthcare federation
    healthcare_fed = HealthcareFederation("Global Pneumonia Detection Network")

    # Create the federation
    federation = healthcare_fed.create_healthcare_federation()
    print(f"✅ Created federation: {federation.id}")

    # Add hospital participants
    hospitals = [
        {
            'name': 'Mayo Clinic',
            'location': 'Rochester, MN',
            'data_size': 50000,
            'specialties': ['radiology', 'pulmonology']
        },
        {
            'name': 'Johns Hopkins Hospital',
            'location': 'Baltimore, MD',
            'data_size': 75000,
            'specialties': ['radiology', 'infectious_diseases']
        },
        {
            'name': 'Cleveland Clinic',
            'location': 'Cleveland, OH',
            'data_size': 60000,
            'specialties': ['radiology', 'emergency_medicine']
        }
    ]

    for hospital in hospitals:
        participant_id = healthcare_fed.add_hospital_participant(hospital)
        print(f"✅ Added hospital: {hospital['name']} ({participant_id})")

    # Configure medical data processing
    healthcare_fed.configure_medical_data_processing({
        'image_format': 'DICOM',
        'preprocessing_enabled': True,
        'privacy_filters_enabled': True,
        'quality_checks_enabled': True
    })

    # Set up healthcare compliance
    healthcare_fed.setup_healthcare_compliance()

    # Configure and start training
    training_config = {
        'epochs': 50,
        'batch_size': 32,
        'learning_rate': 0.001,
        'optimizer': 'adam',
        'loss_function': 'categorical_crossentropy',
        'metrics': ['accuracy', 'precision', 'recall', 'f1_score'],
        'early_stopping': {
            'patience': 10,
            'min_delta': 0.001,
            'restore_best_weights': True
        },
        'validation_split': 0.2,
        'cross_validation_folds': 5
    }

    # Start training
    training_session = healthcare_fed.start_medical_federated_training(training_config)
    print(f"🚀 Started training session: {training_session.id}")

    # Monitor training
    healthcare_fed.monitor_healthcare_training(training_session)

    # Wait for training to complete (in real scenario, this would be asynchronous)
    print("⏳ Training in progress... (simulated)")
    await asyncio.sleep(5)  # Simulate training time

    # Generate final report
    final_report = healthcare_fed.generate_healthcare_report(training_session)

    print("\n📊 Final Healthcare Training Report")
    print("-" * 40)
    print(f"Model Accuracy: {final_report['model_performance']['accuracy']:.3f}")
    print(f"Privacy Budget Used: {final_report['privacy_metrics']['epsilon_used']:.3f}")
    print(f"HIPAA Compliant: {final_report['compliance_status']['hipaa_compliant']}")
    print(f"GDPR Compliant: {final_report['compliance_status']['gdpr_compliant']}")
    print(f"Participants: {final_report['training_metadata']['total_participants']}")

    print("\n✅ Healthcare federation example completed successfully!")
    print("This demonstrates how FEDZK enables privacy-preserving collaborative")
    print("medical research across multiple institutions.")


if __name__ == "__main__":
    asyncio.run(main())
