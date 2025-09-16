#!/usr/bin/env python3
"""
FEDZK Financial Risk Assessment Example

This example demonstrates how to create a financial services federation for
collaborative fraud detection and risk assessment while maintaining regulatory
compliance and data privacy.

Use Case: Multiple banks collaborate to train fraud detection models and assess
credit risk without sharing customer transaction data.

Key Features Demonstrated:
- Financial industry compliance (SOX, GLBA)
- Multi-bank collaboration
- Financial data handling and privacy
- Regulatory reporting capabilities
- Risk assessment modeling
"""

import asyncio
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
import pandas as pd
import numpy as np

# FEDZK imports
from fedzk.core import FederatedLearning, ModelConfig, PrivacyConfig
from fedzk.privacy import DifferentialPrivacy, SecureAggregation
from fedzk.compliance import SOXCompliance, GLBACompliance, SOC2Compliance
from fedzk.monitoring import FinancialMetrics

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FinancialFederation:
    """
    Financial services-focused federated learning federation.

    This class provides financial industry-specific configurations and compliance
    measures for banking and financial services federated learning scenarios.
    """

    def __init__(self, federation_name: str):
        self.federation_name = federation_name
        self.federation = None
        self.participants = []
        self.metrics = FinancialMetrics()

        # Financial industry privacy and compliance requirements
        self.compliance_requirements = {
            'sox_compliant': True,
            'glba_compliant': True,
            'soc2_compliant': True,
            'data_retention_years': 7,
            'audit_trail_enabled': True,
            'regulatory_reporting': True
        }

        # Financial data privacy requirements
        self.privacy_requirements = {
            'epsilon': 0.5,  # Balanced privacy-utility for financial data
            'delta': 1e-6,
            'pii_detection': True,
            'pci_compliance': True,
            'data_anonymization': 'k_anonymity'
        }

    def create_financial_federation(self) -> FederatedLearning:
        """
        Create a financial services-focused federated learning federation.

        Returns:
            Configured FederatedLearning instance
        """

        # Financial risk assessment model configuration
        model_config = ModelConfig(
            model_type='gradient_boosting',
            input_shape=(50,),  # Financial features (age, income, credit_score, etc.)
            output_shape=(3,),  # Risk categories: low, medium, high
            architecture={
                'algorithm': 'xgboost',
                'hyperparameters': {
                    'max_depth': 6,
                    'learning_rate': 0.1,
                    'n_estimators': 100,
                    'subsample': 0.8,
                    'colsample_bytree': 0.8
                },
                'objective': 'multi:softprob',
                'eval_metric': ['mlogloss', 'merror'],
                'early_stopping_rounds': 10
            }
        )

        # Financial industry privacy configuration
        privacy_config = PrivacyConfig(
            enable_zk_proofs=True,
            differential_privacy={
                'epsilon': self.privacy_requirements['epsilon'],
                'delta': self.privacy_requirements['delta'],
                'noise_mechanism': 'laplace',
                'clipping_norm': 2.0,
                'adaptive_noise': True
            },
            encryption='AES-256-GCM',
            secure_aggregation=True,
            audit_trail_enabled=True,
            compliance_frameworks=['SOX', 'GLBA', 'SOC2'],
            data_classification={
                'pii_fields': ['ssn', 'account_number', 'name', 'address'],
                'sensitive_fields': ['income', 'credit_score', 'transaction_amount'],
                'public_fields': ['age', 'gender', 'zip_code_prefix']
            }
        )

        # Create the federation
        self.federation = FederatedLearning.create_federation(
            name=self.federation_name,
            description="Collaborative financial risk assessment and fraud detection",
            model_config=model_config,
            privacy_config=privacy_config,
            min_participants=3,  # Require at least 3 financial institutions
            max_participants=15,
            domain='financial_services',
            compliance_requirements=['SOX', 'GLBA', 'SOC2']
        )

        logger.info(f"Created financial federation: {self.federation.id}")
        return self.federation

    def add_bank_participant(self, bank_info: Dict[str, Any]) -> str:
        """
        Add a bank as a participant in the federation.

        Args:
            bank_info: Dictionary containing bank information

        Returns:
            Participant ID
        """

        required_fields = ['name', 'location', 'customer_base', 'services']
        for field in required_fields:
            if field not in bank_info:
                raise ValueError(f"Missing required field: {field}")

        # Generate cryptographic keys for the bank
        key_manager = self.federation.get_key_manager()
        public_key, private_key = key_manager.generate_financial_keys()

        # Create participant with financial-specific metadata
        participant_data = {
            'participant_id': f"bank_{bank_info['name'].lower().replace(' ', '_')}",
            'organization_type': 'financial_institution',
            'public_key': public_key,
            'metadata': {
                'bank_name': bank_info['name'],
                'location': bank_info['location'],
                'customer_base': bank_info['customer_base'],
                'services': bank_info['services'],
                'regulatory_approvals': ['FDIC', 'OCC', 'FINRA'],
                'data_retention_policy': '7_years',
                'encryption_standard': 'AES-256',
                'audit_frequency': 'quarterly'
            }
        }

        # Join the federation
        participant = self.federation.join(**participant_data)
        self.participants.append(participant)

        logger.info(f"Added bank participant: {participant.id}")
        return participant.id

    def configure_financial_data_processing(self, data_config: Dict[str, Any]):
        """
        Configure data processing for financial transactions and customer data.

        Args:
            data_config: Configuration for financial data processing
        """

        # Set up financial data preprocessing
        self.federation.configure_data_processing({
            'data_type': 'financial_transactions',
            'preprocessing': {
                'feature_engineering': {
                    'transaction_velocity': True,
                    'amount_percentiles': True,
                    'time_based_features': True,
                    'behavioral_patterns': True
                },
                'normalization': {
                    'method': 'robust_scaler',  # Handles outliers in financial data
                    'feature_range': (-1, 1)
                },
                'categorical_encoding': {
                    'method': 'target_encoding',
                    'handle_unknown': 'value'
                }
            },
            'privacy_filters': {
                'pii_detection': True,
                'pci_compliance': True,
                'data_anonymization': True,
                'tokenization': {
                    'account_numbers': True,
                    'social_security': True,
                    'names': True
                }
            },
            'quality_checks': {
                'data_validation': True,
                'outlier_detection': True,
                'consistency_checks': True,
                'regulatory_compliance': True
            }
        })

        # Configure differential privacy for financial data
        dp_config = DifferentialPrivacy.configure_for_finance(
            epsilon=self.privacy_requirements['epsilon'],
            sensitivity_bounds={
                'transaction_amount': [0, 1000000],  # Up to $1M transactions
                'account_balance': [0, 10000000],   # Up to $10M balances
                'credit_score': [300, 850]          # Standard FICO range
            }
        )

        self.federation.set_differential_privacy_config(dp_config)

    def setup_financial_compliance(self):
        """Set up financial industry-specific compliance measures."""

        # Configure SOX compliance
        sox_compliance = SOXCompliance()
        sox_config = {
            'internal_controls': True,
            'financial_reporting': True,
            'audit_trail': True,
            'access_controls': 'role_based',
            'change_management': True,
            'documentation': True
        }

        sox_compliance.configure(sox_config)
        self.federation.register_compliance_framework(sox_compliance)

        # Configure GLBA compliance
        glba_compliance = GLBACompliance()
        glba_config = {
            'privacy_notice': True,
            'opt_out_procedures': True,
            'data_security_program': True,
            'service_provider_contracts': True,
            'incident_response_plan': True,
            'customer_data_protection': True
        }

        glba_compliance.configure(glba_config)
        self.federation.register_compliance_framework(glba_compliance)

        # Configure SOC 2 compliance
        soc2_compliance = SOC2Compliance()
        soc2_config = {
            'security_criteria': True,
            'availability_criteria': True,
            'processing_integrity': True,
            'confidentiality': True,
            'privacy_criteria': True,
            'independent_audit': True
        }

        soc2_compliance.configure(soc2_config)
        self.federation.register_compliance_framework(soc2_compliance)

        logger.info("Financial compliance frameworks configured")

    def start_financial_risk_training(self, training_config: Dict[str, Any]):
        """
        Start federated training for financial risk assessment.

        Args:
            training_config: Training configuration parameters
        """

        # Validate training configuration for financial use case
        self._validate_financial_training_config(training_config)

        # Set up financial-specific training parameters
        financial_training_config = {
            **training_config,
            'financial_validation': {
                'regulatory_validation': True,
                'model_explainability': True,
                'bias_fairness_checks': True,
                'economic_impact_assessment': True,
                'stress_testing': True
            },
            'risk_assessment': {
                'credit_risk_modeling': True,
                'fraud_detection': True,
                'market_risk_analysis': False,  # Can be enabled for additional models
                'operational_risk': False
            },
            'privacy_preserving_techniques': {
                'secure_aggregation': True,
                'zk_proof_validation': True,
                'differential_privacy': True,
                'federated_averaging': True
            }
        }

        # Start the training session
        training_session = self.federation.start_training(
            training_config=financial_training_config
        )

        logger.info(f"Started financial risk training: {training_session.id}")
        return training_session

    def _validate_financial_training_config(self, config: Dict[str, Any]):
        """Validate training configuration for financial use case."""

        required_fields = ['epochs', 'learning_rate', 'validation_split']
        for field in required_fields:
            if field not in config:
                raise ValueError(f"Missing required training parameter: {field}")

        # Financial-specific validations
        if config.get('learning_rate', 0) > 0.1:
            raise ValueError("Learning rate too high for financial model stability")

        if config.get('validation_split', 0) < 0.2:
            raise ValueError("Validation split too small for financial risk assessment")

        # Check for financial-specific requirements
        if not config.get('class_weight_balancing', False):
            logger.warning("Class weight balancing recommended for imbalanced financial datasets")

        if not config.get('early_stopping', False):
            logger.warning("Early stopping recommended to prevent overfitting in financial models")

    def monitor_financial_training(self, training_session):
        """Monitor training progress with financial-specific metrics."""

        def financial_progress_callback(progress_data):
            """Callback for monitoring financial training progress."""

            # Extract financial-specific metrics
            metrics = {
                'epoch': progress_data.get('epoch'),
                'accuracy': progress_data.get('accuracy'),
                'precision': progress_data.get('precision'),
                'recall': progress_data.get('recall'),
                'f1_score': progress_data.get('f1_score'),
                'auc_roc': progress_data.get('auc_roc'),
                'privacy_budget_remaining': progress_data.get('privacy_budget'),
                'regulatory_compliance_score': progress_data.get('compliance_score'),
                'false_positive_rate': progress_data.get('false_positive_rate'),
                'false_negative_rate': progress_data.get('false_negative_rate')
            }

            # Financial-specific monitoring
            if metrics.get('false_positive_rate', 0) > 0.05:
                logger.warning("High false positive rate detected - may impact customer experience")

            if metrics.get('false_negative_rate', 0) > 0.10:
                logger.warning("High false negative rate detected - may impact fraud detection")

            if metrics.get('privacy_budget_remaining', 1.0) < 0.2:
                logger.warning("Privacy budget running low - consider stopping training")

            # Log metrics to financial monitoring system
            self.metrics.record_financial_metrics(metrics)

        # Set up progress monitoring
        training_session.set_progress_callback(financial_progress_callback)

        return training_session

    def generate_financial_report(self, training_session) -> Dict[str, Any]:
        """Generate comprehensive financial training report."""

        # Get training results
        results = training_session.get_final_results()

        # Generate compliance report
        compliance_report = self.federation.generate_compliance_report()

        # Create financial-specific report
        financial_report = {
            'federation_name': self.federation_name,
            'training_session_id': training_session.id,
            'model_performance': {
                'accuracy': results.get('accuracy'),
                'precision': results.get('precision'),
                'recall': results.get('recall'),
                'f1_score': results.get('f1_score'),
                'auc_roc': results.get('auc_roc'),
                'false_positive_rate': results.get('false_positive_rate'),
                'false_negative_rate': results.get('false_negative_rate')
            },
            'risk_assessment_metrics': {
                'credit_risk_accuracy': results.get('credit_risk_accuracy'),
                'fraud_detection_precision': results.get('fraud_detection_precision'),
                'default_prediction_auc': results.get('default_prediction_auc')
            },
            'privacy_metrics': {
                'epsilon_used': results.get('epsilon_used'),
                'privacy_budget_remaining': results.get('privacy_budget_remaining'),
                'zk_proofs_generated': results.get('zk_proofs_generated'),
                'secure_aggregations_performed': results.get('secure_aggregations_performed')
            },
            'compliance_status': {
                'sox_compliant': compliance_report.get('sox_compliant'),
                'glba_compliant': compliance_report.get('glba_compliant'),
                'soc2_compliant': compliance_report.get('soc2_compliant'),
                'regulatory_approval_status': compliance_report.get('regulatory_approval_status')
            },
            'participant_contributions': results.get('participant_contributions', []),
            'training_metadata': {
                'total_epochs': results.get('total_epochs'),
                'total_participants': len(self.participants),
                'training_duration': results.get('training_duration'),
                'model_size_mb': results.get('model_size_mb'),
                'data_processed_gb': results.get('data_processed_gb')
            },
            'regulatory_validation': {
                'model_validation_passed': results.get('model_validation_passed', False),
                'fairness_assessment_completed': results.get('fairness_assessment_completed', False),
                'bias_testing_completed': results.get('bias_testing_completed', False),
                'stress_testing_completed': results.get('stress_testing_completed', False)
            }
        }

        return financial_report

    def perform_regulatory_stress_testing(self, model, test_scenarios: List[Dict[str, Any]]):
        """
        Perform regulatory stress testing on the trained model.

        Args:
            model: Trained federated model
            test_scenarios: List of stress test scenarios
        """

        stress_test_results = []

        for scenario in test_scenarios:
            scenario_name = scenario['name']
            test_data = scenario['data']
            expected_behavior = scenario['expected_behavior']

            logger.info(f"Running stress test: {scenario_name}")

            # Run model on stress test data
            predictions = model.predict(test_data)

            # Evaluate against regulatory requirements
            evaluation = self._evaluate_stress_test_results(
                predictions, expected_behavior, scenario
            )

            stress_test_results.append({
                'scenario': scenario_name,
                'passed': evaluation['passed'],
                'metrics': evaluation['metrics'],
                'regulatory_compliance': evaluation['regulatory_compliance']
            })

        return stress_test_results

    def _evaluate_stress_test_results(self, predictions, expected_behavior, scenario):
        """Evaluate stress test results against regulatory requirements."""

        # Implement regulatory evaluation logic
        # This would include checks for:
        # - False positive rates within acceptable limits
        # - Model stability under stress conditions
        # - Compliance with regulatory thresholds
        # - Fairness and bias assessments

        evaluation = {
            'passed': True,  # Would be determined by actual evaluation logic
            'metrics': {
                'max_false_positive_rate': 0.05,
                'model_stability_score': 0.95,
                'regulatory_threshold_compliance': 0.98
            },
            'regulatory_compliance': {
                'fair_lending_compliant': True,
                'bias_assessment_passed': True,
                'stress_test_passed': True
            }
        }

        return evaluation


async def main():
    """Main example execution."""

    print("🏦 FEDZK Financial Risk Assessment Example")
    print("=" * 50)

    # Create financial federation
    financial_fed = FinancialFederation("Global Banking Risk Network")

    # Create the federation
    federation = financial_fed.create_financial_federation()
    print(f"✅ Created federation: {federation.id}")

    # Add bank participants
    banks = [
        {
            'name': 'Chase Bank',
            'location': 'New York, NY',
            'customer_base': 5000000,
            'services': ['retail_banking', 'credit_cards', 'mortgages']
        },
        {
            'name': 'Bank of America',
            'location': 'Charlotte, NC',
            'customer_base': 6500000,
            'services': ['retail_banking', 'investment_banking', 'wealth_management']
        },
        {
            'name': 'Wells Fargo',
            'location': 'San Francisco, CA',
            'customer_base': 4500000,
            'services': ['retail_banking', 'commercial_banking', 'insurance']
        }
    ]

    for bank in banks:
        participant_id = financial_fed.add_bank_participant(bank)
        print(f"✅ Added bank: {bank['name']} ({participant_id})")

    # Configure financial data processing
    financial_fed.configure_financial_data_processing({
        'data_format': 'transaction_logs',
        'preprocessing_enabled': True,
        'privacy_filters_enabled': True,
        'regulatory_checks_enabled': True
    })

    # Set up financial compliance
    financial_fed.setup_financial_compliance()

    # Configure and start training
    training_config = {
        'epochs': 100,
        'batch_size': 1024,  # Larger batches for financial data
        'learning_rate': 0.05,  # Conservative learning rate
        'optimizer': 'adam',
        'loss_function': 'multi_class_logloss',
        'metrics': ['accuracy', 'precision', 'recall', 'f1_score', 'auc'],
        'early_stopping': {
            'patience': 15,
            'min_delta': 0.001,
            'restore_best_weights': True
        },
        'validation_split': 0.3,
        'class_weight_balancing': True,
        'cross_validation_folds': 3
    }

    # Start training
    training_session = financial_fed.start_financial_risk_training(training_config)
    print(f"🚀 Started training session: {training_session.id}")

    # Monitor training
    financial_fed.monitor_financial_training(training_session)

    # Wait for training to complete (in real scenario, this would be asynchronous)
    print("⏳ Training in progress... (simulated)")
    await asyncio.sleep(5)  # Simulate training time

    # Generate final report
    final_report = financial_fed.generate_financial_report(training_session)

    print("\n📊 Final Financial Risk Assessment Report")
    print("-" * 50)
    print(f"Model Accuracy: {final_report['model_performance']['accuracy']:.3f}")
    print(f"AUC-ROC Score: {final_report['model_performance']['auc_roc']:.3f}")
    print(f"Privacy Budget Used: {final_report['privacy_metrics']['epsilon_used']:.3f}")
    print(f"SOX Compliant: {final_report['compliance_status']['sox_compliant']}")
    print(f"GLBA Compliant: {final_report['compliance_status']['glba_compliant']}")
    print(f"SOC 2 Compliant: {final_report['compliance_status']['soc2_compliant']}")
    print(f"Participants: {final_report['training_metadata']['total_participants']}")

    print("\n✅ Financial risk assessment example completed successfully!")
    print("This demonstrates how FEDZK enables privacy-preserving collaborative")
    print("financial risk assessment across multiple banking institutions.")


if __name__ == "__main__":
    asyncio.run(main())
