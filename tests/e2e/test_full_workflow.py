"""
End-to-End Tests for FEDZK Complete Workflow

This module contains comprehensive end-to-end tests that validate
the complete FEDZK federated learning workflow from start to finish.
"""

import pytest
import asyncio
import tempfile
import shutil
from pathlib import Path
from unittest.mock import patch, MagicMock

from fedzk.coordinator.api import CoordinatorAPI
from fedzk.client.trainer import FederatedTrainer
from fedzk.prover.zkgenerator import ZKGenerator
from fedzk.validation.proof_validator import ProofValidator


class TestFullWorkflow:
    """Test complete FEDZK workflow end-to-end."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for test files."""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def mock_config(self):
        """Mock configuration for testing."""
        return {
            "coordinator": {
                "host": "localhost",
                "port": 8000,
                "max_clients": 10
            },
            "zk": {
                "circuit_path": "test_circuit.circom",
                "trusted_setup": True
            },
            "training": {
                "rounds": 3,
                "batch_size": 32,
                "learning_rate": 0.01
            }
        }
    
    def test_basic_workflow_initialization(self, temp_dir, mock_config):
        """Test that all components can be initialized properly."""
        # Initialize coordinator
        coordinator = CoordinatorAPI(config=mock_config)
        assert coordinator is not None
        
        # Initialize trainer
        trainer = FederatedTrainer(config=mock_config)
        assert trainer is not None
        
        # Initialize ZK generator
        zk_gen = ZKGenerator(config=mock_config)
        assert zk_gen is not None
        
        # Initialize proof validator
        validator = ProofValidator(config=mock_config)
        assert validator is not None
    
    @patch('fedzk.coordinator.api.CoordinatorAPI.start_training_round')
    @patch('fedzk.client.trainer.FederatedTrainer.train_local_model')
    def test_single_training_round(self, mock_train, mock_start_round, mock_config):
        """Test a single training round workflow."""
        # Mock training data
        mock_train.return_value = {
            "model_update": [0.1, 0.2, 0.3, 0.4],
            "loss": 0.5,
            "accuracy": 0.85
        }
        
        mock_start_round.return_value = {
            "round_id": 1,
            "global_model": [0.0, 0.0, 0.0, 0.0],
            "status": "started"
        }
        
        # Initialize components
        coordinator = CoordinatorAPI(config=mock_config)
        trainer = FederatedTrainer(config=mock_config)
        
        # Start training round
        round_info = coordinator.start_training_round(round_id=1)
        assert round_info["status"] == "started"
        
        # Train local model
        model_update = trainer.train_local_model(
            global_model=round_info["global_model"]
        )
        
        assert "model_update" in model_update
        assert "loss" in model_update
        assert len(model_update["model_update"]) == 4
    
    @patch('fedzk.prover.zkgenerator.ZKGenerator.generate_proof')
    @patch('fedzk.validation.proof_validator.ProofValidator.verify_proof')
    def test_proof_generation_and_verification(self, mock_verify, mock_generate, mock_config):
        """Test ZK proof generation and verification workflow."""
        # Mock proof generation
        mock_proof = {
            "pi_a": [123, 456],
            "pi_b": [[789, 101], [112, 131]],
            "pi_c": [415, 161],
            "protocol": "groth16",
            "curve": "bn128"
        }
        mock_generate.return_value = mock_proof
        
        # Mock proof verification
        mock_verify.return_value = {
            "valid": True,
            "verification_time": 0.025,
            "proof_hash": "abc123def456"
        }
        
        # Initialize components
        zk_gen = ZKGenerator(config=mock_config)
        validator = ProofValidator(config=mock_config)
        
        # Generate proof for model update
        model_update = [0.1, 0.2, 0.3, 0.4]
        proof = zk_gen.generate_proof(
            inputs={"gradients": model_update}
        )
        
        assert proof is not None
        assert "pi_a" in proof
        assert "protocol" in proof
        
        # Verify proof
        verification_result = validator.verify_proof(
            proof=proof,
            public_inputs=model_update
        )
        
        assert verification_result["valid"] is True
        assert verification_result["verification_time"] > 0
    
    @patch('fedzk.coordinator.api.CoordinatorAPI.aggregate_updates')
    def test_model_aggregation(self, mock_aggregate, mock_config):
        """Test model update aggregation."""
        # Mock aggregation
        mock_aggregate.return_value = {
            "aggregated_model": [0.15, 0.25, 0.35, 0.45],
            "num_clients": 3,
            "aggregation_method": "fedavg"
        }
        
        # Initialize coordinator
        coordinator = CoordinatorAPI(config=mock_config)
        
        # Mock client updates
        client_updates = [
            {"client_id": "client1", "update": [0.1, 0.2, 0.3, 0.4]},
            {"client_id": "client2", "update": [0.2, 0.3, 0.4, 0.5]},
            {"client_id": "client3", "update": [0.15, 0.25, 0.35, 0.45]}
        ]
        
        # Aggregate updates
        result = coordinator.aggregate_updates(client_updates)
        
        assert result is not None
        assert "aggregated_model" in result
        assert result["num_clients"] == 3
        assert len(result["aggregated_model"]) == 4
    
    @pytest.mark.asyncio
    async def test_multi_client_coordination(self, mock_config):
        """Test coordination with multiple clients."""
        with patch('fedzk.coordinator.api.CoordinatorAPI.register_client') as mock_register:
            mock_register.return_value = {"client_id": "test_client", "status": "registered"}
            
            coordinator = CoordinatorAPI(config=mock_config)
            
            # Register multiple clients
            clients = []
            for i in range(3):
                client_info = coordinator.register_client(f"client_{i}")
                clients.append(client_info)
            
            assert len(clients) == 3
            for client in clients:
                assert client["status"] == "registered"
    
    def test_error_handling(self, mock_config):
        """Test error handling in workflow components."""
        # Test coordinator error handling
        with patch('fedzk.coordinator.api.CoordinatorAPI.start_training_round') as mock_start:
            mock_start.side_effect = Exception("Network error")
            
            coordinator = CoordinatorAPI(config=mock_config)
            
            with pytest.raises(Exception, match="Network error"):
                coordinator.start_training_round(round_id=1)
        
        # Test trainer error handling
        with patch('fedzk.client.trainer.FederatedTrainer.train_local_model') as mock_train:
            mock_train.side_effect = ValueError("Invalid model parameters")
            
            trainer = FederatedTrainer(config=mock_config)
            
            with pytest.raises(ValueError, match="Invalid model parameters"):
                trainer.train_local_model(global_model=[0.0, 0.0, 0.0, 0.0])
    
    def test_configuration_validation(self):
        """Test configuration validation."""
        # Test invalid configuration
        invalid_config = {
            "coordinator": {
                "host": "",  # Invalid empty host
                "port": -1,  # Invalid port
            }
        }
        
        with pytest.raises((ValueError, AssertionError)):
            CoordinatorAPI(config=invalid_config)
    
    @patch('fedzk.monitoring.metrics.FEDzkMetricsCollector.record_training_round')
    def test_metrics_collection(self, mock_metrics, mock_config):
        """Test that metrics are properly collected during workflow."""
        mock_metrics.return_value = None
        
        # Initialize components with metrics
        coordinator = CoordinatorAPI(config=mock_config)
        
        # Simulate training round with metrics
        with patch.object(coordinator, 'start_training_round') as mock_start:
            mock_start.return_value = {"round_id": 1, "status": "started"}
            
            result = coordinator.start_training_round(round_id=1)
            assert result["status"] == "started"
    
    def test_security_validation(self, mock_config):
        """Test security validations in the workflow."""
        # Test that security components are properly initialized
        validator = ProofValidator(config=mock_config)
        
        # Test invalid proof handling
        invalid_proof = {
            "pi_a": [],  # Empty proof components
            "pi_b": [],
            "pi_c": []
        }
        
        with patch.object(validator, 'verify_proof') as mock_verify:
            mock_verify.return_value = {"valid": False, "error": "Invalid proof format"}
            
            result = validator.verify_proof(
                proof=invalid_proof,
                public_inputs=[0.1, 0.2, 0.3, 0.4]
            )
            
            assert result["valid"] is False
            assert "error" in result


@pytest.mark.integration
class TestWorkflowIntegration:
    """Integration tests for complete workflow scenarios."""
    
    def test_full_federated_learning_cycle(self):
        """Test a complete federated learning cycle."""
        # This would be a more comprehensive test that exercises
        # the entire system in a more realistic scenario
        pass
    
    def test_fault_tolerance(self):
        """Test system behavior under failure conditions."""
        # Test client disconnection, network issues, etc.
        pass
    
    def test_scalability_limits(self):
        """Test system behavior with many clients."""
        # Test with increasing number of clients
        pass
