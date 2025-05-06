"""
Tests for the ZKVerifier class and the aggregation logic in the FedZKAggregator.
"""

from unittest.mock import patch

import pytest

from fedzk.coordinator.aggregator import (
    UpdateSubmission,
    get_status,
    pending_updates,
    submit_update,
)
from fedzk.prover.verifier import ZKVerifier


@pytest.fixture
def reset_state():
    """Reset global state between tests."""
    global current_version, pending_updates
    current_version = 1
    pending_updates.clear()
    yield
    # Clean up after each test
    current_version = 1
    pending_updates.clear()


@pytest.fixture
def mock_gradients1():
    """Mock gradients for first client."""
    return {
        "fc1.weight": [0.1, 0.2, 0.3, 0.4],
        "fc1.bias": [0.01, 0.02]
    }


@pytest.fixture
def mock_gradients2():
    """Mock gradients for second client."""
    return {
        "fc1.weight": [0.2, 0.3, 0.4, 0.5],
        "fc1.bias": [0.02, 0.03]
    }


@pytest.fixture
def mock_public_signals():
    """Mock public signals for ZK proof."""
    return [
        {"param_name": "fc1.weight", "norm": 0.5, "hash_prefix": "abcd1234"},
        {"param_name": "fc1.bias", "norm": 0.1, "hash_prefix": "5678efgh"}
    ]


def test_verifier_valid_proof():
    """Test that ZKVerifier correctly validates a valid proof."""
    verifier = ZKVerifier("dummy_vk.json")
    valid_proof = "dummy_proof_123456"
    public_signals = [{"param_name": "fc1.weight", "norm": 0.5}]

    assert verifier.verify_proof(valid_proof, public_signals) is True


def test_verifier_invalid_proof():
    """Test that ZKVerifier correctly rejects an invalid proof."""
    verifier = ZKVerifier("dummy_vk.json")
    invalid_proof = "invalid_proof_123456"  # Doesn't start with dummy_proof_
    public_signals = [{"param_name": "fc1.weight", "norm": 0.5}]

    assert verifier.verify_proof(invalid_proof, public_signals) is False

    # Also test with empty public signals
    valid_proof_format = "dummy_proof_123456"
    empty_signals = []
    assert verifier.verify_proof(valid_proof_format, empty_signals) is False


def test_status_endpoint(reset_state):
    """Test that the status endpoint returns correct initial state."""
    status = get_status()

    assert status["current_model_version"] == 1
    assert status["pending_updates"] == 0


def test_submit_update_single(reset_state, mock_gradients1, mock_public_signals):
    """Test submitting a single update."""
    # Create update submission
    update = UpdateSubmission(
        gradients=mock_gradients1,
        proof="dummy_proof_123456",
        public_signals=mock_public_signals
    )

    # Patch the verifier to return True
    with patch("fedzk.prover.verifier.ZKVerifier.verify_proof", return_value=True):
        # Submit update
        result = submit_update(update)

        assert result["status"] == "accepted"
        assert result["version"] == 1

        # Check aggregator state
        status = get_status()
        assert status["pending_updates"] == 1
        assert status["current_model_version"] == 1


def test_submit_update_with_aggregation(reset_state, mock_gradients1, mock_gradients2, mock_public_signals):
    """Test submitting two updates, which should trigger aggregation."""
    # Patch the verifier to return True
    with patch("fedzk.prover.verifier.ZKVerifier.verify_proof", return_value=True):
        # Submit first update
        update1 = UpdateSubmission(
            gradients=mock_gradients1,
            proof="dummy_proof_123456",
            public_signals=mock_public_signals
        )
        result1 = submit_update(update1)
        assert result1["status"] == "accepted"

        # Submit second update (should trigger aggregation)
        update2 = UpdateSubmission(
            gradients=mock_gradients2,
            proof="dummy_proof_789012",
            public_signals=mock_public_signals
        )
        result2 = submit_update(update2)

        # Check aggregation result
        assert result2["status"] == "aggregated"
        assert result2["version"] == 2  # Version should be incremented
        assert "global_update" in result2

        # Verify that the aggregation logic is correct (FedAvg)
        global_update = result2["global_update"]
        for param_name, values in global_update.items():
            for i, value in enumerate(values):
                expected = (mock_gradients1[param_name][i] + mock_gradients2[param_name][i]) / 2
                assert abs(value - expected) < 1e-6

        # Check aggregator state
        status = get_status()
        assert status["pending_updates"] == 0  # Should be reset after aggregation
        assert status["current_model_version"] == 2  # Should be incremented
