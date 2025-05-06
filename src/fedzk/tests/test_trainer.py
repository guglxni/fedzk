"""
Tests for the LocalTrainer class.
"""

from typing import Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from fedzk.client.trainer import LocalTrainer


class SimpleLinearClassifier(nn.Module):
    """A simple linear classifier for testing."""

    def __init__(self, input_dim: int, num_classes: int):
        """
        Initialize a simple linear classifier.
        
        Args:
            input_dim: Input feature dimension
            num_classes: Number of output classes
        """
        super().__init__()
        self.fc = nn.Linear(input_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.fc(x)


def create_dummy_data(
    num_samples: int = 10,
    input_dim: int = 5,
    num_classes: int = 3
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Create dummy data for testing.
    
    Args:
        num_samples: Number of samples to generate
        input_dim: Dimension of input features
        num_classes: Number of classes for classification
        
    Returns:
        Tuple of (inputs, targets) tensors
    """
    # Random inputs
    inputs = torch.randn(num_samples, input_dim)

    # Random class labels
    targets = torch.randint(0, num_classes, (num_samples,))

    return inputs, targets


def test_trainer_returns_gradients():
    """Test that LocalTrainer.train_one_epoch returns gradients with correct structure."""
    # Setup
    input_dim = 5
    num_classes = 3
    batch_size = 2

    # Create model and data
    model = SimpleLinearClassifier(input_dim, num_classes)
    inputs, targets = create_dummy_data(
        num_samples=10,
        input_dim=input_dim,
        num_classes=num_classes
    )
    dataset = TensorDataset(inputs, targets)
    dataloader = DataLoader(dataset, batch_size=batch_size)

    # Create trainer and train
    trainer = LocalTrainer(model, dataloader)
    gradients = trainer.train_one_epoch()

    # Assertions
    assert isinstance(gradients, dict), "Gradients should be returned as a dictionary"

    # Check expected parameter names
    expected_params = {"fc.weight", "fc.bias"}
    assert set(gradients.keys()) == expected_params, f"Expected parameters {expected_params}, got {set(gradients.keys())}"

    # Check gradient shapes
    assert gradients["fc.weight"].shape == (num_classes, input_dim), "Weight gradient has incorrect shape"
    assert gradients["fc.bias"].shape == (num_classes,), "Bias gradient has incorrect shape"

    # Check that gradients are not all zeros (training should have happened)
    assert torch.any(gradients["fc.weight"] != 0), "Weight gradients are all zeros"
    assert torch.any(gradients["fc.bias"] != 0), "Bias gradients are all zeros"


def test_trainer_with_custom_loss():
    """Test that LocalTrainer works with a custom loss function."""
    # Setup
    model = SimpleLinearClassifier(5, 3)
    inputs, targets = create_dummy_data(num_samples=10)
    dataset = TensorDataset(inputs, targets)
    dataloader = DataLoader(dataset, batch_size=2)

    # Custom loss function
    custom_loss = nn.NLLLoss()

    # Create trainer with custom loss
    trainer = LocalTrainer(model, dataloader, loss_fn=custom_loss)

    # Assert that the trainer uses the custom loss
    assert trainer.loss_fn == custom_loss, "Trainer should use the provided custom loss function"

    # Verify training still works
    gradients = trainer.train_one_epoch()
    assert isinstance(gradients, dict), "Training with custom loss should return gradients"
