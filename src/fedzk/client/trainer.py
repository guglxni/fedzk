# Copyright (c) 2025 Aaryan Guglani and FedZK Contributors
# SPDX-License-Identifier: MIT

"""
Trainer module for federated learning clients.

This module contains the LocalTrainer class which handles local model training
on a client's private data and returns the gradients for federated aggregation.
"""

from typing import Dict, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


class LocalTrainer:
    """
    Handles local training of a model on a client's private data.
    
    This class performs local training for one epoch and returns the gradients
    which can then be used in the federated learning protocol.
    """

    def __init__(
        self,
        model: nn.Module,
        dataloader: DataLoader,
        learning_rate: float = 0.01,
        loss_fn: Optional[nn.Module] = None
    ):
        """
        Initialize a LocalTrainer with a model and dataloader.
        
        Args:
            model: PyTorch neural network model to train
            dataloader: DataLoader providing batches of training data
            learning_rate: Learning rate for optimization (default: 0.01)
            loss_fn: Loss function to use (default: nn.CrossEntropyLoss())
        """
        self.model = model
        self.dataloader = dataloader
        self.learning_rate = learning_rate
        self.loss_fn = loss_fn if loss_fn is not None else nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate)

    def train_one_epoch(self) -> Dict[str, torch.Tensor]:
        """
        Trains the model for one epoch, returns gradients by parameter name.
        
        This method:
        1. Trains the model through one complete pass of the dataloader
        2. Accumulates and returns the gradients for each parameter
        
        Returns:
            Dictionary mapping parameter names to their gradient tensors
        """
        self.model.train()

        # Store initial parameter values
        initial_params = {name: param.clone().detach()
                        for name, param in self.model.named_parameters()}

        # Training loop
        for inputs, targets in self.dataloader:
            # Forward pass
            outputs = self.model(inputs)
            loss = self.loss_fn(outputs, targets)

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        # Compute parameter gradients (change from initial parameters)
        gradients = {}
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                # Calculate change: initial - current = negative gradient
                # Multiply by -1 to get actual gradient direction
                gradient = (initial_params[name] - param.clone().detach()) * -1.0
                gradients[name] = gradient

        return gradients
