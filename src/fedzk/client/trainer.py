# Functional Source License 1.1 with Apache-2.0 Future Grant (FSL-1.1-Apache-2.0)
# Copyright (c) 2025 Aaryan Guglani and FEDzk Contributors
# Licensed under FSL-1.1-Apache-2.0. See LICENSE for details.

"""
Production-grade trainer module for federated learning clients.

This module contains the LocalTrainer class which handles local model training
on a client's private data with support for multiple model types, optimizers,
and training strategies.
"""

import json
import logging
import os
from typing import Dict, Optional, Tuple, Any, List
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, Dataset
import torch.nn.functional as F

# Setup logger
logger = logging.getLogger(__name__)

class SimpleLinearModel(nn.Module):
    """Simple linear model for basic federated learning."""
    
    def __init__(self, input_size: int, hidden_size: int = 64, output_size: int = 10):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

class SimpleCNN(nn.Module):
    """Simple CNN for image classification."""
    
    def __init__(self, input_channels: int = 1, num_classes: int = 10):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, num_classes)
    
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.dropout2(x)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

class SimpleTransformer(nn.Module):
    """Simple transformer model for sequence tasks."""
    
    def __init__(self, vocab_size: int = 10000, d_model: int = 128, nhead: int = 8, 
                 num_layers: int = 2, num_classes: int = 10):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = nn.Parameter(torch.randn(1000, d_model))
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        self.classifier = nn.Linear(d_model, num_classes)
        self.d_model = d_model
    
    def forward(self, x):
        seq_len = x.size(1)
        x = self.embedding(x) * torch.sqrt(torch.tensor(self.d_model, dtype=torch.float32))
        x += self.pos_encoding[:seq_len, :]
        x = self.transformer(x)
        x = x.mean(dim=1)  # Global average pooling
        return self.classifier(x)

class LocalTrainer:
    """
    Production-grade local trainer for federated learning clients.
    
    Supports multiple model types, optimizers, and training strategies.
    """

    def __init__(
        self,
        model_or_model_type=None,
        dataloader_or_learning_rate=None,
        learning_rate: float = 0.01,
        optimizer_type: str = "adam", 
        secure: bool = False,
        device: Optional[str] = None,
        **model_kwargs
    ):
        """
        Initialize the LocalTrainer.
        
        Supports both new and legacy constructor signatures:
        - New: LocalTrainer(model_type="linear", learning_rate=0.01, ...)
        - Legacy: LocalTrainer(model, dataloader)
        
        Args:
            model_or_model_type: Model instance (legacy) or model type string (new)
            dataloader_or_learning_rate: DataLoader (legacy) or learning rate (new)
            learning_rate: Learning rate for optimization (new API)
            optimizer_type: Type of optimizer ('adam', 'sgd', 'rmsprop')
            secure: Whether to use secure training protocols
            device: Device to train on ('cpu', 'cuda', 'auto')
            **model_kwargs: Additional arguments for model initialization
        """
        
        # Detect legacy vs new API usage
        if (model_or_model_type is not None and 
            hasattr(model_or_model_type, 'parameters') and 
            dataloader_or_learning_rate is not None and
            hasattr(dataloader_or_learning_rate, '__len__') and
            hasattr(dataloader_or_learning_rate, '__iter__')):
            # Legacy API: LocalTrainer(model, dataloader)
            self._init_legacy(model_or_model_type, dataloader_or_learning_rate, 
                            learning_rate, optimizer_type, secure, device, **model_kwargs)
        else:
            # New API: LocalTrainer(model_type="linear", ...)
            model_type = model_or_model_type or "linear"
            if dataloader_or_learning_rate is not None and isinstance(dataloader_or_learning_rate, (int, float)):
                learning_rate = dataloader_or_learning_rate
            self._init_new(model_type, learning_rate, optimizer_type, secure, device, **model_kwargs)
    
    def _init_legacy(self, model, dataloader, learning_rate, optimizer_type, secure, device, **model_kwargs):
        """Initialize with legacy API."""
        self.model_type = "legacy"
        self.learning_rate = learning_rate
        self.optimizer_type = optimizer_type
        self.secure = secure
        self.model_kwargs = model_kwargs
        
        # Set device
        if device == "auto" or device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        # Use provided model and dataloader
        self.model = model.to(self.device)
        self.dataloader = dataloader
        
        # Handle custom loss function
        if "loss_fn" in model_kwargs:
            self.criterion = model_kwargs["loss_fn"]
        else:
            self.criterion = nn.CrossEntropyLoss()
            
        self.optimizer = self.create_optimizer(self.model)
        self.input_size = None  # Not needed for legacy API
        
        logger.info(f"LocalTrainer initialized (legacy mode): device={self.device}")
    
    def _init_new(self, model_type, learning_rate, optimizer_type, secure, device, **model_kwargs):
        """Initialize with new API."""
        self.model_type = model_type
        self.learning_rate = learning_rate
        self.optimizer_type = optimizer_type
        self.secure = secure
        self.model_kwargs = model_kwargs
        
        # Set device
        if device == "auto" or device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        logger.info(f"LocalTrainer initialized: model={model_type}, lr={learning_rate}, device={self.device}")
        
        # Initialize model and optimizer (will be set when data is loaded)
        self.model = None
        self.optimizer = None
        self.criterion = None
        self.dataloader = None
        self.input_size = None
        
    def create_model(self, input_size: int, num_classes: int = 10) -> nn.Module:
        """Create model based on type."""
        if self.model_type == "linear":
            hidden_size = self.model_kwargs.get("hidden_size", 64)
            return SimpleLinearModel(input_size, hidden_size, num_classes)
        elif self.model_type == "cnn":
            input_channels = self.model_kwargs.get("input_channels", 1)
            return SimpleCNN(input_channels, num_classes)
        elif self.model_type == "transformer":
            vocab_size = self.model_kwargs.get("vocab_size", 10000)
            d_model = self.model_kwargs.get("d_model", 128)
            nhead = self.model_kwargs.get("nhead", 8)
            num_layers = self.model_kwargs.get("num_layers", 2)
            return SimpleTransformer(vocab_size, d_model, nhead, num_layers, num_classes)
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
    
    def create_optimizer(self, model: nn.Module) -> optim.Optimizer:
        """Create optimizer based on type."""
        if self.optimizer_type == "adam":
            return optim.Adam(model.parameters(), lr=self.learning_rate)
        elif self.optimizer_type == "sgd":
            momentum = self.model_kwargs.get("momentum", 0.9)
            return optim.SGD(model.parameters(), lr=self.learning_rate, momentum=momentum)
        elif self.optimizer_type == "rmsprop":
            return optim.RMSprop(model.parameters(), lr=self.learning_rate)
        else:
            raise ValueError(f"Unsupported optimizer type: {self.optimizer_type}")
    
    def load_data(self, data_path: str):
        """
        Load training data from various formats.
        
        Supports .pt (PyTorch), .json (custom format), and synthetic data generation.
        """
        data_path = Path(data_path)
        
        if not data_path.exists():
            if data_path.name == "synthetic":
                logger.info("Generating synthetic data for testing")
                self._generate_synthetic_data()
                return
            else:
                raise FileNotFoundError(f"Data path not found: {data_path}")
        
        if data_path.suffix == ".pt":
            self._load_pytorch_data(data_path)
        elif data_path.suffix == ".json":
            self._load_json_data(data_path)
        else:
            raise ValueError(f"Unsupported data format: {data_path.suffix}")
    
    def _load_pytorch_data(self, data_path: Path):
        """Load PyTorch tensor data."""
        data = torch.load(data_path, map_location="cpu")
        
        if isinstance(data, dict):
            X = data["X"]
            y = data["y"]
        elif isinstance(data, (list, tuple)) and len(data) == 2:
            X, y = data
        else:
            raise ValueError("PyTorch data must be dict with 'X', 'y' keys or tuple (X, y)")
        
        # Infer dimensions
        if self.model_type == "linear":
            self.input_size = X.view(X.size(0), -1).size(1)
        elif self.model_type == "cnn":
            self.input_size = X.size(1) * X.size(2) * X.size(3) if len(X.shape) > 3 else X.size(1)
        elif self.model_type == "transformer":
            self.input_size = X.size(1)  # Sequence length
        
        num_classes = int(y.max().item()) + 1
        
        # Create dataset and dataloader
        dataset = TensorDataset(X, y)
        self.dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
        
        # Initialize model
        self.model = self.create_model(self.input_size, num_classes).to(self.device)
        self.optimizer = self.create_optimizer(self.model)
        self.criterion = nn.CrossEntropyLoss()
        
        logger.info(f"Loaded PyTorch data: {len(dataset)} samples, {num_classes} classes")
    
    def _load_json_data(self, data_path: Path):
        """Load data from JSON format."""
        with open(data_path, 'r') as f:
            data = json.load(f)
        
        X = torch.tensor(data["X"], dtype=torch.float32)
        y = torch.tensor(data["y"], dtype=torch.long)
        
        # Convert to appropriate format for model type
        if self.model_type == "linear":
            X = X.view(X.size(0), -1)
            self.input_size = X.size(1)
        elif self.model_type == "cnn":
            if len(X.shape) == 2:  # Flatten data, reshape for CNN
                side_len = int(X.size(1) ** 0.5)
                X = X.view(-1, 1, side_len, side_len)
            self.input_size = X.size(1) * X.size(2) * X.size(3)
        
        num_classes = int(y.max().item()) + 1
        
        # Create dataset and dataloader
        dataset = TensorDataset(X, y)
        self.dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
        
        # Initialize model
        self.model = self.create_model(self.input_size, num_classes).to(self.device)
        self.optimizer = self.create_optimizer(self.model)
        self.criterion = nn.CrossEntropyLoss()
        
        logger.info(f"Loaded JSON data: {len(dataset)} samples, {num_classes} classes")
    
    def _generate_synthetic_data(self):
        """Generate synthetic data for testing."""
        logger.info("Generating synthetic data...")
        
        if self.model_type == "linear":
            self.input_size = 784  # 28x28 like MNIST
            X = torch.randn(1000, self.input_size)
            y = torch.randint(0, 10, (1000,))
        elif self.model_type == "cnn":
            X = torch.randn(1000, 1, 28, 28)  # MNIST-like
            y = torch.randint(0, 10, (1000,))
            self.input_size = 784
        elif self.model_type == "transformer":
            seq_len = 50
            vocab_size = 1000
            X = torch.randint(0, vocab_size, (500, seq_len))
            y = torch.randint(0, 10, (500,))
            self.input_size = seq_len
        
        num_classes = 10
        
        # Create dataset and dataloader
        dataset = TensorDataset(X, y)
        self.dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
        
        # Initialize model
        self.model = self.create_model(self.input_size, num_classes).to(self.device)
        self.optimizer = self.create_optimizer(self.model)
        self.criterion = nn.CrossEntropyLoss()
        
        logger.info(f"Generated synthetic data: {len(dataset)} samples, {num_classes} classes")
    
    def train(self, epochs: int = 1) -> Dict[str, Any]:
        """
        Train the model for specified number of epochs.
        
        Returns:
            Dictionary containing training metrics and gradients
        """
        if self.model is None:
            raise RuntimeError("Model not initialized. Call load_data() first.")
        
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            epoch_correct = 0
            epoch_total = 0
            
            for batch_idx, (data, target) in enumerate(self.dataloader):
                data, target = data.to(self.device), target.to(self.device)
                
                self.optimizer.zero_grad()
                output = self.model(data)
                loss = self.criterion(output, target)
                loss.backward()
                self.optimizer.step()
                
                # Statistics
                epoch_loss += loss.item()
                pred = output.argmax(dim=1, keepdim=True)
                epoch_correct += pred.eq(target.view_as(pred)).sum().item()
                epoch_total += target.size(0)
                
                if batch_idx % 10 == 0:
                    logger.debug(f"Epoch {epoch+1}/{epochs}, Batch {batch_idx}, Loss: {loss.item():.6f}")
            
            total_loss += epoch_loss
            correct += epoch_correct
            total += epoch_total
            
            epoch_accuracy = epoch_correct / epoch_total
            logger.info(f"Epoch {epoch+1}/{epochs}: Loss={epoch_loss/len(self.dataloader):.6f}, Accuracy={epoch_accuracy:.4f}")
        
        # Calculate final metrics
        avg_loss = total_loss / (len(self.dataloader) * epochs)
        accuracy = correct / total
        
        # Extract gradients
        gradients = self.get_gradients()
        
        metrics = {
            "loss": avg_loss,
            "accuracy": accuracy,
            "epochs": epochs,
            "total_samples": total,
            "model_type": self.model_type,
            "gradients": gradients
        }
        
        logger.info(f"Training completed: Loss={avg_loss:.6f}, Accuracy={accuracy:.4f}")
        return metrics
    
    def get_gradients(self) -> Dict[str, torch.Tensor]:
        """
        Extract gradients from the model.
        
        Returns:
            Dictionary mapping parameter names to gradient tensors
        """
        if self.model is None:
            raise RuntimeError("Model not initialized.")
        
        gradients = {}
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                gradients[name] = param.grad.clone().detach()
            else:
                # If no gradient (e.g., frozen layers), use zeros
                gradients[name] = torch.zeros_like(param)
        
        return gradients
    
    def save_model(self, path: str):
        """Save the trained model."""
        if self.model is None:
            raise RuntimeError("Model not initialized.")
        
        save_dict = {
            "model_state_dict": self.model.state_dict(),
            "model_type": self.model_type,
            "input_size": self.input_size,
            "optimizer_state_dict": self.optimizer.state_dict(),
            "model_kwargs": self.model_kwargs
        }
        
        torch.save(save_dict, path)
        logger.info(f"Model saved to {path}")
    
    def load_model(self, path: str):
        """Load a previously saved model."""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model_type = checkpoint["model_type"]
        self.input_size = checkpoint["input_size"]
        self.model_kwargs = checkpoint.get("model_kwargs", {})
        
        # Recreate model
        num_classes = 10  # Default, should ideally be saved too
        self.model = self.create_model(self.input_size, num_classes).to(self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        
        # Recreate optimizer
        self.optimizer = self.create_optimizer(self.model)
        if "optimizer_state_dict" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        
        self.criterion = nn.CrossEntropyLoss()
        
        logger.info(f"Model loaded from {path}")

    # Backward compatibility methods for tests
    @property
    def loss_fn(self):
        """Backward compatibility: return the criterion as loss_fn."""
        return self.criterion
    
    @loss_fn.setter
    def loss_fn(self, value):
        """Backward compatibility: allow setting criterion via loss_fn."""
        self.criterion = value
    
    def train_one_epoch(self) -> Dict[str, torch.Tensor]:
        """
        Backward compatibility: train for one epoch and return gradients.
        
        Returns:
            Dictionary of parameter gradients
        """
        if self.model is None or self.dataloader is None:
            raise RuntimeError("Model and data must be loaded before training")
        
        self.model.train()
        total_loss = 0.0
        
        for batch_idx, (data, target) in enumerate(self.dataloader):
            data, target = data.to(self.device), target.to(self.device)
            
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()
            
            total_loss += loss.item()
            self.optimizer.step()
        
        # Extract gradients
        gradients = {}
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                gradients[name] = param.grad.clone()
        
        avg_loss = total_loss / len(self.dataloader)
        logger.info(f"Epoch completed - Average loss: {avg_loss:.4f}")
        
        return gradients
    
    def train_step(self, inputs, targets) -> float:
        """
        Perform a single training step on the given batch of data.
        
        Args:
            inputs: Input data batch (tensors)
            targets: Target data batch (tensors)
            
        Returns:
            float: The loss value from this training step
        """
        self.model.train()
        
        # Move data to the correct device
        inputs = inputs.to(self.device)
        targets = targets.to(self.device)
        
        # Forward pass
        self.optimizer.zero_grad()
        outputs = self.model(inputs)
        
        # Handle different output and target shapes
        if isinstance(self.criterion, nn.MSELoss) and outputs.shape != targets.shape:
            if len(targets.shape) == 1:
                targets = targets.reshape(-1, 1)
        
        # Compute loss and backpropagate
        loss = self.criterion(outputs, targets)
        loss.backward()
        
        # Update weights
        self.optimizer.step()
        
        return float(loss.item())
