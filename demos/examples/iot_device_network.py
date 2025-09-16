#!/usr/bin/env python3
"""
IoT Device Network with Real ZK Proofs
======================================

This example demonstrates federated learning on resource-constrained IoT devices
using FEDzk with real zero-knowledge proofs. Multiple IoT devices collaborate
to train an anomaly detection model while maintaining data privacy.

Real-World Scenario:
- 10 IoT devices training a predictive maintenance model
- Real ZK proofs ensure model updates are valid
- Resource-constrained environment (limited memory/CPU)
- Edge computing with intermittent connectivity
- Production-ready error handling for unreliable networks

Prerequisites:
- ZK toolchain installed
- Lightweight MPC server for edge devices
- Quantized models suitable for edge deployment
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import time
import logging
import random
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import threading

# FEDzk imports
from fedzk.client import Trainer
from fedzk.mpc.client import MPCClient
from fedzk.coordinator import SecureCoordinator
from fedzk.utils import GradientQuantizer

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TinyIoTModel(nn.Module):
    """Tiny neural network for resource-constrained IoT devices."""
    def __init__(self, input_features=8, hidden_size=16, num_classes=2):
        super(TinyIoTModel, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_features, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_classes)
        )

    def forward(self, x):
        return self.model(x)

class IoTFLClient:
    """IoT Device Federated Learning Client with Real ZK Proofs."""

    def __init__(self, device_id: str, mpc_server_url: str, coordinator_url: str):
        self.device_id = device_id

        # Resource-constrained model
        self.model = TinyIoTModel()
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.01, momentum=0.9)
        self.criterion = nn.CrossEntropyLoss()

        # FEDzk components optimized for edge devices
        self.trainer = Trainer(
            model_config={
                'architecture': 'tiny_iot_mlp',
                'layers': [8, 16, 16, 2],  # Sensor anomaly detection
                'activation': 'relu'
            },
            security_config={
                'zk_circuit': 'model_update_secure',
                'max_norm': 1000,  # Smaller for IoT constraints
                'min_nonzero': 2
            }
        )

        # Lightweight MPC client for edge devices
        self.mpc_client = MPCClient(
            server_url=mpc_server_url,
            api_key=f"iot_device_{device_id}_key",
            tls_verify=True,
            timeout=30,  # Shorter timeout for edge networks
            max_retries=2,
            retry_delay=5
        )

        # Low-precision quantizer for edge efficiency
        self.quantizer = GradientQuantizer(
            scale_factor=256,  # 8-bit quantization for edge devices
            precision_bits=8,
            adaptive_scaling=True
        )

        # Coordinator connection with retry logic
        self.coordinator = SecureCoordinator(coordinator_url)

        # Edge device state
        self.battery_level = random.uniform(20, 100)
        self.network_quality = random.uniform(0.3, 1.0)
        self.memory_usage = 0.0
        self.last_sync_time = 0

        logger.info(f"📱 IoT Device {device_id} initialized (Battery: {self.battery_level:.1f}%, Network: {self.network_quality:.2f})")

    def simulate_sensor_data(self, num_samples: int = 100) -> DataLoader:
        """Generate synthetic IoT sensor data."""
        # IoT sensor features: temperature, vibration, pressure, etc.
        features = torch.randn(num_samples, 8)

        # Add realistic sensor patterns
        features[:, 0] = torch.normal(25, 5, (num_samples,))  # Temperature
        features[:, 1] = torch.normal(0.1, 0.05, (num_samples,))  # Vibration
        features[:, 2] = torch.normal(1.0, 0.1, (num_samples,))  # Pressure

        # Anomaly labels: 0=Normal, 1=Anomaly
        anomaly_scores = features.abs().mean(dim=1)
        labels = (anomaly_scores > anomaly_scores.mean() + anomaly_scores.std()).long()

        dataset = TensorDataset(features, labels)
        dataloader = DataLoader(dataset, batch_size=16, shuffle=True)  # Small batches for edge

        logger.info(f"Generated IoT sensor data: {num_samples} samples")
        return dataloader

    def check_device_resources(self) -> bool:
        """Check if device has sufficient resources for FL operations."""
        # Update device state
        self.battery_level = max(5, self.battery_level - random.uniform(1, 5))
        self.network_quality = max(0.1, self.network_quality + random.uniform(-0.2, 0.1))

        # Resource checks
        battery_ok = self.battery_level > 15
        network_ok = self.network_quality > 0.3
        memory_ok = self.memory_usage < 80

        if not battery_ok:
            logger.warning(f"📱 Device {self.device_id}: Low battery ({self.battery_level:.1f}%) - skipping round")
        if not network_ok:
            logger.warning(f"📱 Device {self.device_id}: Poor network ({self.network_quality:.2f}) - may fail")
        if not memory_ok:
            logger.warning(f"📱 Device {self.device_id}: High memory usage ({self.memory_usage:.1f}%)")

        return battery_ok and network_ok and memory_ok

    def local_training_round(self, dataloader: DataLoader, epochs: int = 2) -> Optional[Dict[str, torch.Tensor]]:
        """Perform local training with resource constraints."""
        if not self.check_device_resources():
            logger.info(f"📱 Device {self.device_id}: Insufficient resources - skipping training")
            return None

        logger.info(f"📱 Device {self.device_id}: Starting edge training")

        # Track memory usage
        self.memory_usage = random.uniform(30, 60)

        self.model.train()
        initial_params = {name: param.clone() for name, param in self.model.named_parameters()}

        for epoch in range(epochs):
            epoch_loss = 0.0
            for batch_x, batch_y in dataloader:
                # Check battery during training
                if self.battery_level < 10:
                    logger.warning(f"📱 Device {self.device_id}: Battery critical - aborting training")
                    return None

                self.optimizer.zero_grad()
                outputs = self.model(batch_x)
                loss = self.criterion(outputs, batch_y)
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()

                # Simulate battery drain
                self.battery_level -= random.uniform(0.1, 0.5)

            logger.info(".2f")

        # Calculate parameter updates
        updates = {}
        for name, param in self.model.named_parameters():
            updates[name] = param - initial_params[name]

        # Reset memory usage
        self.memory_usage = random.uniform(10, 30)

        logger.info(f"📱 Device {self.device_id}: Edge training completed")
        return updates

    def generate_zk_proof(self, updates: Dict[str, torch.Tensor]) -> Tuple[Optional[Dict], Optional[Dict]]:
        """Generate ZK proof with edge device constraints."""
        if not self.check_device_resources():
            logger.info(f"📱 Device {self.device_id}: Insufficient resources for ZK proof generation")
            return None, None

        logger.info(f"📱 Device {self.device_id}: Generating lightweight ZK proof")

        try:
            # Quantize gradients for edge efficiency
            quantized_updates = self.quantizer.quantize(updates)

            # Lightweight proof payload for IoT
            proof_payload = {
                "gradients": quantized_updates,
                "secure": True,
                "maxNorm": 1000,
                "minNonZero": 2,
                "client_id": self.device_id,
                "timestamp": time.time(),
                "device_type": "iot_sensor",
                "resource_constraints": {
                    "battery_level": self.battery_level,
                    "network_quality": self.network_quality,
                    "memory_usage": self.memory_usage
                },
                "model_version": "tiny_iot_v1.0"
            }

            # Generate proof with timeout protection
            start_time = time.time()
            proof, public_signals = self.mpc_client.generate_proof(proof_payload)
            proof_time = time.time() - start_time

            # Update device state
            self.memory_usage += random.uniform(5, 15)
            self.battery_level -= random.uniform(2, 8)

            logger.info(".2f"
            return proof, public_signals

        except Exception as e:
            logger.error(f"📱 Device {self.device_id}: ZK proof generation failed: {e}")
            return None, None

    def submit_to_coordinator(self, quantized_updates: Dict, proof: Dict, public_signals: Dict) -> bool:
        """Submit updates with edge device reliability handling."""
        if not self.check_device_resources():
            logger.info(f"📱 Device {self.device_id}: Poor connectivity - queuing update")
            # In production, would queue for later submission
            return False

        logger.info(f"📱 Device {self.device_id}: Submitting to coordinator")

        try:
            # Simulate network delay
            time.sleep(random.uniform(0.1, 2.0))

            success = self.coordinator.submit_update(
                client_id=self.device_id,
                model_updates=quantized_updates,
                zk_proof=proof,
                public_signals=public_signals,
                metadata={
                    'device_type': 'iot_sensor_node',
                    'battery_level': self.battery_level,
                    'network_quality': self.network_quality,
                    'data_samples': 100,
                    'model_size': 'tiny',
                    'power_consumption': random.uniform(0.5, 2.0),
                    'uptime_hours': random.uniform(100, 1000)
                }
            )

            if success:
                self.last_sync_time = time.time()
                logger.info(f"📱 Device {self.device_id}: ✅ Update submitted successfully")
            else:
                logger.error(f"📱 Device {self.device_id}: ❌ Submission failed")

            return success

        except Exception as e:
            logger.error(f"📱 Device {self.device_id}: Coordinator submission failed: {e}")
            return False

    def run_training_round(self) -> bool:
        """Complete federated learning round with edge constraints."""
        try:
            logger.info(f"📱 IoT Device {self.device_id}: Starting FL round")

            # Step 1: Check device health
            if not self.check_device_resources():
                logger.info(f"📱 IoT Device {self.device_id}: Device not ready - skipping round")
                return False

            # Step 2: Generate sensor data
            dataloader = self.simulate_sensor_data(50)  # Smaller dataset for edge

            # Step 3: Local training
            updates = self.local_training_round(dataloader, epochs=1)
            if updates is None:
                return False

            # Step 4: Generate ZK proof
            proof, public_signals = self.generate_zk_proof(updates)
            if proof is None or public_signals is None:
                return False

            # Step 5: Submit to coordinator
            success = self.submit_to_coordinator(
                self.quantizer.quantize(updates),
                proof,
                public_signals
            )

            logger.info(f"📱 IoT Device {self.device_id}: FL round {'completed' if success else 'failed'}")
            return success

        except Exception as e:
            logger.error(f"📱 IoT Device {self.device_id}: FL round failed: {e}")
            return False

def simulate_network_conditions(devices: List[IoTFLClient]):
    """Simulate real-world network conditions for IoT devices."""
    def network_fluctuation():
        while True:
            time.sleep(random.uniform(5, 15))
            for device in devices:
                # Random network fluctuations
                device.network_quality = max(0.1, min(1.0,
                    device.network_quality + random.uniform(-0.3, 0.2)))

                # Battery drain
                device.battery_level = max(5, device.battery_level - random.uniform(0.5, 2.0))

    # Start network monitoring in background
    network_thread = threading.Thread(target=network_fluctuation, daemon=True)
    network_thread.start()
    return network_thread

def run_iot_federated_learning():
    """Run IoT federated learning with real ZK proofs on edge devices."""
    logger.info("📱 Starting IoT Device Network Federated Learning")
    logger.info("=" * 60)

    # Configuration for edge deployment
    MPC_SERVER_URL = "https://mpc.iot-network.local:9000"
    COORDINATOR_URL = "https://coordinator.iot-network.local:8443"

    # Initialize IoT devices
    devices = []
    device_ids = [f"IoT_{i:03d}" for i in range(1, 11)]  # 10 devices

    for device_id in device_ids:
        try:
            device = IoTFLClient(device_id, MPC_SERVER_URL, COORDINATOR_URL)
            devices.append(device)
            logger.info(f"✅ Initialized IoT device: {device_id}")
        except Exception as e:
            logger.error(f"❌ Failed to initialize {device_id}: {e}")

    # Monitor network conditions
    logger.info("🌐 Monitoring network conditions...")
    network_thread = simulate_network_conditions(devices)

    # Run federated learning rounds
    successful_rounds = 0
    total_rounds = 5  # More rounds to simulate edge conditions

    for round_num in range(1, total_rounds + 1):
        logger.info(f"\n🔄 Starting Round {round_num}/{total_rounds}")
        logger.info("-" * 40)

        round_success = 0
        round_participation = 0

        for device in devices:
            participation = device.run_training_round()
            round_participation += 1 if participation else 0
            if participation:
                round_success += 1

        successful_rounds += round_success
        success_rate = (round_success / round_participation) * 100 if round_participation > 0 else 0
        participation_rate = (round_participation / len(devices)) * 100

        logger.info(f"Round {round_num}: {round_success}/{round_participation} devices successful ({success_rate:.1f}%), {participation_rate:.1f}% participation")

        # Brief pause between rounds
        time.sleep(2)

    # Device health summary
    logger.info("\n📊 Device Health Summary:")
    for device in devices:
        status = "🟢 Good" if device.battery_level > 30 and device.network_quality > 0.5 else "🟡 Poor" if device.battery_level > 15 else "🔴 Critical"
        logger.info(f"📱 {device.device_id}: {status} - Battery: {device.battery_level:.1f}%, Network: {device.network_quality:.2f}")
    # Final results
    total_success_rate = (successful_rounds / (total_rounds * len(devices))) * 100
    logger.info(f"\n🏆 Final Results:")
    logger.info(".1f")
    logger.info(f"   Total successful proofs: {successful_rounds}")
    logger.info(f"   Total devices: {len(devices)}")
    logger.info(f"   Total rounds: {total_rounds}")
    logger.info(f"   Environment: Resource-constrained edge network")

    if total_success_rate >= 60:  # Lower bar for edge devices
        logger.info("✅ IoT FL: SUCCESS - Real ZK Proofs on Edge Devices")
        return True
    else:
        logger.warning("⚠️ IoT FL: EDGE CONSTRAINTS DETECTED - Check ZK Proof Generation")
        return False

def main():
    """Main entry point with proper error handling."""
    try:
        success = run_iot_federated_learning()
        exit(0 if success else 1)
    except KeyboardInterrupt:
        logger.info("🛑 IoT process interrupted by user")
        exit(1)
    except Exception as e:
        logger.error(f"💥 IoT process failed: {e}")
        exit(1)

if __name__ == "__main__":
    main()
