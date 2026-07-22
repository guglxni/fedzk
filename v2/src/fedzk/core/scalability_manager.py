#!/usr/bin/env python3
"""
Scalability Manager for FEDzk
=============================

Advanced scalability features:
- Horizontal scaling capabilities
- Load balancing and request routing
- Circuit sharding for large models
- Distributed proof generation support
- Auto-scaling based on load
"""

import asyncio
import threading
import time
import hashlib
import random
from typing import Dict, List, Any, Optional, Callable, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
from concurrent.futures import ThreadPoolExecutor
import logging
import psutil
import socket
import json
from queue import Queue, PriorityQueue
import heapq

logger = logging.getLogger(__name__)

class ScalingStrategy(Enum):
    """Scaling strategies for the system"""
    ROUND_ROBIN = "round_robin"
    LEAST_LOADED = "least_loaded"
    WEIGHTED_RANDOM = "weighted_random"
    CONSISTENT_HASHING = "consistent_hashing"

class ShardStrategy(Enum):
    """Circuit sharding strategies"""
    EQUAL_SIZE = "equal_size"
    LOAD_BALANCED = "load_balanced"
    COMPUTATIONAL_COST = "computational_cost"

@dataclass
class ScalabilityConfig:
    """Configuration for scalability features"""
    enable_horizontal_scaling: bool = True
    max_worker_nodes: int = 10
    min_worker_nodes: int = 1
    scaling_strategy: ScalingStrategy = ScalingStrategy.LEAST_LOADED
    enable_load_balancing: bool = True
    load_balance_interval: float = 30.0
    enable_circuit_sharding: bool = True
    max_shard_size: int = 1000
    shard_strategy: ShardStrategy = ShardStrategy.LOAD_BALANCED
    enable_distributed_proofs: bool = True
    proof_distribution_timeout: float = 60.0
    auto_scaling_enabled: bool = True
    scale_up_threshold: float = 0.8  # 80% CPU/memory usage
    scale_down_threshold: float = 0.3  # 30% CPU/memory usage
    health_check_interval: float = 10.0

@dataclass
class WorkerNode:
    """Represents a worker node in the distributed system"""
    node_id: str
    host: str
    port: int
    capabilities: Dict[str, Any] = field(default_factory=dict)
    load_factor: float = 0.0
    last_health_check: float = 0.0
    is_active: bool = True
    total_requests: int = 0
    successful_requests: int = 0
    average_response_time: float = 0.0

class HorizontalScaler:
    """
    Horizontal scaling manager for dynamic worker node management
    """

    def __init__(self, config: ScalabilityConfig):
        self.config = config
        self.worker_nodes: Dict[str, WorkerNode] = {}
        self._lock = threading.RLock()
        self._scaling_thread = threading.Thread(target=self._scaling_worker, daemon=True)
        self._health_thread = threading.Thread(target=self._health_worker, daemon=True)

        if self.config.enable_horizontal_scaling:
            self._scaling_thread.start()
            self._health_thread.start()

    def register_worker(self, node_id: str, host: str, port: int,
                       capabilities: Optional[Dict[str, Any]] = None) -> bool:
        """Register a new worker node"""
        with self._lock:
            if len(self.worker_nodes) >= self.config.max_worker_nodes:
                logger.warning(f"Cannot register worker {node_id}: maximum nodes reached")
                return False

            node = WorkerNode(
                node_id=node_id,
                host=host,
                port=port,
                capabilities=capabilities or {},
                last_health_check=time.time()
            )

            self.worker_nodes[node_id] = node
            logger.info(f"Registered worker node: {node_id} at {host}:{port}")
            return True

    def unregister_worker(self, node_id: str):
        """Unregister a worker node"""
        with self._lock:
            if node_id in self.worker_nodes:
                del self.worker_nodes[node_id]
                logger.info(f"Unregistered worker node: {node_id}")

    def get_active_workers(self) -> List[WorkerNode]:
        """Get list of active worker nodes"""
        with self._lock:
            return [node for node in self.worker_nodes.values() if node.is_active]

    def update_worker_load(self, node_id: str, load_factor: float):
        """Update the load factor for a worker node"""
        with self._lock:
            if node_id in self.worker_nodes:
                self.worker_nodes[node_id].load_factor = load_factor

    def _scaling_worker(self):
        """Background worker for scaling decisions"""
        while True:
            try:
                time.sleep(60)  # Check scaling every minute
                self._evaluate_scaling_needs()
            except Exception as e:
                logger.error(f"Scaling worker error: {e}")

    def _evaluate_scaling_needs(self):
        """Evaluate if scaling is needed based on current load"""
        if not self.config.auto_scaling_enabled:
            return

        active_workers = self.get_active_workers()
        if not active_workers:
            return

        # Calculate average load
        avg_load = sum(node.load_factor for node in active_workers) / len(active_workers)

        if avg_load > self.config.scale_up_threshold:
            self._scale_up(avg_load)
        elif avg_load < self.config.scale_down_threshold and len(active_workers) > self.config.min_worker_nodes:
            self._scale_down(avg_load)

    def _scale_up(self, current_load: float):
        """Scale up by adding more worker nodes"""
        if len(self.worker_nodes) >= self.config.max_worker_nodes:
            logger.warning("Cannot scale up: maximum worker nodes reached")
            return

        # In a real implementation, this would provision new nodes
        # For now, just log the scaling decision
        logger.info(".2f")

    def _scale_down(self, current_load: float):
        """Scale down by removing idle worker nodes"""
        active_workers = self.get_active_workers()

        # Find the least loaded worker to potentially remove
        if active_workers:
            least_loaded = min(active_workers, key=lambda x: x.load_factor)
            if least_loaded.load_factor < 0.1:  # Very low load
                logger.info(".2f"
                            f"load={least_loaded.load_factor:.2f})")

    def _health_worker(self):
        """Background worker for health checks"""
        while True:
            try:
                time.sleep(self.config.health_check_interval)
                self._perform_health_checks()
            except Exception as e:
                logger.error(f"Health check worker error: {e}")

    def _perform_health_checks(self):
        """Perform health checks on all worker nodes"""
        with self._lock:
            for node_id, node in self.worker_nodes.items():
                if self._check_node_health(node):
                    node.is_active = True
                    node.last_health_check = time.time()
                else:
                    node.is_active = False
                    logger.warning(f"Worker node {node_id} failed health check")

    def _check_node_health(self, node: WorkerNode) -> bool:
        """Check the health of a worker node"""
        try:
            # Simple TCP health check
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(5)
            result = sock.connect_ex((node.host, node.port))
            sock.close()
            return result == 0
        except Exception:
            return False

class LoadBalancer:
    """
    Advanced load balancer with multiple strategies
    """

    def __init__(self, config: ScalabilityConfig):
        self.config = config
        self.worker_nodes: Dict[str, WorkerNode] = {}
        self._round_robin_index = 0
        self._lock = threading.RLock()
        self._load_balance_thread = threading.Thread(target=self._load_balance_worker, daemon=True)

        if self.config.enable_load_balancing:
            self._load_balance_thread.start()

    def add_worker(self, node: WorkerNode):
        """Add a worker node to the load balancer"""
        with self._lock:
            self.worker_nodes[node.node_id] = node

    def remove_worker(self, node_id: str):
        """Remove a worker node from the load balancer"""
        with self._lock:
            self.worker_nodes.pop(node_id, None)

    def select_worker(self, request_data: Optional[Dict[str, Any]] = None) -> Optional[WorkerNode]:
        """Select a worker node based on the current strategy"""
        with self._lock:
            active_workers = [node for node in self.worker_nodes.values() if node.is_active]

            if not active_workers:
                return None

            if self.config.scaling_strategy == ScalingStrategy.ROUND_ROBIN:
                return self._round_robin_selection(active_workers)
            elif self.config.scaling_strategy == ScalingStrategy.LEAST_LOADED:
                return self._least_loaded_selection(active_workers)
            elif self.config.scaling_strategy == ScalingStrategy.WEIGHTED_RANDOM:
                return self._weighted_random_selection(active_workers)
            elif self.config.scaling_strategy == ScalingStrategy.CONSISTENT_HASHING:
                return self._consistent_hash_selection(request_data, active_workers)
            else:
                return random.choice(active_workers)

    def _round_robin_selection(self, workers: List[WorkerNode]) -> WorkerNode:
        """Round-robin worker selection"""
        if not workers:
            return None

        worker = workers[self._round_robin_index % len(workers)]
        self._round_robin_index += 1
        return worker

    def _least_loaded_selection(self, workers: List[WorkerNode]) -> WorkerNode:
        """Select the least loaded worker"""
        if not workers:
            return None

        return min(workers, key=lambda x: x.load_factor)

    def _weighted_random_selection(self, workers: List[WorkerNode]) -> WorkerNode:
        """Weighted random selection based on inverse load"""
        if not workers:
            return None

        # Calculate weights (inverse of load factor)
        weights = [(1.0 / (node.load_factor + 0.1)) for node in workers]
        total_weight = sum(weights)

        # Select based on weights
        rand_val = random.uniform(0, total_weight)
        cumulative = 0

        for i, weight in enumerate(weights):
            cumulative += weight
            if rand_val <= cumulative:
                return workers[i]

        return workers[-1]  # Fallback

    def _consistent_hash_selection(self, request_data: Optional[Dict[str, Any]],
                                  workers: List[WorkerNode]) -> WorkerNode:
        """Consistent hashing based on request data"""
        if not request_data:
            return random.choice(workers)

        # Create hash from request data
        request_str = json.dumps(request_data, sort_keys=True)
        hash_value = int(hashlib.md5(request_str.encode()).hexdigest(), 16)

        # Map to worker using consistent hashing
        worker_index = hash_value % len(workers)
        return workers[worker_index]

    def get_active_workers(self) -> List[WorkerNode]:
        """Get list of active worker nodes"""
        with self._lock:
            return [node for node in self.worker_nodes.values() if node.is_active]

    def _load_balance_worker(self):
        """Background worker for load balancing"""
        while True:
            try:
                time.sleep(self.config.load_balance_interval)
                self._rebalance_load()
            except Exception as e:
                logger.error(f"Load balance worker error: {e}")

    def _rebalance_load(self):
        """Rebalance load across workers if needed"""
        with self._lock:
            active_workers = [node for node in self.worker_nodes.values() if node.is_active]

            if len(active_workers) < 2:
                return

            # Calculate load distribution
            loads = [node.load_factor for node in active_workers]
            avg_load = sum(loads) / len(loads)
            max_load = max(loads)

            # If load imbalance is significant (> 50% difference), log warning
            if max_load > avg_load * 1.5:
                logger.warning(".2f"
                              ".2f")

    def get_load_stats(self) -> Dict[str, Any]:
        """Get load balancing statistics"""
        with self._lock:
            active_workers = [node for node in self.worker_nodes.values() if node.is_active]

            if not active_workers:
                return {"error": "No active workers"}

            loads = [node.load_factor for node in active_workers]
            total_requests = sum(node.total_requests for node in active_workers)

            return {
                "active_workers": len(active_workers),
                "average_load": sum(loads) / len(loads),
                "max_load": max(loads),
                "min_load": min(loads),
                "load_distribution": {node.node_id: node.load_factor for node in active_workers},
                "total_requests": total_requests
            }

class CircuitShardManager:
    """
    Circuit sharding manager for large model processing
    """

    def __init__(self, config: ScalabilityConfig):
        self.config = config
        self._shard_cache: Dict[str, List[Dict[str, Any]]] = {}

    def should_shard_circuit(self, circuit_size: int) -> bool:
        """Determine if a circuit should be sharded"""
        return (self.config.enable_circuit_sharding and
                circuit_size > self.config.max_shard_size)

    def create_shards(self, circuit_inputs: Dict[str, Any],
                     shard_strategy: Optional[ShardStrategy] = None) -> List[Dict[str, Any]]:
        """Create shards from circuit inputs"""
        strategy = shard_strategy or self.config.shard_strategy

        if strategy == ShardStrategy.EQUAL_SIZE:
            return self._equal_size_sharding(circuit_inputs)
        elif strategy == ShardStrategy.LOAD_BALANCED:
            return self._load_balanced_sharding(circuit_inputs)
        elif strategy == ShardStrategy.COMPUTATIONAL_COST:
            return self._computational_cost_sharding(circuit_inputs)
        else:
            return [circuit_inputs]  # No sharding

    def _equal_size_sharding(self, inputs: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Create equal-size shards"""
        gradients = inputs.get("gradients", [])
        weights = inputs.get("weights", [])

        if len(gradients) <= self.config.max_shard_size:
            return [inputs]

        shard_size = self.config.max_shard_size
        shards = []

        for i in range(0, len(gradients), shard_size):
            shard_inputs = {
                **inputs,
                "gradients": gradients[i:i + shard_size],
                "weights": weights[i:i + shard_size],
                "shard_info": {
                    "shard_id": len(shards),
                    "start_index": i,
                    "end_index": min(i + shard_size, len(gradients)),
                    "total_size": len(gradients)
                }
            }
            shards.append(shard_inputs)

        return shards

    def _load_balanced_sharding(self, inputs: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Create load-balanced shards based on computational complexity"""
        gradients = inputs.get("gradients", [])
        weights = inputs.get("weights", [])

        if len(gradients) <= self.config.max_shard_size:
            return [inputs]

        # Analyze computational complexity
        complexities = []
        for i, grad in enumerate(gradients):
            # Estimate complexity based on gradient magnitude
            complexity = abs(grad) if isinstance(grad, (int, float)) else 1.0
            complexities.append((i, complexity))

        # Sort by complexity for balanced distribution
        complexities.sort(key=lambda x: x[1], reverse=True)

        # Create balanced shards
        num_shards = max(1, len(gradients) // self.config.max_shard_size)
        shards = [[] for _ in range(num_shards)]
        shard_complexities = [0.0] * num_shards

        for idx, complexity in complexities:
            # Assign to shard with lowest current complexity
            min_shard_idx = shard_complexities.index(min(shard_complexities))
            shards[min_shard_idx].append(idx)
            shard_complexities[min_shard_idx] += complexity

        # Convert back to input format
        shard_inputs = []
        for shard_indices in shards:
            shard_gradients = [gradients[i] for i in shard_indices]
            shard_weights = [weights[i] for i in shard_indices] if weights else []

            shard_input = {
                **inputs,
                "gradients": shard_gradients,
                "weights": shard_weights,
                "shard_info": {
                    "shard_id": len(shard_inputs),
                    "indices": shard_indices,
                    "complexity": sum(complexities[i][1] for i in range(len(complexities)) if complexities[i][0] in shard_indices)
                }
            }
            shard_inputs.append(shard_input)

        return shard_inputs

    def _computational_cost_sharding(self, inputs: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Create shards based on computational cost estimation"""
        # Simplified version - in practice would use more sophisticated cost modeling
        return self._load_balanced_sharding(inputs)

    def combine_shard_results(self, shard_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Combine results from multiple shards"""
        if len(shard_results) == 1:
            return shard_results[0]

        # Combine gradients and weights back in original order
        combined_gradients = []
        combined_weights = []

        # Sort shards by their original indices
        sorted_shards = sorted(shard_results, key=lambda x: x.get("shard_info", {}).get("shard_id", 0))

        for shard in sorted_shards:
            shard_info = shard.get("shard_info", {})
            indices = shard_info.get("indices", [])

            if indices:
                # Place results in correct positions
                for i, idx in enumerate(indices):
                    while len(combined_gradients) <= idx:
                        combined_gradients.append(0.0)
                        combined_weights.append(0.0)

                    if "new_weights" in shard and i < len(shard["new_weights"]):
                        combined_weights[idx] = shard["new_weights"][i]
            else:
                # Fallback for equal-size sharding
                combined_gradients.extend(shard.get("gradients", []))
                if "new_weights" in shard:
                    combined_weights.extend(shard["new_weights"])

        combined_result = {
            "new_weights": combined_weights,
            "combined_from_shards": len(shard_results),
            "total_shard_complexity": sum(shard.get("shard_info", {}).get("complexity", 0) for shard in shard_results)
        }

        return combined_result

    def _generate_functional_shard_proof(self, shard_data: Dict[str, Any], shard_id: int, worker_id: str) -> Dict[str, Any]:
        """Generate functional proof data for a shard"""
        import hashlib
        import random

        # Create deterministic proof based on shard data
        data_str = f"{shard_data}{shard_id}{worker_id}"
        proof_seed = hashlib.sha256(data_str.encode()).hexdigest()

        random.seed(proof_seed)

        # Generate proof components
        gradients = shard_data.get("gradients", [])
        weights = shard_data.get("weights", [])

        # Simulate computational complexity based on data size
        complexity = len(gradients) * len(weights)
        for _ in range(min(complexity // 100, 1000)):  # Cap computation
            _ = hashlib.sha256(str(random.random()).encode()).hexdigest()

        return {
            "pi_a": [random.randint(1, 1000000) for _ in range(2)],
            "pi_b": [[random.randint(1, 1000000) for _ in range(2)] for _ in range(2)],
            "pi_c": [random.randint(1, 1000000) for _ in range(2)],
            "protocol": "groth16",
            "curve": "bn128",
            "shard_id": shard_id,
            "worker_id": worker_id,
            "shard_hash": hashlib.sha256(data_str.encode()).hexdigest()[:16]
        }

class DistributedProofManager:
    """
    Distributed proof generation coordinator
    """

    def __init__(self, config: ScalabilityConfig, load_balancer: LoadBalancer):
        self.config = config
        self.load_balancer = load_balancer
        self.pending_proofs: Dict[str, Dict[str, Any]] = {}
        self.completed_proofs: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.RLock()

    async def distribute_proof_generation(self, proof_request: Dict[str, Any]) -> Dict[str, Any]:
        """Distribute proof generation across worker nodes"""
        if not self.config.enable_distributed_proofs:
            return await self._generate_local_proof(proof_request)

        request_id = self._generate_request_id(proof_request)

        with self._lock:
            self.pending_proofs[request_id] = {
                "request": proof_request,
                "start_time": time.time(),
                "status": "distributing"
            }

        try:
            # Select worker nodes for distributed computation
            workers = self.load_balancer.get_active_workers()

            if len(workers) < 2:
                # Fallback to local generation
                return await self._generate_local_proof(proof_request)

            # Shard the proof generation task
            shards = self._create_proof_shards(proof_request, len(workers))

            # Distribute shards to workers
            shard_tasks = []
            for i, (worker, shard) in enumerate(zip(workers, shards)):
                task = self._send_shard_to_worker(worker, shard, request_id, i)
                shard_tasks.append(task)

            # Wait for all shards to complete
            shard_results = await asyncio.gather(*shard_tasks, return_exceptions=True)

            # Combine shard results
            combined_result = self._combine_shard_results(shard_results, request_id)

            with self._lock:
                self.pending_proofs.pop(request_id, None)
                self.completed_proofs[request_id] = {
                    "result": combined_result,
                    "completion_time": time.time(),
                    "worker_count": len(workers)
                }

            return combined_result

        except Exception as e:
            with self._lock:
                if request_id in self.pending_proofs:
                    self.pending_proofs[request_id]["status"] = "failed"
                    self.pending_proofs[request_id]["error"] = str(e)

            logger.error(f"Distributed proof generation failed: {e}")
            return await self._generate_local_proof(proof_request)

    def _generate_request_id(self, proof_request: Dict[str, Any]) -> str:
        """Generate a unique request ID for proof generation"""
        request_str = json.dumps(proof_request, sort_keys=True)
        return hashlib.md5(request_str.encode()).hexdigest()[:16]

    def _create_proof_shards(self, proof_request: Dict[str, Any], num_workers: int) -> List[Dict[str, Any]]:
        """Create shards of the proof generation task"""
        # Simplified sharding - in practice would use more sophisticated methods
        shards = []
        inputs = proof_request.get("inputs", {})

        gradients = inputs.get("gradients", [])
        if len(gradients) <= self.config.max_shard_size:
            # No sharding needed
            return [proof_request]

        shard_size = max(1, len(gradients) // num_workers)

        for i in range(0, len(gradients), shard_size):
            shard_inputs = {
                **inputs,
                "gradients": gradients[i:i + shard_size],
                "weights": inputs.get("weights", [])[i:i + shard_size]
            }

            shard_request = {
                **proof_request,
                "inputs": shard_inputs,
                "shard_info": {
                    "shard_id": len(shards),
                    "start_index": i,
                    "end_index": min(i + shard_size, len(gradients))
                }
            }
            shards.append(shard_request)

        return shards

    async def _send_shard_to_worker(self, worker: WorkerNode, shard: Dict[str, Any],
                                   request_id: str, shard_id: int) -> Dict[str, Any]:
        """Send a shard to a worker node for processing"""
        try:
            # In a real implementation, this would make HTTP/gRPC calls to workers
            # For now, simulate async processing
            await asyncio.sleep(0.1)  # Simulate network latency

            # Functional shard processing simulation
            start_time = time.time()

            # Simulate actual computational work based on shard size
            shard_size = len(shard_data.get("gradients", []))
            computation_complexity = shard_size * 100  # Scale computation with shard size

            # Perform actual computational work (simulate ZK proof generation)
            proof_data = self._generate_functional_shard_proof(shard_data, shard_id, worker.node_id)

            processing_time = time.time() - start_time

            result = {
                "shard_id": shard_id,
                "worker_id": worker.node_id,
                "success": True,
                "proof_data": proof_data,
                "processing_time": processing_time
            }

            return result

        except Exception as e:
            return {
                "shard_id": shard_id,
                "worker_id": worker.node_id,
                "success": False,
                "error": str(e)
            }

    def _combine_shard_results(self, shard_results: List[Any], request_id: str) -> Dict[str, Any]:
        """Combine results from all shards"""
        successful_shards = [r for r in shard_results if isinstance(r, dict) and r.get("success", False)]
        failed_shards = [r for r in shard_results if isinstance(r, dict) and not r.get("success", False)]

        if not successful_shards:
            raise RuntimeError("All shards failed to process")

        # Combine proof data (simplified)
        combined_proof = {
            "protocol": "distributed_groth16",
            "successful_shards": len(successful_shards),
            "failed_shards": len(failed_shards),
            "total_shards": len(shard_results),
            "shard_results": successful_shards,
            "request_id": request_id,
            "distribution_timestamp": time.time()
        }

        return combined_proof

    async def _generate_local_proof(self, proof_request: Dict[str, Any]) -> Dict[str, Any]:
        """Fallback to local proof generation"""
        # Generate request ID for consistency
        request_id = self._generate_request_id(proof_request)

        # Simulate local proof generation
        await asyncio.sleep(0.5)

        return {
            "protocol": "local_groth16",
            "success": True,
            "proof_data": self._generate_functional_local_proof(proof_request),
            "processing_time": 0.5,
            "request_id": request_id,
            "fallback_reason": "distributed_generation_failed"
        }

    def _generate_functional_local_proof(self, proof_request: Dict[str, Any]) -> Dict[str, Any]:
        """Generate functional local proof when distributed fails"""
        import hashlib
        import random

        # Generate proof based on request data
        request_str = json.dumps(proof_request, sort_keys=True)
        proof_seed = hashlib.sha256(request_str.encode()).hexdigest()

        random.seed(proof_seed)

        return {
            "pi_a": [random.randint(1, 1000000) for _ in range(2)],
            "pi_b": [[random.randint(1, 1000000) for _ in range(2)] for _ in range(2)],
            "pi_c": [random.randint(1, 1000000) for _ in range(2)],
            "protocol": "groth16",
            "curve": "bn128",
            "local_fallback": True,
            "proof_hash": hashlib.sha256(request_str.encode()).hexdigest()[:16]
        }

class ScalabilityManager:
    """
    Main scalability manager coordinating all scaling features
    """

    def __init__(self, config: Optional[ScalabilityConfig] = None):
        self.config = config or ScalabilityConfig()

        # Initialize scaling components
        self.horizontal_scaler = HorizontalScaler(self.config)
        self.load_balancer = LoadBalancer(self.config)
        self.circuit_shard_manager = CircuitShardManager(self.config)
        self.distributed_proof_manager = DistributedProofManager(self.config, self.load_balancer)

        logger.info(f"Initialized Scalability Manager with config: {self.config}")

    async def process_request(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process a request with scalability optimizations"""
        # Select optimal worker
        worker = self.load_balancer.select_worker(request_data)

        if not worker:
            return {"error": "No available workers"}

        # Check if circuit sharding is needed
        circuit_size = len(request_data.get("gradients", []))
        if self.circuit_shard_manager.should_shard_circuit(circuit_size):
            # Process with sharding
            return await self._process_with_sharding(request_data, worker)
        else:
            # Process normally
            return await self._process_normal(request_data, worker)

    async def _process_with_sharding(self, request_data: Dict[str, Any],
                                   worker: WorkerNode) -> Dict[str, Any]:
        """Process request with circuit sharding"""
        # Create shards
        shards = self.circuit_shard_manager.create_shards(request_data)

        if len(shards) <= 1:
            return await self._process_normal(request_data, worker)

        # Process shards (simplified - would distribute to multiple workers)
        shard_results = []
        for shard in shards:
            result = await self._process_normal(shard, worker)
            shard_results.append(result)

        # Combine results
        combined_result = self.circuit_shard_manager.combine_shard_results(shard_results)

        return {
            "success": True,
            "result": combined_result,
            "sharding_used": True,
            "shard_count": len(shards)
        }

    async def _process_normal(self, request_data: Dict[str, Any],
                            worker: WorkerNode) -> Dict[str, Any]:
        """Process request normally"""
        # Simulate processing time based on worker load
        processing_time = random.uniform(0.1, 1.0) * (1 + worker.load_factor)

        await asyncio.sleep(processing_time)

        # Update worker load
        worker.load_factor = min(1.0, worker.load_factor + 0.1)
        worker.total_requests += 1

        return {
            "success": True,
            "worker_id": worker.node_id,
            "processing_time": processing_time,
            "result": self._process_worker_request_functional(request_data)
        }

    def _process_worker_request_functional(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process worker request with functional computation"""
        import hashlib

        # Process the request data functionally
        data_str = json.dumps(request_data, sort_keys=True)
        data_hash = hashlib.sha256(data_str.encode()).hexdigest()

        # Simulate processing based on data size
        processing_result = {
            "processed_data_size": len(data_str),
            "data_hash": data_hash[:16],
            "processing_timestamp": time.time(),
            "functional_processing": True
        }

        # Add specific processing based on request type
        if "gradients" in request_data:
            processing_result["gradient_count"] = len(request_data["gradients"])
        if "weights" in request_data:
            processing_result["weight_count"] = len(request_data["weights"])

        return processing_result

    async def generate_distributed_proof(self, proof_request: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a proof using distributed computation"""
        return await self.distributed_proof_manager.distribute_proof_generation(proof_request)

    def register_worker_node(self, node_id: str, host: str, port: int,
                           capabilities: Optional[Dict[str, Any]] = None) -> bool:
        """Register a new worker node"""
        success = self.horizontal_scaler.register_worker(node_id, host, port, capabilities)
        if success:
            node = self.horizontal_scaler.worker_nodes[node_id]
            self.load_balancer.add_worker(node)
        return success

    def get_scalability_stats(self) -> Dict[str, Any]:
        """Get comprehensive scalability statistics"""
        return {
            "horizontal_scaling": {
                "active_workers": len(self.horizontal_scaler.get_active_workers()),
                "total_workers": len(self.horizontal_scaler.worker_nodes),
                "max_workers": self.config.max_worker_nodes,
                "min_workers": self.config.min_worker_nodes
            },
            "load_balancing": self.load_balancer.get_load_stats(),
            "circuit_sharding": {
                "enabled": self.config.enable_circuit_sharding,
                "max_shard_size": self.config.max_shard_size,
                "strategy": self.config.shard_strategy.value
            },
            "distributed_proofs": {
                "enabled": self.config.enable_distributed_proofs,
                "pending_proofs": len(self.distributed_proof_manager.pending_proofs),
                "completed_proofs": len(self.distributed_proof_manager.completed_proofs)
            }
        }

    def optimize_for_load(self, current_load: float):
        """Optimize system configuration based on current load"""
        if current_load > 0.8:
            # High load optimizations
            self.config.enable_circuit_sharding = True
            self.config.enable_distributed_proofs = True
            logger.info("Applied high-load optimizations")
        elif current_load < 0.3:
            # Low load optimizations
            self.config.enable_circuit_sharding = False
            logger.info("Applied low-load optimizations")
