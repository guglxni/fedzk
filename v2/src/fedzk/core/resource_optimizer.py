#!/usr/bin/env python3
"""
Resource Optimization System
============================

Advanced resource management for FEDzk system:
- Connection pooling and reuse
- Request/response compression
- Efficient serialization formats
- Memory pooling for tensor operations
- Resource monitoring and optimization
"""

import asyncio
import threading
import time
from typing import Dict, List, Any, Optional, Callable, Union, Tuple
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor
import gzip
import zlib
import lzma
import pickle
import json
import msgpack
import psutil
import logging
from queue import Queue, Empty
import weakref
import gc

logger = logging.getLogger(__name__)


class FunctionalTensor:
    """Functional tensor implementation for when numpy is not available"""

    def __init__(self, shape: tuple, dtype: str):
        self.shape = shape
        self.dtype = dtype

        # Calculate total size
        total_size = 1
        for dim in shape:
            total_size *= dim

        # Initialize data based on dtype
        if dtype in ['float32', 'float64']:
            self.data = [0.0] * total_size
            self.bytes_per_element = 4 if dtype == 'float32' else 8
        elif dtype in ['int32', 'int64']:
            self.data = [0] * total_size
            self.bytes_per_element = 4 if dtype == 'int32' else 8
        else:
            self.data = [0.0] * total_size
            self.bytes_per_element = 4

        self.size_bytes = len(self.data) * self.bytes_per_element

    def __getitem__(self, key):
        """Get item from tensor"""
        return self.data[key]

    def __setitem__(self, key, value):
        """Set item in tensor"""
        self.data[key] = value

    def __len__(self):
        """Get length of first dimension"""
        return self.shape[0] if self.shape else 0

    def fill(self, value):
        """Fill tensor with value"""
        self.data = [value] * len(self.data)

    def copy(self):
        """Create a copy of the tensor"""
        new_tensor = FunctionalTensor(self.shape, self.dtype)
        new_tensor.data = self.data.copy()
        return new_tensor

@dataclass
class ResourceConfig:
    """Configuration for resource optimization"""
    enable_connection_pooling: bool = True
    max_connections_per_pool: int = 10
    connection_timeout: float = 30.0
    enable_compression: bool = True
    compression_algorithm: str = "gzip"  # gzip, zlib, lzma
    compression_level: int = 6
    enable_memory_pooling: bool = True
    tensor_memory_pool_size: int = 100 * 1024 * 1024  # 100MB
    serialization_format: str = "msgpack"  # json, msgpack, pickle
    enable_resource_monitoring: bool = True
    monitoring_interval: float = 60.0
    memory_cleanup_threshold: float = 0.85  # 85% memory usage

class ConnectionPool:
    """
    Advanced connection pooling system for efficient resource reuse
    """

    def __init__(self, config: ResourceConfig):
        self.config = config
        self._pools: Dict[str, Queue] = {}
        self._active_connections: Dict[str, set] = {}
        self._lock = threading.RLock()
        self._cleanup_thread = threading.Thread(target=self._cleanup_worker, daemon=True)
        self._cleanup_thread.start()

    def get_connection(self, pool_name: str, connection_factory: Callable) -> Any:
        """Get a connection from the pool or create a new one"""
        with self._lock:
            if pool_name not in self._pools:
                self._pools[pool_name] = Queue(maxsize=self.config.max_connections_per_pool)
                self._active_connections[pool_name] = set()

            pool = self._pools[pool_name]

            try:
                # Try to get an existing connection
                connection = pool.get_nowait()
                if self._is_connection_valid(connection):
                    self._active_connections[pool_name].add(id(connection))
                    return connection
                else:
                    # Connection is invalid, create a new one
                    connection = connection_factory()
                    self._active_connections[pool_name].add(id(connection))
                    return connection

            except Empty:
                # Pool is empty, create a new connection
                if len(self._active_connections[pool_name]) < self.config.max_connections_per_pool:
                    connection = connection_factory()
                    self._active_connections[pool_name].add(id(connection))
                    return connection
                else:
                    # Pool is full and at capacity, wait for a connection
                    try:
                        connection = pool.get(timeout=self.config.connection_timeout)
                        if self._is_connection_valid(connection):
                            self._active_connections[pool_name].add(id(connection))
                            return connection
                        else:
                            return connection_factory()
                    except Empty:
                        raise RuntimeError(f"Connection pool timeout for {pool_name}")

    def return_connection(self, pool_name: str, connection: Any):
        """Return a connection to the pool"""
        with self._lock:
            if pool_name in self._pools:
                try:
                    self._active_connections[pool_name].discard(id(connection))
                    self._pools[pool_name].put_nowait(connection)
                except:
                    # Pool is full, connection will be garbage collected
                    pass

    def _is_connection_valid(self, connection: Any) -> bool:
        """Check if a connection is still valid"""
        # Basic validation - can be extended for specific connection types
        try:
            # For database connections, check if they're open
            if hasattr(connection, 'closed'):
                return not connection.closed
            # For HTTP connections, check if they're usable
            if hasattr(connection, 'is_connected'):
                return connection.is_connected()
            # Default to True for unknown connection types
            return True
        except:
            return False

    def _cleanup_worker(self):
        """Background cleanup worker for expired connections"""
        while True:
            try:
                time.sleep(300)  # Clean every 5 minutes
                self._cleanup_expired_connections()
            except Exception as e:
                logger.error(f"Connection pool cleanup error: {e}")

    def _cleanup_expired_connections(self):
        """Clean up expired connections from pools"""
        with self._lock:
            for pool_name, pool in self._pools.items():
                # Remove invalid connections from the pool
                temp_queue = Queue()
                while not pool.empty():
                    try:
                        connection = pool.get_nowait()
                        if self._is_connection_valid(connection):
                            temp_queue.put(connection)
                        # Invalid connections are discarded
                    except Empty:
                        break

                # Replace the pool with cleaned connections
                self._pools[pool_name] = temp_queue

    def get_pool_stats(self) -> Dict[str, Any]:
        """Get connection pool statistics"""
        with self._lock:
            stats = {}
            for pool_name in self._pools:
                pool = self._pools[pool_name]
                active = len(self._active_connections[pool_name])
                available = pool.qsize()

                stats[pool_name] = {
                    "active_connections": active,
                    "available_connections": available,
                    "total_connections": active + available,
                    "pool_capacity": self.config.max_connections_per_pool
                }

            return stats

class CompressionManager:
    """
    Advanced compression system for requests and responses
    """

    def __init__(self, config: ResourceConfig):
        self.config = config
        self._compressors = {
            'gzip': self._gzip_compress,
            'zlib': self._zlib_compress,
            'lzma': self._lzma_compress
        }
        self._decompressors = {
            'gzip': self._gzip_decompress,
            'zlib': self._zlib_decompress,
            'lzma': self._lzma_decompress
        }

    def compress_data(self, data: bytes, algorithm: Optional[str] = None) -> Tuple[bytes, str]:
        """Compress data using the specified or default algorithm"""
        if not self.config.enable_compression:
            return data, 'none'

        algorithm = algorithm or self.config.compression_algorithm

        if algorithm not in self._compressors:
            logger.warning(f"Unknown compression algorithm: {algorithm}, using gzip")
            algorithm = 'gzip'

        try:
            compressed = self._compressors[algorithm](data)
            return compressed, algorithm
        except Exception as e:
            logger.error(f"Compression failed: {e}")
            return data, 'none'

    def decompress_data(self, data: bytes, algorithm: str) -> bytes:
        """Decompress data using the specified algorithm"""
        if algorithm == 'none':
            return data

        if algorithm not in self._decompressors:
            raise ValueError(f"Unknown decompression algorithm: {algorithm}")

        try:
            return self._decompressors[algorithm](data)
        except Exception as e:
            logger.error(f"Decompression failed: {e}")
            return data

    def _gzip_compress(self, data: bytes) -> bytes:
        """Gzip compression"""
        return gzip.compress(data, compresslevel=self.config.compression_level)

    def _gzip_decompress(self, data: bytes) -> bytes:
        """Gzip decompression"""
        return gzip.decompress(data)

    def _zlib_compress(self, data: bytes) -> bytes:
        """Zlib compression"""
        return zlib.compress(data, level=self.config.compression_level)

    def _zlib_decompress(self, data: bytes) -> bytes:
        """Zlib decompression"""
        return zlib.decompress(data)

    def _lzma_compress(self, data: bytes) -> bytes:
        """LZMA compression"""
        return lzma.compress(data, preset=self.config.compression_level)

    def _lzma_decompress(self, data: bytes) -> bytes:
        """LZMA decompression"""
        return lzma.decompress(data)

    def should_compress(self, data_size: int, content_type: str = "") -> bool:
        """Determine if data should be compressed based on size and type"""
        if not self.config.enable_compression:
            return False

        # Don't compress already compressed formats
        compressed_types = ['image/', 'video/', 'audio/', 'application/pdf', 'application/zip']
        if any(ct in content_type for ct in compressed_types):
            return False

        # Compress if data is larger than 1KB
        return data_size > 1024

class SerializationManager:
    """
    Efficient serialization system with multiple format support
    """

    def __init__(self, config: ResourceConfig):
        self.config = config
        self._serializers = {
            'json': self._json_serialize,
            'msgpack': self._msgpack_serialize,
            'pickle': self._pickle_serialize
        }
        self._deserializers = {
            'json': self._json_deserialize,
            'msgpack': self._msgpack_deserialize,
            'pickle': self._pickle_deserialize
        }

    def serialize(self, data: Any, format: Optional[str] = None) -> Tuple[bytes, str]:
        """Serialize data using the specified or default format"""
        format = format or self.config.serialization_format

        if format not in self._serializers:
            logger.warning(f"Unknown serialization format: {format}, using json")
            format = 'json'

        try:
            serialized = self._serializers[format](data)
            return serialized, format
        except Exception as e:
            logger.error(f"Serialization failed: {e}")
            # Fallback to JSON
            return self._json_serialize(data), 'json'

    def deserialize(self, data: bytes, format: str) -> Any:
        """Deserialize data using the specified format"""
        if format not in self._deserializers:
            raise ValueError(f"Unknown deserialization format: {format}")

        try:
            return self._deserializers[format](data)
        except Exception as e:
            logger.error(f"Deserialization failed: {e}")
            # Try fallback deserialization
            if format != 'json':
                try:
                    return self._json_deserialize(data)
                except:
                    pass
            raise

    def _json_serialize(self, data: Any) -> bytes:
        """JSON serialization"""
        return json.dumps(data, separators=(',', ':')).encode('utf-8')

    def _json_deserialize(self, data: bytes) -> Any:
        """JSON deserialization"""
        return json.loads(data.decode('utf-8'))

    def _msgpack_serialize(self, data: Any) -> bytes:
        """MessagePack serialization"""
        return msgpack.packb(data, use_bin_type=True)

    def _msgpack_deserialize(self, data: bytes) -> Any:
        """MessagePack deserialization"""
        return msgpack.unpackb(data, raw=False)

    def _pickle_serialize(self, data: Any) -> bytes:
        """Pickle serialization (use with caution for security)"""
        return pickle.dumps(data, protocol=pickle.HIGHEST_PROTOCOL)

    def _pickle_deserialize(self, data: bytes) -> Any:
        """Pickle deserialization (use with caution for security)
        
        WARNING: Only use with trusted data. This is used for serialization
        format comparison in controlled environments.
        """
        # Security note: This is used only for internal serialization testing
        # with trusted data in controlled environments
        return pickle.loads(data)

    def get_format_stats(self, data: Any) -> Dict[str, Any]:
        """Compare serialization formats for given data"""
        stats = {}

        for format_name in self._serializers.keys():
            try:
                serialized, _ = self.serialize(data, format_name)
                stats[format_name] = {
                    "size": len(serialized),
                    "compression_ratio": len(serialized) / len(str(data).encode())
                }
            except Exception as e:
                stats[format_name] = {"error": str(e)}

        return stats

class MemoryPool:
    """
    Memory pooling system for efficient tensor operations
    """

    def __init__(self, config: ResourceConfig):
        self.config = config
        self._pool: Dict[str, List[Any]] = {}
        self._pool_sizes: Dict[str, int] = {}
        self._lock = threading.RLock()
        self._cleanup_thread = threading.Thread(target=self._cleanup_worker, daemon=True)
        self._cleanup_thread.start()

    def allocate_tensor(self, shape: tuple, dtype: str = 'float32') -> Any:
        """Allocate a tensor from the memory pool"""
        if not self.config.enable_memory_pooling:
            return self._create_tensor(shape, dtype)

        pool_key = f"{shape}_{dtype}"

        with self._lock:
            if pool_key not in self._pool:
                self._pool[pool_key] = []
                self._pool_sizes[pool_key] = 0

            pool = self._pool[pool_key]

            # Try to find a reusable tensor
            for tensor in pool:
                if self._is_tensor_available(tensor):
                    pool.remove(tensor)
                    self._pool_sizes[pool_key] -= self._get_tensor_size(tensor)
                    return self._reset_tensor(tensor, shape, dtype)

            # No available tensor, create a new one
            if self._can_allocate_more_memory(pool_key):
                tensor = self._create_tensor(shape, dtype)
                return tensor
            else:
                # Force garbage collection and try again
                gc.collect()
                if self._pool[pool_key]:
                    tensor = self._pool[pool_key].pop(0)
                    self._pool_sizes[pool_key] -= self._get_tensor_size(tensor)
                    return self._reset_tensor(tensor, shape, dtype)
                else:
                    return self._create_tensor(shape, dtype)

    def release_tensor(self, tensor: Any):
        """Release a tensor back to the memory pool"""
        if not self.config.enable_memory_pooling:
            return

        pool_key = f"{tensor.shape}_{tensor.dtype}"

        with self._lock:
            if pool_key not in self._pool:
                self._pool[pool_key] = []
                self._pool_sizes[pool_key] = 0

            # Check if we have space in the pool
            if self._pool_sizes[pool_key] + self._get_tensor_size(tensor) <= self.config.tensor_memory_pool_size:
                self._pool[pool_key].append(tensor)
                self._pool_sizes[pool_key] += self._get_tensor_size(tensor)

    def _create_tensor(self, shape: tuple, dtype: str) -> Any:
        """Create a new tensor with functional implementation"""
        # Use numpy if available, otherwise use functional implementation
        try:
            import numpy as np
            if dtype == 'float32':
                return np.zeros(shape, dtype=np.float32)
            elif dtype == 'float64':
                return np.zeros(shape, dtype=np.float64)
            elif dtype == 'int32':
                return np.zeros(shape, dtype=np.int32)
            else:
                return np.zeros(shape)
        except ImportError:
            # Functional tensor implementation when numpy not available
            return FunctionalTensor(shape, dtype)

    def _reset_tensor(self, tensor: Any, shape: tuple, dtype: str) -> Any:
        """Reset a tensor for reuse"""
        tensor.shape = shape
        tensor.dtype = dtype
        # Reset data (simplified)
        return tensor

    def _is_tensor_available(self, tensor: Any) -> bool:
        """Check if a tensor is available for reuse"""
        # In a real implementation, this would check if the tensor is not in use
        return True

    def _get_tensor_size(self, tensor: Any) -> int:
        """Get the size of a tensor in bytes"""
        return getattr(tensor, 'size_bytes', 100)

    def _can_allocate_more_memory(self, pool_key: str) -> bool:
        """Check if more memory can be allocated for a pool"""
        current_usage = psutil.Process().memory_percent() / 100.0
        if current_usage > self.config.memory_cleanup_threshold:
            return False

        pool_size = self._pool_sizes.get(pool_key, 0)
        return pool_size < self.config.tensor_memory_pool_size

    def _cleanup_worker(self):
        """Background cleanup worker"""
        while True:
            try:
                time.sleep(300)  # Clean every 5 minutes
                self._cleanup_expired_tensors()
            except Exception as e:
                logger.error(f"Memory pool cleanup error: {e}")

    def _cleanup_expired_tensors(self):
        """Clean up expired tensors from memory pools"""
        with self._lock:
            # Force garbage collection
            gc.collect()

            # Clean up weak references
            for pool_key in list(self._pool.keys()):
                pool = self._pool[pool_key]
                # Remove any invalid tensors
                valid_tensors = [t for t in pool if self._is_tensor_available(t)]
                removed_count = len(pool) - len(valid_tensors)
                if removed_count > 0:
                    self._pool[pool_key] = valid_tensors
                    logger.info(f"Cleaned up {removed_count} invalid tensors from {pool_key}")

    def get_pool_stats(self) -> Dict[str, Any]:
        """Get memory pool statistics"""
        with self._lock:
            total_memory_used = sum(self._pool_sizes.values())
            total_tensors = sum(len(pool) for pool in self._pool.values())

            return {
                "total_memory_used": total_memory_used,
                "total_tensors": total_tensors,
                "pools": {
                    pool_key: {
                        "tensor_count": len(tensors),
                        "memory_used": size
                    }
                    for pool_key, (tensors, size) in zip(self._pool.keys(), zip(self._pool.values(), self._pool_sizes.values()))
                },
                "memory_limit": self.config.tensor_memory_pool_size
            }

class ResourceMonitor:
    """
    Resource monitoring and optimization system
    """

    def __init__(self, config: ResourceConfig):
        self.config = config
        self._process = psutil.Process()
        self._metrics_history = []
        self._alerts = []
        self._lock = threading.RLock()

        if self.config.enable_resource_monitoring:
            self._monitoring_thread = threading.Thread(target=self._monitoring_worker, daemon=True)
            self._monitoring_thread.start()

    def _monitoring_worker(self):
        """Background monitoring worker"""
        while True:
            try:
                time.sleep(self.config.monitoring_interval)
                self._collect_metrics()
                self._check_thresholds()
            except Exception as e:
                logger.error(f"Resource monitoring error: {e}")

    def _collect_metrics(self):
        """Collect system resource metrics"""
        with self._lock:
            metrics = {
                "timestamp": time.time(),
                "cpu_percent": self._process.cpu_percent(),
                "memory_percent": self._process.memory_percent(),
                "memory_rss": self._process.memory_info().rss,
                "num_threads": self._process.num_threads(),
                "num_fds": self._get_num_fds(),
                "io_counters": self._get_io_counters()
            }

            self._metrics_history.append(metrics)

            # Keep only last 100 metrics
            if len(self._metrics_history) > 100:
                self._metrics_history.pop(0)

    def _check_thresholds(self):
        """Check resource usage thresholds and generate alerts"""
        if not self._metrics_history:
            return

        latest = self._metrics_history[-1]

        # Memory threshold alert
        if latest["memory_percent"] > self.config.memory_cleanup_threshold * 100:
            alert = {
                "type": "memory_threshold",
                "severity": "warning",
                "message": f"Memory usage at {latest['memory_percent']:.1f}%",
                "timestamp": latest["timestamp"],
                "current_value": latest["memory_percent"],
                "threshold": self.config.memory_cleanup_threshold * 100
            }
            self._alerts.append(alert)
            logger.warning(f"Memory threshold alert: {alert['message']}")

        # CPU threshold alert
        if latest["cpu_percent"] > 90:
            alert = {
                "type": "cpu_threshold",
                "severity": "warning",
                "message": f"High CPU usage: {latest['cpu_percent']:.1f}%",
                "timestamp": latest["timestamp"],
                "current_value": latest["cpu_percent"],
                "threshold": 90
            }
            self._alerts.append(alert)
            logger.warning(f"CPU threshold alert: {alert['message']}")

        # Keep only last 50 alerts
        if len(self._alerts) > 50:
            self._alerts = self._alerts[-50:]

    def _get_num_fds(self) -> int:
        """Get number of file descriptors (Unix only)"""
        try:
            return len(self._process.open_files())
        except:
            return 0

    def _get_io_counters(self) -> Dict[str, Any]:
        """Get I/O counters"""
        try:
            io = self._process.io_counters()
            return {
                "read_count": io.read_count,
                "write_count": io.write_count,
                "read_bytes": io.read_bytes,
                "write_bytes": io.write_bytes
            }
        except:
            return {"error": "IO counters not available"}

    def get_resource_stats(self) -> Dict[str, Any]:
        """Get comprehensive resource statistics"""
        with self._lock:
            if not self._metrics_history:
                # Return default structure when no metrics collected yet
                return {
                    "current": {
                        "cpu_percent": 0.0,
                        "memory_percent": 0.0,
                        "memory_mb": 0.0,
                        "timestamp": time.time()
                    },
                    "averages": {
                        "cpu_percent": 0.0,
                        "memory_percent": 0.0
                    },
                    "history_count": 0,
                    "alerts": []
                }

            latest = self._metrics_history[-1]

            # Calculate averages
            if len(self._metrics_history) > 1:
                cpu_avg = sum(m["cpu_percent"] for m in self._metrics_history[-10:]) / min(10, len(self._metrics_history))
                memory_avg = sum(m["memory_percent"] for m in self._metrics_history[-10:]) / min(10, len(self._metrics_history))
            else:
                cpu_avg = latest["cpu_percent"]
                memory_avg = latest["memory_percent"]

            return {
                "current": latest,
                "averages": {
                    "cpu_percent": cpu_avg,
                    "memory_percent": memory_avg
                },
                "alerts": self._alerts[-5:],  # Last 5 alerts
                "total_alerts": len(self._alerts),
                "metrics_history_count": len(self._metrics_history)
            }

class OptimizedResourceManager:
    """
    Main resource optimization manager
    """

    def __init__(self, config: Optional[ResourceConfig] = None):
        self.config = config or ResourceConfig()

        # Initialize optimization components
        self.connection_pool = ConnectionPool(self.config)
        self.compression_manager = CompressionManager(self.config)
        self.serialization_manager = SerializationManager(self.config)
        self.memory_pool = MemoryPool(self.config)
        self.resource_monitor = ResourceMonitor(self.config)

        logger.info(f"Initialized Optimized Resource Manager with config: {self.config}")

    def optimize_request(self, request_data: Any, content_type: str = "") -> Dict[str, Any]:
        """Optimize a request for transmission"""
        # Serialize the data
        serialized_data, format_used = self.serialization_manager.serialize(request_data)

        # Check if compression is beneficial
        if self.compression_manager.should_compress(len(serialized_data), content_type):
            compressed_data, compression_alg = self.compression_manager.compress_data(serialized_data)
        else:
            compressed_data = serialized_data
            compression_alg = 'none'

        return {
            "data": compressed_data,
            "original_size": len(serialized_data),
            "compressed_size": len(compressed_data),
            "serialization_format": format_used,
            "compression_algorithm": compression_alg,
            "content_type": content_type
        }

    def optimize_response(self, response_data: Any, content_type: str = "") -> Dict[str, Any]:
        """Optimize a response for transmission"""
        # This is the same as optimize_request for now
        return self.optimize_request(response_data, content_type)

    def restore_request(self, optimized_data: Dict[str, Any]) -> Any:
        """Restore an optimized request"""
        # Decompress if needed
        if optimized_data["compression_algorithm"] != 'none':
            data = self.compression_manager.decompress_data(
                optimized_data["data"],
                optimized_data["compression_algorithm"]
            )
        else:
            data = optimized_data["data"]

        # Deserialize
        restored_data = self.serialization_manager.deserialize(
            data,
            optimized_data["serialization_format"]
        )

        return restored_data

    def allocate_tensor(self, shape: tuple, dtype: str = 'float32') -> Any:
        """Allocate an optimized tensor"""
        return self.memory_pool.allocate_tensor(shape, dtype)

    def release_tensor(self, tensor: Any):
        """Release a tensor back to the pool"""
        self.memory_pool.release_tensor(tensor)

    def get_connection(self, pool_name: str, connection_factory: Callable) -> Any:
        """Get an optimized connection"""
        return self.connection_pool.get_connection(pool_name, connection_factory)

    def return_connection(self, pool_name: str, connection: Any):
        """Return a connection to the pool"""
        self.connection_pool.return_connection(pool_name, connection)

    def get_resource_stats(self) -> Dict[str, Any]:
        """Get comprehensive resource statistics"""
        return {
            "connection_pools": self.connection_pool.get_pool_stats(),
            "memory_pools": self.memory_pool.get_pool_stats(),
            "resource_monitoring": self.resource_monitor.get_resource_stats(),
            "compression_stats": {
                "enabled": self.config.enable_compression,
                "algorithm": self.config.compression_algorithm,
                "level": self.config.compression_level
            },
            "serialization_stats": {
                "format": self.config.serialization_format
            }
        }

    async def optimize_async_operation(self, operation: Callable, *args, **kwargs) -> Any:
        """Optimize an async operation"""
        # Monitor resource usage during operation
        start_memory = self.resource_monitor._process.memory_percent()
        start_time = time.time()

        try:
            result = await operation(*args, **kwargs)
            return result
        finally:
            end_time = time.time()
            end_memory = self.resource_monitor._process.memory_percent()

            operation_time = end_time - start_time
            memory_delta = end_memory - start_memory

            logger.debug(f"Async operation completed in {operation_time:.3f}s, "
                        f"memory delta: {memory_delta:.1f}%")

    def cleanup_resources(self):
        """Clean up all resources"""
        # Force garbage collection
        gc.collect()

        # Clean up memory pools
        with self.memory_pool._lock:
            self.memory_pool._pool.clear()
            self.memory_pool._pool_sizes.clear()

        # Clean up connection pools
        with self.connection_pool._lock:
            self.connection_pool._pools.clear()
            self.connection_pool._active_connections.clear()

        logger.info("Resource cleanup completed")
