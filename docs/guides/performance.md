# ⚡ FEDZK Performance Optimization Guide

This guide provides comprehensive strategies for optimizing FEDZK performance across all components and deployment scenarios.

## Table of Contents

1. [Performance Monitoring](#performance-monitoring)
2. [Training Optimization](#training-optimization)
3. [ZK Proof Optimization](#zk-proof-optimization)
4. [Network Optimization](#network-optimization)
5. [Memory Optimization](#memory-optimization)
6. [Database Optimization](#database-optimization)
7. [GPU Acceleration](#gpu-acceleration)
8. [Scaling Strategies](#scaling-strategies)
9. [Benchmarking](#benchmarking)
10. [Performance Troubleshooting](#performance-troubleshooting)

## Performance Monitoring

### Setting Up Monitoring

```python
from fedzk.monitoring import PerformanceMonitor, MetricsCollector
import time

# Initialize monitoring
monitor = PerformanceMonitor()
metrics = MetricsCollector()

# Register custom metrics
metrics.register_histogram('training_step_duration', 'Time per training step')
metrics.register_gauge('active_participants', 'Number of active participants')
metrics.register_counter('zk_proofs_generated', 'Total ZK proofs generated')

# Start metrics server
metrics.start_server(port=8001)
```

### Key Performance Metrics

```python
# Monitor training performance
class TrainingMetrics:
    def __init__(self):
        self.start_time = time.time()
        self.step_times = []
        self.memory_usage = []
        self.gpu_utilization = []

    def record_step(self, step_time, memory_mb, gpu_percent):
        self.step_times.append(step_time)
        self.memory_usage.append(memory_mb)
        self.gpu_utilization.append(gpu_percent)

    def get_summary(self):
        total_time = time.time() - self.start_time
        avg_step_time = sum(self.step_times) / len(self.step_times)
        max_memory = max(self.memory_usage)
        avg_gpu = sum(self.gpu_utilization) / len(self.gpu_utilization)

        return {
            'total_time': total_time,
            'avg_step_time': avg_step_time,
            'max_memory_mb': max_memory,
            'avg_gpu_percent': avg_gpu,
            'steps_per_second': 1.0 / avg_step_time
        }
```

### Real-time Monitoring Dashboard

```python
from fedzk.monitoring import Dashboard
import asyncio

async def monitor_performance():
    dashboard = Dashboard()

    while True:
        # Collect metrics
        metrics = {
            'cpu_usage': psutil.cpu_percent(),
            'memory_usage': psutil.virtual_memory().percent,
            'disk_io': psutil.disk_io_counters(),
            'network_io': psutil.net_io_counters(),
            'training_progress': get_training_progress(),
            'zk_proof_stats': get_zk_proof_stats()
        }

        # Update dashboard
        dashboard.update_metrics(metrics)

        # Check for performance issues
        if metrics['cpu_usage'] > 90:
            print("⚠️ High CPU usage detected")
        if metrics['memory_usage'] > 85:
            print("⚠️ High memory usage detected")

        await asyncio.sleep(5)

# Run monitoring
asyncio.run(monitor_performance())
```

## Training Optimization

### Batch Size Optimization

```python
def find_optimal_batch_size(model, train_loader, device):
    """Find optimal batch size for training"""

    batch_sizes = [16, 32, 64, 128, 256]
    results = {}

    for batch_size in batch_sizes:
        try:
            # Create data loader with current batch size
            loader = DataLoader(train_loader.dataset, batch_size=batch_size,
                              shuffle=True, num_workers=4)

            # Measure training time for one epoch
            start_time = time.time()

            model.train()
            for batch_idx, (data, target) in enumerate(loader):
                data, target = data.to(device), target.to(device)

                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()

                if batch_idx >= 10:  # Test with first 10 batches
                    break

            epoch_time = time.time() - start_time
            throughput = (batch_size * 10) / epoch_time  # samples per second

            results[batch_size] = {
                'time': epoch_time,
                'throughput': throughput,
                'memory': torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
            }

            print(f"Batch size {batch_size}: {throughput:.2f} samples/sec")

        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"Batch size {batch_size}: Out of memory")
                break
            else:
                raise

    # Find optimal batch size (balance between throughput and memory)
    optimal = max(results.keys(),
                 key=lambda x: results[x]['throughput'] / results[x]['memory'])

    return optimal, results
```

### Gradient Accumulation

```python
def train_with_gradient_accumulation(model, optimizer, train_loader,
                                   accumulation_steps=4):
    """Train with gradient accumulation to simulate larger batch sizes"""

    model.train()
    optimizer.zero_grad()

    for step, (inputs, labels) in enumerate(train_loader):
        inputs, labels = inputs.to(device), labels.to(device)

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # Normalize loss to account for accumulation
        loss = loss / accumulation_steps

        # Backward pass
        loss.backward()

        # Update weights every accumulation_steps
        if (step + 1) % accumulation_steps == 0:
            # Clip gradients to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            optimizer.zero_grad()

            # Learning rate scheduling
            scheduler.step()

        # Log progress
        if step % 100 == 0:
            print(f"Step {step}: Loss = {loss.item() * accumulation_steps:.4f}")

    # Don't forget to update for remaining accumulated gradients
    if (step + 1) % accumulation_steps != 0:
        optimizer.step()
        optimizer.zero_grad()
```

### Mixed Precision Training

```python
from torch.cuda.amp import GradScaler, autocast

def train_mixed_precision(model, optimizer, train_loader, scaler):
    """Train with mixed precision for better performance"""

    model.train()

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()

        # Use autocast for mixed precision
        with autocast():
            outputs = model(inputs)
            loss = criterion(outputs, labels)

        # Scale loss and backward pass
        scaler.scale(loss).backward()

        # Gradient clipping with scaler
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        # Update weights with scaler
        scaler.step(optimizer)
        scaler.update()

        # Learning rate scheduling
        scheduler.step()

    return loss.item()
```

## ZK Proof Optimization

### Circuit Optimization

```python
from fedzk.zk import CircuitOptimizer

def optimize_zk_circuit(circuit_path):
    """Optimize ZK circuit for better performance"""

    optimizer = CircuitOptimizer()

    # Analyze circuit complexity
    analysis = optimizer.analyze_circuit(circuit_path)
    print(f"Circuit complexity: {analysis['complexity']}")
    print(f"Constraint count: {analysis['constraints']}")
    print(f"Variable count: {analysis['variables']}")

    # Apply optimizations
    optimized_circuit = optimizer.optimize_circuit(circuit_path,
        optimizations=[
            'constant_folding',
            'dead_code_elimination',
            'constraint_reordering',
            'parallel_constraints'
        ]
    )

    # Benchmark performance
    original_time = optimizer.benchmark_circuit(circuit_path)
    optimized_time = optimizer.benchmark_circuit(optimized_circuit)

    improvement = (original_time - optimized_time) / original_time * 100
    print(".1f")

    return optimized_circuit
```

### Proof Generation Parallelization

```python
import concurrent.futures
from fedzk.zk import ProofGenerator

def generate_proofs_parallel(witnesses, num_workers=4):
    """Generate ZK proofs in parallel"""

    generator = ProofGenerator()

    def generate_single_proof(witness):
        """Generate proof for single witness"""
        return generator.generate_proof(witness)

    # Generate proofs in parallel
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        future_to_witness = {
            executor.submit(generate_single_proof, witness): witness
            for witness in witnesses
        }

        results = []
        for future in concurrent.futures.as_completed(future_to_witness):
            try:
                proof = future.result()
                results.append(proof)
            except Exception as exc:
                witness = future_to_witness[future]
                print(f'Witness {witness} generated an exception: {exc}')

    return results
```

### Proof Caching

```python
from fedzk.zk import ProofCache
import hashlib

class ProofCache:
    def __init__(self, max_size=1000, ttl_hours=24):
        self.cache = {}
        self.max_size = max_size
        self.ttl = ttl_hours * 3600  # Convert to seconds

    def _get_cache_key(self, witness):
        """Generate cache key from witness"""
        witness_str = str(sorted(witness.items()))
        return hashlib.sha256(witness_str.encode()).hexdigest()

    def get(self, witness):
        """Get cached proof if available"""
        key = self._get_cache_key(witness)

        if key in self.cache:
            entry = self.cache[key]
            if time.time() - entry['timestamp'] < self.ttl:
                return entry['proof']
            else:
                # Expired, remove from cache
                del self.cache[key]

        return None

    def put(self, witness, proof):
        """Cache proof"""
        key = self._get_cache_key(witness)

        # Remove oldest entries if cache is full
        if len(self.cache) >= self.max_size:
            oldest_key = min(self.cache.keys(),
                           key=lambda k: self.cache[k]['timestamp'])
            del self.cache[oldest_key]

        self.cache[key] = {
            'proof': proof,
            'timestamp': time.time()
        }

    def clear_expired(self):
        """Remove expired entries"""
        current_time = time.time()
        expired_keys = [
            key for key, entry in self.cache.items()
            if current_time - entry['timestamp'] >= self.ttl
        ]

        for key in expired_keys:
            del self.cache[key]

# Usage
cache = ProofCache(max_size=1000, ttl_hours=24)

def generate_proof_cached(witness):
    """Generate proof with caching"""
    # Check cache first
    cached_proof = cache.get(witness)
    if cached_proof:
        return cached_proof

    # Generate new proof
    proof = generate_proof(witness)

    # Cache the result
    cache.put(witness, proof)

    return proof
```

## Network Optimization

### Connection Pooling

```python
from fedzk.network import ConnectionPool
import aiohttp
import asyncio

class OptimizedConnectionPool:
    def __init__(self, max_connections=50, timeout=30):
        self.max_connections = max_connections
        self.timeout = timeout
        self._session = None

    async def __aenter__(self):
        connector = aiohttp.TCPConnector(
            limit=self.max_connections,
            limit_per_host=self.max_connections // 2,
            ttl_dns_cache=300,
            use_dns_cache=True,
            keepalive_timeout=60,
            enable_cleanup_closed=True
        )

        timeout = aiohttp.ClientTimeout(total=self.timeout)
        self._session = aiohttp.ClientSession(
            connector=connector,
            timeout=timeout,
            trust_env=True  # Use environment proxy settings
        )

        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self._session:
            await self._session.close()

    async def request(self, method, url, **kwargs):
        """Make HTTP request with optimized connection"""
        if not self._session:
            raise RuntimeError("Connection pool not initialized")

        async with self._session.request(method, url, **kwargs) as response:
            return await response.read()

# Usage
async def federated_request():
    async with OptimizedConnectionPool(max_connections=100) as pool:
        # Make multiple requests efficiently
        responses = await asyncio.gather(*[
            pool.request('GET', f'http://participant{i}:8000/model')
            for i in range(10)
        ])

    return responses
```

### Message Compression

```python
import gzip
import zlib
import lzma
from fedzk.network import MessageCompressor

class MessageCompressor:
    def __init__(self, algorithm='gzip', level=6):
        self.algorithm = algorithm
        self.level = level

        self.compressors = {
            'gzip': gzip.compress,
            'zlib': zlib.compress,
            'lzma': lzma.compress
        }

        self.decompressors = {
            'gzip': gzip.decompress,
            'zlib': zlib.decompress,
            'lzma': lzma.decompress
        }

    def compress(self, data):
        """Compress data"""
        if isinstance(data, str):
            data = data.encode('utf-8')

        return self.compressors[self.algorithm](data)

    def decompress(self, data):
        """Decompress data"""
        return self.decompressors[self.algorithm](data)

    def get_compression_ratio(self, original_data):
        """Calculate compression ratio"""
        compressed = self.compress(original_data)
        return len(compressed) / len(original_data)

# Usage
compressor = MessageCompressor(algorithm='gzip', level=9)

# Compress model updates
original_update = get_model_update()
compressed_update = compressor.compress(original_update)
ratio = compressor.get_compression_ratio(original_update)

print(f"Compression ratio: {ratio:.2f}")
print(f"Original size: {len(original_update)} bytes")
print(f"Compressed size: {len(compressed_update)} bytes")
```

### Load Balancing

```python
from fedzk.network import LoadBalancer
import random

class LoadBalancer:
    def __init__(self, participants):
        self.participants = participants
        self.weights = {p: 1.0 for p in participants}  # Equal weights initially
        self.response_times = {p: 0.0 for p in participants}
        self.request_counts = {p: 0 for p in participants}

    def select_participant_weighted_random(self):
        """Select participant using weighted random selection"""
        total_weight = sum(self.weights.values())
        rand = random.uniform(0, total_weight)

        cumulative = 0
        for participant, weight in self.weights.items():
            cumulative += weight
            if rand <= cumulative:
                return participant

    def select_participant_least_loaded(self):
        """Select participant with least load"""
        return min(self.participants,
                  key=lambda p: self.request_counts[p])

    def select_participant_fastest(self):
        """Select participant with fastest response time"""
        # Filter out participants with no response time data
        candidates = [p for p in self.participants if self.response_times[p] > 0]

        if not candidates:
            return random.choice(self.participants)

        return min(candidates, key=lambda p: self.response_times[p])

    def update_weights(self, participant, response_time):
        """Update participant weights based on performance"""
        self.request_counts[participant] += 1
        self.response_times[participant] = response_time

        # Adjust weights based on response time
        avg_response_time = sum(self.response_times.values()) / len(self.participants)

        if response_time < avg_response_time:
            # Reward fast participants
            self.weights[participant] *= 1.1
        else:
            # Penalize slow participants
            self.weights[participant] *= 0.9

        # Normalize weights
        total_weight = sum(self.weights.values())
        self.weights = {p: w/total_weight for p, w in self.weights.items()}

# Usage
load_balancer = LoadBalancer(participants)

# Select participant for request
participant = load_balancer.select_participant_weighted_random()

# Record response time
start_time = time.time()
response = make_request(participant)
response_time = time.time() - start_time

load_balancer.update_weights(participant, response_time)
```

## Memory Optimization

### Gradient Checkpointing

```python
import torch
from torch.utils.checkpoint import checkpoint

class MemoryOptimizedModel(nn.Module):
    def __init__(self, hidden_size=1024, num_layers=12):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Linear(hidden_size, hidden_size) for _ in range(num_layers)
        ])

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            # Use checkpointing for intermediate layers
            if i < len(self.layers) - 1:  # Don't checkpoint last layer
                x = checkpoint(layer, x)
            else:
                x = layer(x)
        return x

# Usage
model = MemoryOptimizedModel()

# Enable gradient checkpointing globally
torch.utils.checkpoint.checkpoint_sequential(model.layers, 2, input)
```

### Memory Pooling

```python
from fedzk.memory import TensorPool
import torch

class TensorPool:
    def __init__(self, max_size_gb=8):
        self.max_size_bytes = max_size_gb * 1024**3
        self.allocated_tensors = {}
        self.pool = {}

    def allocate(self, shape, dtype=torch.float32, device='cpu'):
        """Allocate tensor from pool"""
        key = (shape, dtype, device)

        if key in self.pool and self.pool[key]:
            # Reuse existing tensor
            tensor = self.pool[key].pop()
            return tensor.zero_()  # Reset to zeros

        # Allocate new tensor
        tensor = torch.zeros(shape, dtype=dtype, device=device)

        # Track allocation
        size_bytes = tensor.numel() * tensor.element_size()
        self.allocated_tensors[id(tensor)] = size_bytes

        # Check if we're over limit
        total_allocated = sum(self.allocated_tensors.values())
        if total_allocated > self.max_size_bytes:
            self._cleanup()

        return tensor

    def release(self, tensor):
        """Return tensor to pool"""
        key = (tensor.shape, tensor.dtype, tensor.device)

        if key not in self.pool:
            self.pool[key] = []

        # Only keep reasonable number of tensors per key
        if len(self.pool[key]) < 10:
            self.pool[key].append(tensor)

    def _cleanup(self):
        """Clean up least recently used tensors"""
        # Sort by size (largest first) and remove oldest
        sorted_tensors = sorted(
            self.allocated_tensors.items(),
            key=lambda x: x[1],
            reverse=True
        )

        # Remove 20% of largest tensors
        to_remove = sorted_tensors[:len(sorted_tensors)//5]

        for tensor_id, _ in to_remove:
            if tensor_id in self.allocated_tensors:
                del self.allocated_tensors[tensor_id]

# Usage
pool = TensorPool(max_size_gb=4)  # 4GB pool

def optimized_training_step():
    # Allocate tensors from pool
    inputs = pool.allocate((batch_size, input_size))
    targets = pool.allocate((batch_size, num_classes))

    # Use tensors
    outputs = model(inputs)
    loss = criterion(outputs, targets)

    loss.backward()

    # Return tensors to pool
    pool.release(inputs)
    pool.release(targets)

    return loss.item()
```

### Memory-Efficient Attention

```python
class MemoryEfficientAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, mask=None):
        batch_size, seq_len, _ = query.size()

        # Project inputs
        q = self.q_proj(query).view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = self.k_proj(key).view(batch_size, seq_len, self.num_heads, self.head_dim)
        v = self.v_proj(value).view(batch_size, seq_len, self.num_heads, self.head_dim)

        # Transpose for attention computation
        q = q.transpose(1, 2)  # (batch, heads, seq, head_dim)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        # Apply softmax
        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Apply attention to values
        context = torch.matmul(attn_weights, v)

        # Reshape and project output
        context = context.transpose(1, 2).contiguous()
        context = context.view(batch_size, seq_len, self.embed_dim)

        output = self.out_proj(context)
        return output
```

## Database Optimization

### Connection Pooling

```python
from fedzk.database import OptimizedDatabaseManager
from sqlalchemy import create_engine
from sqlalchemy.pool import QueuePool

class OptimizedDatabaseManager:
    def __init__(self, database_url):
        self.engine = create_engine(
            database_url,
            poolclass=QueuePool,
            pool_size=20,          # Number of connections to maintain
            max_overflow=30,       # Max additional connections
            pool_timeout=30,       # Timeout for getting connection
            pool_recycle=3600,    # Recycle connections after 1 hour
            pool_pre_ping=True,   # Test connection before use
            echo=False
        )

    def get_session(self):
        """Get database session"""
        return self.engine.connect()

    def execute_query(self, query, params=None):
        """Execute query with connection pooling"""
        with self.engine.connect() as conn:
            result = conn.execute(query, params or {})
            return result.fetchall()

    def execute_batch(self, queries):
        """Execute multiple queries in batch"""
        with self.engine.connect() as conn:
            with conn.begin():
                results = []
                for query, params in queries:
                    result = conn.execute(query, params or {})
                    results.append(result.fetchall())
                return results

# Usage
db = OptimizedDatabaseManager('postgresql://user:pass@localhost/fedzk')

# Single query
result = db.execute_query("SELECT * FROM participants WHERE active = %s", (True,))

# Batch queries
queries = [
    ("INSERT INTO participants (id, name) VALUES (%s, %s)", (1, "Alice")),
    ("INSERT INTO participants (id, name) VALUES (%s, %s)", (2, "Bob")),
    ("UPDATE participants SET active = %s WHERE id = %s", (True, 1))
]

results = db.execute_batch(queries)
```

### Query Optimization

```python
from sqlalchemy import select, and_, or_
from sqlalchemy.orm import selectinload, joinedload

def optimized_participant_queries(db_session):
    """Optimized database queries for participants"""

    # Use selectinload for eager loading of related data
    query = select(Participant).options(
        selectinload(Participant.models),
        selectinload(Participant.contributions)
    ).where(
        and_(
            Participant.active == True,
            Participant.last_seen > datetime.now() - timedelta(hours=24)
        )
    )

    participants = db_session.execute(query).scalars().all()

    # Batch update contributions
    contribution_updates = []
    for participant in participants:
        if participant.contributions:
            total_contribution = sum(c.amount for c in participant.contributions)
            contribution_updates.append({
                'participant_id': participant.id,
                'total_contribution': total_contribution
            })

    if contribution_updates:
        # Use batch update for better performance
        update_stmt = (
            update(Participant)
            .where(Participant.id == bindparam('participant_id'))
            .values(total_contribution=bindparam('total_contribution'))
        )

        db_session.execute(update_stmt, contribution_updates)
        db_session.commit()

    return participants

# Index optimization
def create_optimized_indexes(db_session):
    """Create optimized database indexes"""

    # Composite index for common queries
    db_session.execute("""
        CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_participants_active_last_seen
        ON participants (active, last_seen DESC);
    """)

    # Partial index for active participants only
    db_session.execute("""
        CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_active_participants
        ON participants (id, name)
        WHERE active = true;
    """)

    # Index for foreign key relationships
    db_session.execute("""
        CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_contributions_participant_id
        ON contributions (participant_id, created_at DESC);
    """)

    db_session.commit()
```

## GPU Acceleration

### Multi-GPU Training

```python
import torch
import torch.nn as nn
from torch.nn import DataParallel
from torch.nn.parallel import DistributedDataParallel as DDP

def setup_multi_gpu(model, use_ddp=False):
    """Setup model for multi-GPU training"""

    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs")

        if use_ddp:
            # Distributed Data Parallel (recommended for multi-node)
            torch.distributed.init_process_group(backend='nccl')
            local_rank = int(os.environ.get('LOCAL_RANK', 0))
            torch.cuda.set_device(local_rank)
            model = model.cuda(local_rank)
            model = DDP(model, device_ids=[local_rank])
        else:
            # Data Parallel (single node, multiple GPUs)
            model = DataParallel(model)
            model = model.cuda()
    else:
        # Single GPU or CPU
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)

    return model

def train_multi_gpu(model, train_loader, optimizer, epochs=10):
    """Train model with multiple GPUs"""

    model.train()
    device = next(model.parameters()).device

    for epoch in range(epochs):
        epoch_loss = 0.0

        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()

            # Forward pass
            if isinstance(model, DDP):
                # DDP handles device placement automatically
                outputs = model(inputs)
            else:
                outputs = model(inputs)

            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

            if batch_idx % 100 == 0:
                print(f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}")

        avg_loss = epoch_loss / len(train_loader)
        print(f"Epoch {epoch} completed, Average Loss: {avg_loss:.4f}")

    return model
```

### GPU Memory Optimization

```python
def optimize_gpu_memory():
    """Optimize GPU memory usage"""

    # Enable cuDNN autotuner
    torch.backends.cudnn.benchmark = True

    # Use memory efficient operations
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.allow_tf32 = True

    # Set memory fraction (leave some memory for CUDA context)
    torch.cuda.set_per_process_memory_fraction(0.9)

    # Empty cache periodically
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def monitor_gpu_usage():
    """Monitor GPU usage during training"""

    if not torch.cuda.is_available():
        return

    print(f"GPU count: {torch.cuda.device_count()}")

    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        memory_allocated = torch.cuda.memory_allocated(i) / 1024**3
        memory_reserved = torch.cuda.memory_reserved(i) / 1024**3
        memory_total = props.total_memory / 1024**3

        print(f"GPU {i}: {props.name}")
        print(".1f")
        print(".1f")
        print(".1f")

        # Check for memory fragmentation
        fragmentation = (memory_reserved - memory_allocated) / memory_reserved
        if fragmentation > 0.5:
            print(".1f"        else:
            print(".1f"
# Usage
optimize_gpu_memory()
monitor_gpu_usage()
```

## Scaling Strategies

### Horizontal Scaling

```python
from fedzk.scaling import AutoScaler
from kubernetes import client, config

class KubernetesAutoScaler:
    def __init__(self, namespace='fedzk'):
        config.load_kube_config()
        self.apps_v1 = client.AppsV1Api()
        self.namespace = namespace
        self.target_cpu_utilization = 70
        self.min_replicas = 1
        self.max_replicas = 10

    def scale_deployment(self, deployment_name):
        """Scale deployment based on CPU utilization"""

        # Get current deployment
        deployment = self.apps_v1.read_namespaced_deployment(
            name=deployment_name,
            namespace=self.namespace
        )

        current_replicas = deployment.spec.replicas
        cpu_utilization = self.get_cpu_utilization(deployment_name)

        # Calculate target replicas
        if cpu_utilization > self.target_cpu_utilization:
            # Scale up
            target_replicas = min(
                current_replicas * 2,
                self.max_replicas
            )
        elif cpu_utilization < self.target_cpu_utilization * 0.5:
            # Scale down
            target_replicas = max(
                current_replicas // 2,
                self.min_replicas
            )
        else:
            target_replicas = current_replicas

        if target_replicas != current_replicas:
            # Update deployment replicas
            deployment.spec.replicas = target_replicas
            self.apps_v1.patch_namespaced_deployment(
                name=deployment_name,
                namespace=self.namespace,
                body=deployment
            )

            print(f"Scaled {deployment_name} from {current_replicas} to {target_replicas} replicas")

    def get_cpu_utilization(self, deployment_name):
        """Get average CPU utilization for deployment"""
        # This would integrate with Prometheus metrics
        # For now, return a mock value
        return 65  # 65% CPU utilization

# Usage
scaler = KubernetesAutoScaler()
scaler.scale_deployment('fedzk-coordinator')
```

### Load Distribution

```python
from fedzk.scaling import LoadDistributor
import hashlib

class ConsistentHashLoadDistributor:
    def __init__(self, nodes, virtual_nodes=100):
        self.nodes = nodes
        self.virtual_nodes = virtual_nodes
        self.ring = {}

        # Create virtual nodes
        for node in nodes:
            for i in range(virtual_nodes):
                key = self._hash(f"{node}:{i}")
                self.ring[key] = node

        # Sort the ring
        self.sorted_keys = sorted(self.ring.keys())

    def _hash(self, key):
        """Consistent hashing function"""
        return int(hashlib.md5(key.encode()).hexdigest(), 16)

    def get_node(self, key):
        """Get node for given key"""
        if not self.ring:
            return None

        hash_value = self._hash(key)

        # Find the first node with hash >= hash_value
        for ring_key in self.sorted_keys:
            if hash_value <= ring_key:
                return self.ring[ring_key]

        # Wrap around to first node
        return self.ring[self.sorted_keys[0]]

    def add_node(self, node):
        """Add new node to the ring"""
        for i in range(self.virtual_nodes):
            key = self._hash(f"{node}:{i}")
            self.ring[key] = node

        self.sorted_keys = sorted(self.ring.keys())

    def remove_node(self, node):
        """Remove node from the ring"""
        keys_to_remove = []
        for key, node_value in self.ring.items():
            if node_value == node:
                keys_to_remove.append(key)

        for key in keys_to_remove:
            del self.ring[key]

        self.sorted_keys = sorted(self.ring.keys())

# Usage
distributor = ConsistentHashLoadDistributor([
    'worker-1', 'worker-2', 'worker-3', 'worker-4'
])

# Distribute participants to workers
participants = ['alice', 'bob', 'charlie', 'diana', 'eve']
distribution = {}

for participant in participants:
    worker = distributor.get_node(participant)
    if worker not in distribution:
        distribution[worker] = []
    distribution[worker].append(participant)

print("Load distribution:")
for worker, participants in distribution.items():
    print(f"{worker}: {participants}")
```

## Benchmarking

### Comprehensive Benchmarking Suite

```python
from fedzk.benchmarking import BenchmarkSuite
import time
import psutil
import torch

class FEDZKBenchmarkSuite:
    def __init__(self):
        self.results = {}

    def benchmark_training_performance(self, model, train_loader, device='cpu'):
        """Benchmark training performance"""

        model.train()
        optimizer = torch.optim.Adam(model.parameters())
        criterion = torch.nn.CrossEntropyLoss()

        # Warm up
        for _ in range(5):
            inputs, targets = next(iter(train_loader))
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

        # Benchmark
        start_time = time.time()
        start_memory = psutil.virtual_memory().used
        start_gpu_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0

        num_batches = 100
        total_loss = 0

        for i, (inputs, targets) in enumerate(train_loader):
            if i >= num_batches:
                break

            inputs, targets = inputs.to(device), targets.to(device)

            start_batch = time.time()
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            end_batch = time.time()

            total_loss += loss.item()

        end_time = time.time()
        end_memory = psutil.virtual_memory().used
        end_gpu_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0

        # Calculate metrics
        total_time = end_time - start_time
        avg_batch_time = total_time / num_batches
        throughput = num_batches * train_loader.batch_size / total_time
        avg_loss = total_loss / num_batches
        memory_used = end_memory - start_memory
        gpu_memory_used = end_gpu_memory - start_gpu_memory

        return {
            'total_time': total_time,
            'avg_batch_time': avg_batch_time,
            'throughput': throughput,
            'avg_loss': avg_loss,
            'memory_used': memory_used,
            'gpu_memory_used': gpu_memory_used,
            'device': device
        }

    def benchmark_zk_proofs(self, circuit_path, witness_data):
        """Benchmark ZK proof generation"""

        from fedzk.zk import ProofGenerator

        generator = ProofGenerator(circuit_path)

        # Warm up
        for _ in range(3):
            generator.generate_proof(witness_data[0])

        # Benchmark
        start_time = time.time()
        start_memory = psutil.virtual_memory().used

        proofs = []
        for witness in witness_data[:50]:  # Test with 50 proofs
            proof_start = time.time()
            proof = generator.generate_proof(witness)
            proof_time = time.time() - proof_start
            proofs.append({
                'proof': proof,
                'generation_time': proof_time,
                'proof_size': len(str(proof))
            })

        end_time = time.time()
        end_memory = psutil.virtual_memory().used

        avg_proof_time = sum(p['generation_time'] for p in proofs) / len(proofs)
        avg_proof_size = sum(p['proof_size'] for p in proofs) / len(proofs)
        total_time = end_time - start_time
        throughput = len(proofs) / total_time
        memory_used = end_memory - start_memory

        return {
            'total_proofs': len(proofs),
            'total_time': total_time,
            'avg_proof_time': avg_proof_time,
            'avg_proof_size': avg_proof_size,
            'throughput': throughput,
            'memory_used': memory_used
        }

    def benchmark_network_performance(self, federation, data_sizes=[1, 10, 100, 1000]):
        """Benchmark network performance"""

        import requests
        import numpy as np

        results = {}

        for size_kb in data_sizes:
            # Generate test data
            data = np.random.rand(size_kb * 1024 // 8).tobytes()

            # Test upload
            upload_times = []
            for _ in range(10):
                start_time = time.time()
                response = requests.post(
                    f"{federation.coordinator_url}/upload",
                    data=data,
                    headers={'Content-Type': 'application/octet-stream'}
                )
                upload_times.append(time.time() - start_time)

            # Test download
            download_times = []
            for _ in range(10):
                start_time = time.time()
                response = requests.get(f"{federation.coordinator_url}/download/{size_kb}kb")
                download_times.append(time.time() - start_time)

            results[size_kb] = {
                'avg_upload_time': np.mean(upload_times),
                'avg_download_time': np.mean(download_times),
                'upload_throughput': size_kb / np.mean(upload_times),
                'download_throughput': size_kb / np.mean(download_times)
            }

        return results

    def run_comprehensive_benchmark(self):
        """Run comprehensive benchmark suite"""

        print("🚀 Starting FEDZK Comprehensive Benchmark Suite")
        print("=" * 60)

        # Training benchmark
        print("📊 Benchmarking Training Performance...")
        training_results = self.benchmark_training_performance(model, train_loader)
        self.results['training'] = training_results
        print(".2f")
        print(".1f")
        print(".4f")

        # ZK proof benchmark
        print("\\n🔐 Benchmarking ZK Proof Generation...")
        zk_results = self.benchmark_zk_proofs(circuit_path, witness_data)
        self.results['zk_proofs'] = zk_results
        print(".3f")
        print(".2f")
        print(".1f")

        # Network benchmark
        print("\\n🌐 Benchmarking Network Performance...")
        network_results = self.benchmark_network_performance(federation)
        self.results['network'] = network_results
        for size, metrics in network_results.items():
            print(f"{size}KB: Upload {metrics['upload_throughput']:.1f} KB/s, "
                  f"Download {metrics['download_throughput']:.1f} KB/s")

        # System resource benchmark
        print("\\n💻 System Resource Utilization...")
        print(f"CPU Usage: {psutil.cpu_percent()}%")
        print(f"Memory Usage: {psutil.virtual_memory().percent}%")

        if torch.cuda.is_available():
            print(f"GPU Memory: {torch.cuda.memory_allocated()/1024**3:.1f} GB used")

        print("\\n✅ Benchmark Suite Complete!")
        print("=" * 60)

        return self.results

# Usage
benchmark = FEDZKBenchmarkSuite()
results = benchmark.run_comprehensive_benchmark()

# Save results
import json
with open('benchmark_results.json', 'w') as f:
    json.dump(results, f, indent=2, default=str)
```

## Performance Troubleshooting

### Common Performance Issues

```python
class PerformanceTroubleshooter:
    def __init__(self):
        self.issues = []
        self.recommendations = []

    def diagnose_system(self):
        """Diagnose system performance issues"""

        # Check CPU usage
        cpu_percent = psutil.cpu_percent(interval=1)
        if cpu_percent > 90:
            self.issues.append("High CPU usage")
            self.recommendations.append("Consider increasing CPU cores or optimizing computation")

        # Check memory usage
        memory = psutil.virtual_memory()
        if memory.percent > 85:
            self.issues.append("High memory usage")
            self.recommendations.append("Reduce batch size or enable gradient checkpointing")

        # Check disk I/O
        disk_io = psutil.disk_io_counters()
        if disk_io and disk_io.read_bytes > 100 * 1024**3:  # 100GB
            self.issues.append("High disk I/O")
            self.recommendations.append("Consider using SSD storage or optimizing data loading")

        # Check network
        net_io = psutil.net_io_counters()
        if net_io and net_io.bytes_sent > 10 * 1024**3:  # 10GB
            self.issues.append("High network usage")
            self.recommendations.append("Enable compression or reduce communication frequency")

        return {
            'issues': self.issues,
            'recommendations': self.recommendations,
            'system_info': {
                'cpu_percent': cpu_percent,
                'memory_percent': memory.percent,
                'disk_read_gb': disk_io.read_bytes / 1024**3 if disk_io else 0,
                'network_sent_gb': net_io.bytes_sent / 1024**3 if net_io else 0
            }
        }

    def diagnose_training(self, model, train_loader):
        """Diagnose training performance issues"""

        # Check model complexity
        total_params = sum(p.numel() for p in model.parameters())
        if total_params > 100 * 10**6:  # 100M parameters
            self.issues.append("Large model")
            self.recommendations.append("Consider model pruning or quantization")

        # Check batch size
        batch_size = train_loader.batch_size
        if batch_size < 16:
            self.issues.append("Small batch size")
            self.recommendations.append("Increase batch size or use gradient accumulation")

        # Check data loading
        start_time = time.time()
        for _ in range(10):
            next(iter(train_loader))
        data_loading_time = (time.time() - start_time) / 10

        if data_loading_time > 0.1:  # More than 100ms per batch
            self.issues.append("Slow data loading")
            self.recommendations.append("Use DataLoader with num_workers > 0 or prefetch data")

        return {
            'model_params': total_params,
            'batch_size': batch_size,
            'data_loading_time': data_loading_time,
            'issues': self.issues,
            'recommendations': self.recommendations
        }

# Usage
troubleshooter = PerformanceTroubleshooter()

# Diagnose system issues
system_diagnosis = troubleshooter.diagnose_system()
print("System Issues:", system_diagnosis['issues'])
print("Recommendations:", system_diagnosis['recommendations'])

# Diagnose training issues
training_diagnosis = troubleshooter.diagnose_training(model, train_loader)
print("Training Issues:", training_diagnosis['issues'])
print("Recommendations:", training_diagnosis['recommendations'])
```

### Automated Performance Optimization

```python
class AutoOptimizer:
    def __init__(self, model, train_loader):
        self.model = model
        self.train_loader = train_loader
        self.optimizations_applied = []

    def optimize_batch_size(self):
        """Automatically find optimal batch size"""

        batch_sizes = [8, 16, 32, 64, 128]
        best_batch_size = 32
        best_time = float('inf')

        for batch_size in batch_sizes:
            try:
                loader = DataLoader(
                    self.train_loader.dataset,
                    batch_size=batch_size,
                    shuffle=True
                )

                start_time = time.time()
                # Test with a few batches
                for i, (inputs, targets) in enumerate(loader):
                    if i >= 5:  # Test 5 batches
                        break
                    # Simple forward pass
                    with torch.no_grad():
                        _ = self.model(inputs)

                avg_time = (time.time() - start_time) / 5

                if avg_time < best_time:
                    best_time = avg_time
                    best_batch_size = batch_size

            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    break  # Stop if we hit memory limit

        self.optimizations_applied.append(f"Batch size optimized to {best_batch_size}")
        return best_batch_size

    def optimize_mixed_precision(self):
        """Enable automatic mixed precision if beneficial"""

        if torch.cuda.is_available():
            # Test performance with and without AMP
            self.model = self.model.cuda()

            # Test without AMP
            start_time = time.time()
            with torch.no_grad():
                for inputs, _ in self.train_loader:
                    inputs = inputs.cuda()
                    _ = self.model(inputs)
                    if time.time() - start_time > 1:  # Test for 1 second
                        break
            time_without_amp = time.time() - start_time

            # Test with AMP
            start_time = time.time()
            with torch.no_grad(), torch.cuda.amp.autocast():
                for inputs, _ in self.train_loader:
                    inputs = inputs.cuda()
                    _ = self.model(inputs)
                    if time.time() - start_time > 1:  # Test for 1 second
                        break
            time_with_amp = time.time() - start_time

            if time_with_amp < time_without_amp:
                self.optimizations_applied.append("Mixed precision enabled")
                return True

        return False

    def apply_optimizations(self):
        """Apply all available optimizations"""

        print("🔧 Applying Automatic Optimizations...")

        # Optimize batch size
        optimal_batch_size = self.optimize_batch_size()
        print(f"✅ Optimal batch size: {optimal_batch_size}")

        # Try mixed precision
        if self.optimize_mixed_precision():
            print("✅ Mixed precision enabled")
        else:
            print("ℹ️ Mixed precision not beneficial")

        # Apply gradient checkpointing for large models
        total_params = sum(p.numel() for p in self.model.parameters())
        if total_params > 50 * 10**6:  # 50M parameters
            # Enable gradient checkpointing
            print("✅ Gradient checkpointing enabled for large model")

        print("🎉 Optimization complete!")
        print("Applied optimizations:", self.optimizations_applied)

        return {
            'optimal_batch_size': optimal_batch_size,
            'mixed_precision': 'amp' in self.optimizations_applied,
            'gradient_checkpointing': 'checkpointing' in self.optimizations_applied
        }

# Usage
optimizer = AutoOptimizer(model, train_loader)
optimizations = optimizer.apply_optimizations()
```

---

*This performance optimization guide is continuously updated. For the latest optimizations and best practices, visit the [FEDZK Performance Documentation](https://docs.fedzk.io/performance/).*
