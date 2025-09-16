# 🔧 FEDZK Troubleshooting Guide

This comprehensive guide helps you diagnose and resolve common issues with FEDZK deployments and operations.

## Table of Contents

1. [Quick Diagnosis](#quick-diagnosis)
2. [Installation Issues](#installation-issues)
3. [Startup Problems](#startup-problems)
4. [Network Connectivity](#network-connectivity)
5. [Performance Issues](#performance-issues)
6. [ZK Proof Generation](#zk-proof-generation)
7. [Federation Management](#federation-management)
8. [Security & Privacy](#security--privacy)
9. [Database Issues](#database-issues)
10. [Logging & Monitoring](#logging--monitoring)
11. [Advanced Debugging](#advanced-debugging)

## Quick Diagnosis

### Health Check Script

```bash
#!/bin/bash
# FEDZK Health Check Script

echo "🔍 FEDZK Health Check"
echo "===================="

# Check if FEDZK is running
if pgrep -f "fedzk" > /dev/null; then
    echo "✅ FEDZK process is running"
else
    echo "❌ FEDZK process is not running"
    exit 1
fi

# Check API health
if curl -f -s http://localhost:8000/health > /dev/null; then
    echo "✅ API health check passed"
else
    echo "❌ API health check failed"
fi

# Check database connectivity
if python3 -c "import fedzk; fedzk.check_db_connection()"; then
    echo "✅ Database connection OK"
else
    echo "❌ Database connection failed"
fi

# Check ZK toolchain
if python3 -c "import fedzk; fedzk.check_zk_toolchain()"; then
    echo "✅ ZK toolchain OK"
else
    echo "❌ ZK toolchain issues"
fi

echo "===================="
echo "Health check complete"
```

Run the health check:
```bash
chmod +x health_check.sh
./health_check.sh
```

### Diagnostic Information Collection

```python
# Collect diagnostic information
import fedzk
from fedzk.core import DiagnosticCollector

collector = DiagnosticCollector()
report = collector.collect_all_diagnostics()

# Save diagnostic report
with open('fedzk_diagnostics.json', 'w') as f:
    json.dump(report, f, indent=2)

print("Diagnostic report saved to fedzk_diagnostics.json")
```

## Installation Issues

### Python Version Compatibility

**Problem:** Import errors or compatibility issues

**Solution:**
```bash
# Check Python version
python3 --version  # Should be 3.8+

# Create virtual environment
python3 -m venv fedzk_env
source fedzk_env/bin/activate

# Install with compatible pip
pip install --upgrade pip
pip install fedzk
```

### Dependency Conflicts

**Problem:** Package conflicts during installation

**Solution:**
```bash
# Check for conflicts
pip check

# Install in isolated environment
pip install --isolated fedzk

# Or use pip-tools for dependency resolution
pip install pip-tools
pip-compile requirements.in
pip-sync
```

### Permission Errors

**Problem:** Installation fails due to permissions

**Solution:**
```bash
# Install for user only
pip install --user fedzk

# Or use sudo (not recommended)
sudo pip install fedzk

# Create virtual environment in user directory
python3 -m venv ~/fedzk_env
source ~/fedzk_env/bin/activate
pip install fedzk
```

## Startup Problems

### Service Won't Start

**Problem:** FEDZK service fails to start

**Diagnosis:**
```bash
# Check system logs
sudo journalctl -u fedzk -n 50

# Check application logs
tail -f /var/log/fedzk/app.log

# Check port availability
netstat -tlnp | grep :8000
```

**Common Solutions:**
```python
# Check configuration
from fedzk.config import Config
config = Config()
config.validate()

# Test database connection
from fedzk.database import DatabaseManager
db = DatabaseManager()
db.test_connection()

# Check required directories
import os
required_dirs = ['/var/log/fedzk', '/var/lib/fedzk', '/etc/fedzk']
for dir_path in required_dirs:
    if not os.path.exists(dir_path):
        os.makedirs(dir_path, exist_ok=True)
```

### Configuration Errors

**Problem:** Invalid configuration causing startup failure

**Solution:**
```python
# Validate configuration
from fedzk.config import ConfigValidator
validator = ConfigValidator()

config_file = '/etc/fedzk/config.yaml'
if validator.validate_file(config_file):
    print("✅ Configuration is valid")
else:
    print("❌ Configuration errors:")
    for error in validator.errors:
        print(f"  - {error}")

# Generate default configuration
from fedzk.config import ConfigGenerator
generator = ConfigGenerator()
default_config = generator.generate_default()
print("Default configuration:")
print(yaml.dump(default_config))
```

### Database Connection Issues

**Problem:** Cannot connect to database on startup

**Diagnosis:**
```bash
# Test database connectivity
psql -h localhost -U fedzk -d fedzk -c "SELECT version();"

# Check database logs
sudo tail -f /var/log/postgresql/postgresql-*.log

# Test with Python
python3 -c "
import psycopg2
try:
    conn = psycopg2.connect('postgresql://fedzk:password@localhost:5432/fedzk')
    print('✅ Database connection successful')
    conn.close()
except Exception as e:
    print(f'❌ Database connection failed: {e}')
"
```

**Solutions:**
```sql
-- Create database and user
CREATE DATABASE fedzk;
CREATE USER fedzk WITH PASSWORD 'secure_password';
GRANT ALL PRIVILEGES ON DATABASE fedzk TO fedzk;

-- Test connection
\c fedzk
SELECT current_database(), current_user;
```

## Network Connectivity

### Federation Communication Issues

**Problem:** Participants cannot join or communicate with federation

**Diagnosis:**
```bash
# Test network connectivity
curl -v http://federation-host:8000/health

# Check firewall rules
sudo ufw status
sudo iptables -L

# Test with federation client
python3 -c "
from fedzk.client import FederationClient
client = FederationClient('http://federation-host:8000')
try:
    health = client.health_check()
    print(f'✅ Federation health: {health}')
except Exception as e:
    print(f'❌ Federation connection failed: {e}')
"
```

**Solutions:**
```bash
# Configure firewall
sudo ufw allow 8000/tcp
sudo ufw allow 22/tcp  # SSH access

# Test with different network tools
telnet federation-host 8000
nc -zv federation-host 8000

# Check DNS resolution
nslookup federation-host
dig federation-host
```

### SSL/TLS Certificate Issues

**Problem:** HTTPS connections failing due to certificate problems

**Diagnosis:**
```bash
# Test SSL connection
openssl s_client -connect federation-host:443 -servername federation-host

# Check certificate validity
openssl x509 -in /etc/ssl/certs/fedzk.crt -text -noout

# Test with curl
curl -v https://federation-host:8000/health
```

**Solutions:**
```bash
# Generate self-signed certificate
sudo openssl req -x509 -newkey rsa:4096 \
  -keyout /etc/ssl/private/fedzk.key \
  -out /etc/ssl/certs/fedzk.crt \
  -days 365 -nodes \
  -subj "/C=US/ST=State/L=City/O=Organization/CN=federation-host"

# Configure nginx for SSL
sudo tee /etc/nginx/sites-available/fedzk-ssl << EOF
server {
    listen 443 ssl http2;
    server_name federation-host;

    ssl_certificate /etc/ssl/certs/fedzk.crt;
    ssl_certificate_key /etc/ssl/private/fedzk.key;

    location / {
        proxy_pass http://localhost:8000;
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;
    }
}
EOF
```

## Performance Issues

### Slow Training Performance

**Problem:** Federated training is running slowly

**Diagnosis:**
```python
# Monitor training performance
from fedzk.monitoring import PerformanceMonitor
monitor = PerformanceMonitor()

# Get performance metrics
metrics = monitor.get_training_metrics()
print(f"Training throughput: {metrics.throughput} samples/sec")
print(f"Communication latency: {metrics.latency} ms")
print(f"ZK proof generation time: {metrics.zk_proof_time} ms")

# Check resource utilization
import psutil
print(f"CPU usage: {psutil.cpu_percent()}%")
print(f"Memory usage: {psutil.virtual_memory().percent}%")
print(f"Disk I/O: {psutil.disk_io_counters()}")
```

**Solutions:**
```python
# Enable GPU acceleration
federation.enable_gpu_acceleration()

# Optimize batch size
training_config = {
    'batch_size': 64,  # Increase for better GPU utilization
    'gradient_accumulation_steps': 2,
    'mixed_precision': True
}

# Enable parallel processing
federation.set_parallel_processing(
    num_workers=4,
    use_async=True
)

# Optimize network communication
federation.optimize_network(
    compression=True,
    batch_updates=True
)
```

### Memory Issues

**Problem:** Out of memory errors during training

**Solutions:**
```python
# Reduce memory usage
training_config = {
    'batch_size': 16,
    'gradient_checkpointing': True,
    'memory_efficient_attention': True
}

# Enable memory monitoring
from fedzk.monitoring import MemoryMonitor
monitor = MemoryMonitor()

# Set memory limits
monitor.set_memory_limits(
    max_memory_gb=8,
    warning_threshold=0.8
)

# Monitor memory usage
while training_session.is_active:
    memory_stats = monitor.get_memory_stats()
    if memory_stats.usage_percent > 80:
        print("⚠️ High memory usage detected")
        monitor.trigger_memory_optimization()
```

### High Latency Issues

**Problem:** High network latency affecting federation performance

**Diagnosis:**
```bash
# Test network latency
ping -c 5 federation-host

# Test bandwidth
iperf -c federation-host

# Monitor network traffic
sudo nload
sudo iftop -i eth0
```

**Solutions:**
```python
# Configure network optimization
federation.configure_network(
    compression_algorithm='gzip',
    batch_size=1024,
    timeout=30
)

# Enable connection pooling
from fedzk.network import ConnectionPool
pool = ConnectionPool(
    max_connections=10,
    timeout=30,
    retry_attempts=3
)

# Use CDN for static assets
# Configure load balancer for multiple federation hosts
```

## ZK Proof Generation

### ZK Toolchain Issues

**Problem:** ZK proof generation failing

**Diagnosis:**
```bash
# Check ZK toolchain installation
which circom
which snarkjs
circom --version
snarkjs --version

# Test circuit compilation
circom test_circuit.circom --r1cs --wasm

# Check for compilation errors
cat compilation_errors.log
```

**Solutions:**
```bash
# Install ZK toolchain
curl -L https://github.com/iden3/circom/releases/latest/download/circom-linux-amd64 -o /usr/local/bin/circom
chmod +x /usr/local/bin/circom

npm install -g snarkjs

# Test installation
circom --help
snarkjs --help

# Validate circuit files
python3 -c "
from fedzk.zk import CircuitValidator
validator = CircuitValidator()
if validator.validate_circuit('model_update.circom'):
    print('✅ Circuit is valid')
else:
    print('❌ Circuit validation failed')
    print(validator.get_errors())
"
```

### Proof Generation Timeout

**Problem:** ZK proof generation taking too long or timing out

**Solutions:**
```python
# Configure proof generation parameters
zk_config = {
    'proof_system': 'groth16',
    'curve': 'bn128',
    'timeout_seconds': 300,
    'max_proof_size': '1GB',
    'parallel_proofs': True,
    'gpu_acceleration': True
}

federation.configure_zk_proofs(zk_config)

# Enable proof caching
from fedzk.zk import ProofCache
cache = ProofCache(max_size=1000, ttl_hours=24)
federation.set_proof_cache(cache)

# Monitor proof generation
monitor = federation.get_proof_monitor()
stats = monitor.get_generation_stats()
print(f"Average proof time: {stats.avg_time} seconds")
print(f"Cache hit rate: {stats.cache_hit_rate}%")
```

## Federation Management

### Participant Join Issues

**Problem:** Participants cannot join federation

**Diagnosis:**
```python
# Check federation status
federation = FederationClient('http://federation-host:8000')
status = federation.get_status()
print(f"Federation status: {status.state}")
print(f"Max participants: {status.max_participants}")
print(f"Current participants: {status.current_participants}")

# Test participant registration
try:
    token = federation.request_join(
        participant_id='test_participant',
        public_key=public_key
    )
    print(f"✅ Join request successful: {token}")
except Exception as e:
    print(f"❌ Join request failed: {e}")
```

**Solutions:**
```python
# Check federation capacity
if federation.is_full():
    print("Federation is at capacity")
    # Consider increasing max_participants or creating new federation

# Validate participant credentials
from fedzk.auth import ParticipantValidator
validator = ParticipantValidator()

if validator.validate_credentials(participant_id, public_key):
    print("✅ Participant credentials valid")
else:
    print("❌ Invalid participant credentials")

# Check network policies
federation.check_network_policies(participant_network)
```

### Model Synchronization Issues

**Problem:** Model updates not synchronizing properly across participants

**Diagnosis:**
```python
# Check model versions
current_version = federation.get_current_model_version()
participant_versions = federation.get_participant_versions()

print(f"Current model version: {current_version}")
for participant, version in participant_versions.items():
    if version != current_version:
        print(f"⚠️ {participant} is out of sync: {version} vs {current_version}")

# Check synchronization status
sync_status = federation.get_sync_status()
print(f"Sync status: {sync_status}")
```

**Solutions:**
```python
# Force model synchronization
federation.force_model_sync()

# Configure synchronization parameters
sync_config = {
    'sync_interval': 60,  # seconds
    'max_sync_delay': 300,  # seconds
    'sync_batch_size': 100,
    'compression': True
}

federation.configure_sync(sync_config)

# Monitor synchronization
monitor = federation.get_sync_monitor()
stats = monitor.get_sync_stats()
print(f"Sync success rate: {stats.success_rate}%")
print(f"Average sync time: {stats.avg_sync_time} seconds")
```

## Security & Privacy

### Encryption Key Issues

**Problem:** Encryption/decryption failing due to key problems

**Diagnosis:**
```bash
# Check key files
ls -la /etc/fedzk/keys/

# Validate key format
openssl rsa -in /etc/fedzk/keys/private.key -check

# Test key functionality
python3 -c "
from fedzk.crypto import KeyManager
km = KeyManager()
try:
    km.test_keys()
    print('✅ Keys are valid')
except Exception as e:
    print(f'❌ Key validation failed: {e}')
"
```

**Solutions:**
```python
# Generate new keys
from fedzk.crypto import KeyGenerator
generator = KeyGenerator()

# Generate key pair
private_key, public_key = generator.generate_rsa_keypair(bits=4096)

# Save keys securely
generator.save_private_key(private_key, '/etc/fedzk/keys/private.key')
generator.save_public_key(public_key, '/etc/fedzk/keys/public.key')

# Set proper permissions
import os
os.chmod('/etc/fedzk/keys/private.key', 0o600)
os.chmod('/etc/fedzk/keys/public.key', 0o644)

# Update key references in configuration
config.update_keys(
    private_key_path='/etc/fedzk/keys/private.key',
    public_key_path='/etc/fedzk/keys/public.key'
)
```

### Privacy Parameter Configuration

**Problem:** Differential privacy parameters causing issues

**Diagnosis:**
```python
# Test privacy parameters
from fedzk.privacy import PrivacyTester
tester = PrivacyTester()

test_data = [1.0, 2.0, 3.0, 4.0, 5.0]
privacy_config = {
    'epsilon': 1.0,
    'delta': 1e-5,
    'noise_mechanism': 'gaussian'
}

try:
    privatized_data = tester.apply_privacy(test_data, privacy_config)
    print(f"✅ Privacy mechanism working: {privatized_data[:3]}...")
except Exception as e:
    print(f"❌ Privacy mechanism failed: {e}")
```

**Solutions:**
```python
# Adjust privacy parameters
privacy_config = {
    'epsilon': 2.0,  # Higher epsilon = less privacy but more utility
    'delta': 1e-6,   # Lower delta = stronger privacy guarantee
    'noise_mechanism': 'laplace',  # Alternative to gaussian
    'clipping_norm': 1.0,
    'adaptive_noise': True
}

# Test different configurations
for epsilon in [0.5, 1.0, 2.0, 5.0]:
    config = privacy_config.copy()
    config['epsilon'] = epsilon
    accuracy = tester.test_accuracy_impact(test_data, config)
    privacy_loss = tester.test_privacy_loss(test_data, config)
    print(f"Epsilon {epsilon}: Accuracy={accuracy:.3f}, Privacy Loss={privacy_loss:.3f}")
```

## Database Issues

### Connection Pool Exhaustion

**Problem:** Database connection pool exhausted

**Diagnosis:**
```sql
-- Check active connections
SELECT count(*) as active_connections
FROM pg_stat_activity
WHERE datname = 'fedzk';

-- Check connection limits
SHOW max_connections;
```

**Solutions:**
```python
# Configure connection pooling
db_config = {
    'pool_size': 20,
    'max_overflow': 30,
    'pool_timeout': 30,
    'pool_recycle': 3600
}

# Update database configuration
from fedzk.database import DatabaseManager
db = DatabaseManager()
db.configure_pool(db_config)

# Monitor connection usage
monitor = db.get_connection_monitor()
stats = monitor.get_connection_stats()
print(f"Active connections: {stats.active}")
print(f"Available connections: {stats.available}")
print(f"Pool utilization: {stats.utilization}%")
```

### Database Migration Issues

**Problem:** Database schema migrations failing

**Solutions:**
```bash
# Check migration status
python3 -m fedzk db status

# Run pending migrations
python3 -m fedzk db upgrade

# Rollback if needed
python3 -m fedzk db downgrade --revision=previous

# Create new migration
python3 -m fedzk db revision --autogenerate -m "Add new table"
```

## Logging & Monitoring

### Log Configuration Issues

**Problem:** Logging not working properly

**Solutions:**
```python
# Configure logging
import logging.config

logging_config = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'detailed': {
            'format': '%(asctime)s %(name)s %(levelname)s %(message)s'
        },
        'json': {
            'class': 'pythonjsonlogger.jsonlogger.JsonFormatter',
            'format': '%(asctime)s %(name)s %(levelname)s %(message)s'
        }
    },
    'handlers': {
        'console': {
            'class': 'logging.StreamHandler',
            'formatter': 'detailed',
            'level': 'INFO'
        },
        'file': {
            'class': 'logging.handlers.RotatingFileHandler',
            'filename': '/var/log/fedzk/app.log',
            'maxBytes': 10485760,  # 10MB
            'backupCount': 5,
            'formatter': 'json',
            'level': 'DEBUG'
        }
    },
    'root': {
        'level': 'INFO',
        'handlers': ['console', 'file']
    }
}

logging.config.dictConfig(logging_config)
```

### Monitoring Setup Issues

**Problem:** Monitoring and metrics not working

**Solutions:**
```python
# Configure Prometheus metrics
from fedzk.monitoring import MetricsCollector
metrics = MetricsCollector()

# Register custom metrics
metrics.register_counter('federation_joins_total', 'Total federation joins')
metrics.register_histogram('training_duration_seconds', 'Training duration')
metrics.register_gauge('active_participants', 'Number of active participants')

# Start metrics server
metrics.start_server(port=8001)

# Configure health checks
from fedzk.monitoring import HealthChecker
health = HealthChecker()

@health.register_check('database')
def check_database():
    try:
        db.test_connection()
        return True, "Database OK"
    except Exception as e:
        return False, f"Database error: {e}"

@health.register_check('zk_toolchain')
def check_zk():
    try:
        zk.test_toolchain()
        return True, "ZK toolchain OK"
    except Exception as e:
        return False, f"ZK error: {e}"

# Start health check server
health.start_server(port=8002)
```

## Advanced Debugging

### Core Dump Analysis

```bash
# Enable core dumps
echo "core.%e.%p.%t" | sudo tee /proc/sys/kernel/core_pattern
ulimit -c unlimited

# Analyze core dump
gdb python3 core.python.12345
(gdb) bt  # Show backtrace
(gdb) info threads  # Show threads
(gdb) thread apply all bt  # Show all thread backtraces
```

### Memory Leak Detection

```python
# Use memory profiler
from memory_profiler import profile
import tracemalloc

tracemalloc.start()

@profile
def training_function():
    # Your training code here
    pass

# Run training
training_function()

# Get memory statistics
current, peak = tracemalloc.get_traced_memory()
print(f"Current memory usage: {current / 1024 / 1024:.1f} MB")
print(f"Peak memory usage: {peak / 1024 / 1024:.1f} MB")

# Get top memory consumers
stats = tracemalloc.take_snapshot()
top_stats = stats.statistics('lineno')
for stat in top_stats[:10]:
    print(stat)
```

### Performance Profiling

```python
# Use cProfile for performance profiling
import cProfile
import pstats

def profile_training():
    # Your training code here
    pass

# Profile the function
profiler = cProfile.Profile()
profiler.enable()
profile_training()
profiler.disable()

# Create statistics
stats = pstats.Stats(profiler)
stats.sort_stats('cumulative')
stats.print_stats(20)  # Top 20 functions

# Save to file
stats.dump_stats('training_profile.prof')

# Use snakeviz for visualization
# pip install snakeviz
# snakeviz training_profile.prof
```

### Network Packet Analysis

```bash
# Capture network traffic
sudo tcpdump -i eth0 -w fedzk_traffic.pcap port 8000

# Analyze with Wireshark
# wireshark fedzk_traffic.pcap

# Monitor specific connections
sudo ss -tlnp | grep :8000
sudo netstat -tlnp | grep :8000

# Test with different network tools
curl -v --trace-ascii curl_trace.log http://localhost:8000/health
```

### Distributed Debugging

```python
# Enable distributed logging
from fedzk.logging import DistributedLogger
logger = DistributedLogger(
    logstash_host='logstash.fedzk.internal',
    logstash_port=5044
)

# Add correlation IDs for request tracing
from fedzk.monitoring import RequestTracer
tracer = RequestTracer()

@app.middleware('http')
async def add_request_id(request, call_next):
    request_id = tracer.generate_request_id()
    logger.set_correlation_id(request_id)

    response = await call_next(request)
    response.headers['X-Request-ID'] = request_id
    return response

# Monitor distributed performance
from fedzk.monitoring import DistributedMonitor
monitor = DistributedMonitor()

# Track cross-service calls
with monitor.trace('federation_join'):
    participant = federation.join(participant_data)
```

## Getting Help

### Community Support

- **GitHub Issues**: https://github.com/fedzk/fedzk/issues
- **Community Forum**: https://community.fedzk.io
- **Stack Overflow**: Tag questions with `fedzk`
- **Discord**: https://discord.gg/fedzk

### Professional Support

For enterprise support, contact:
- Email: support@fedzk.io
- Phone: +1 (555) 123-FEDZK
- Enterprise Portal: https://enterprise.fedzk.io

### Diagnostic Information

When requesting help, please provide:

```bash
# System information
uname -a
python3 --version
pip list | grep fedzk

# FEDZK configuration (sanitize sensitive data)
cat /etc/fedzk/config.yaml | grep -v password

# Recent logs
tail -n 100 /var/log/fedzk/app.log

# Diagnostic report
python3 -c "import fedzk; print(fedzk.generate_diagnostic_report())"
```

---

*This troubleshooting guide is continuously updated. For the latest version, visit the [FEDZK Documentation](https://docs.fedzk.io/troubleshooting/).*
