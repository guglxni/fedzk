# FedZK Production Deployment Guide

## 🎯 Production Readiness Checklist

### ✅ Core Implementation
- [x] **Real Zero-Knowledge Proofs**: Groth16 circuits with Circom/SNARKjs (no simulation)
- [x] **Production CLI**: Real training and proof generation commands
- [x] **Enhanced MPC Server**: Security, rate limiting, monitoring, health checks
- [x] **Advanced LocalTrainer**: Multiple models, optimizers, data formats
- [x] **Configuration Management**: Type-safe, environment-based configuration
- [x] **Comprehensive Logging**: JSON logging for production, human-readable for dev

### ✅ Security Features
- [x] **API Key Authentication**: SHA-256 hashed keys with minimum length enforcement
- [x] **Rate Limiting**: Configurable request limits per IP address
- [x] **Input Validation**: Comprehensive validation and sanitization
- [x] **Security Headers**: CORS, trusted hosts, content security
- [x] **Failed Attempt Tracking**: Automatic IP blocking for brute force protection
- [x] **Environment Separation**: Different security policies for dev/staging/production

### ✅ Monitoring & Observability
- [x] **Health Checks**: `/health`, `/ready` endpoints for load balancers
- [x] **Metrics Collection**: Request counts, response times, error rates
- [x] **Performance Monitoring**: Slow request detection and alerting
- [x] **Structured Logging**: JSON logs with correlation IDs and metadata
- [x] **Circuit File Validation**: Automatic detection of missing ZK components

### ✅ Error Handling
- [x] **Graceful Degradation**: Proper fallbacks when ZK toolchain unavailable
- [x] **Detailed Error Messages**: User-friendly errors with technical details in logs
- [x] **HTTP Status Codes**: Proper 4xx/5xx responses for different error types
- [x] **Exception Handling**: Comprehensive try/catch with proper cleanup
- [x] **Timeout Management**: Request timeouts to prevent resource exhaustion

## 🚀 Deployment Instructions

### 1. Environment Setup

```bash
# Install Python dependencies
pip install -r requirements.txt

# Set up ZK toolchain
./scripts/setup_zk.sh

# Generate circuit files
cd src/fedzk/zk/circuits
circom model_update.circom --r1cs --wasm --sym
snarkjs groth16 setup model_update.r1cs powersoftau28_hez_final_08.ptau proving_key.zkey
```

### 2. Configuration

Create a `.env` file:

```bash
# Environment
ENVIRONMENT=production
DEBUG=false
LOG_LEVEL=info

# Server
HOST=0.0.0.0
PORT=8000
WORKERS=4

# Security (REQUIRED for production)
MPC_API_KEYS=your-very-long-secure-api-key-32-chars-minimum,another-secure-key-32-chars-minimum
ALLOWED_ORIGINS=https://your-domain.com,https://api.your-domain.com
TRUSTED_HOSTS=your-domain.com,api.your-domain.com
RATE_LIMIT_REQUESTS=100
RATE_LIMIT_WINDOW=60
MAX_FAILED_ATTEMPTS=5

# ZK Circuit Paths (auto-detected if not specified)
ZK_CIRCUITS_DIR=/opt/fedzk/circuits
MPC_STD_WASM_PATH=/opt/fedzk/circuits/build/model_update.wasm
MPC_STD_ZKEY_PATH=/opt/fedzk/circuits/proving_key.zkey
MPC_STD_VER_KEY_PATH=/opt/fedzk/circuits/verification_key.json

# Performance
ENABLE_GPU=true
MAX_GRADIENT_SIZE=50000
REQUEST_TIMEOUT=300

# Monitoring
ENABLE_METRICS=true
METRICS_PORT=9090
HEALTH_CHECK_INTERVAL=30
```

### 3. Docker Deployment

```bash
# Build production image
docker build -f build/docker/Dockerfile -t fedzk:latest .

# Run with environment variables
docker run -d \
  --name fedzk-server \
  -p 8000:8000 \
  -p 9090:9090 \
  --env-file .env \
  -v /opt/fedzk/circuits:/opt/fedzk/circuits:ro \
  fedzk:latest
```

### 4. Kubernetes Deployment

```bash
# Apply configurations
kubectl apply -f build/k8s/configmap.yaml
kubectl apply -f build/k8s/deployment.yaml

# Create secrets for API keys
kubectl create secret generic fedzk-secrets \
  --from-literal=MPC_API_KEYS="key1,key2"
```

### 5. Load Balancer Health Checks

Configure your load balancer to use:
- **Health Check**: `GET /health`
- **Readiness Check**: `GET /ready`
- **Metrics**: `GET /metrics` (requires authentication)

## 📊 Monitoring Setup

### Prometheus Configuration

```yaml
scrape_configs:
  - job_name: 'fedzk'
    static_configs:
      - targets: ['fedzk:9090']
    metrics_path: '/metrics'
    scrape_interval: 30s
    basic_auth:
      username: admin
      password: your-metrics-password
```

### Key Metrics to Monitor

- **Request Rate**: `fedzk_requests_total`
- **Response Time**: `fedzk_request_duration_seconds`
- **Error Rate**: `fedzk_errors_total`
- **Active Connections**: `fedzk_active_connections`
- **ZK Proof Generation Time**: `fedzk_proof_generation_duration`
- **Circuit File Status**: `fedzk_circuit_files_available`

## 🔧 Performance Tuning

### CPU Optimization
```bash
# Set optimal worker count (usually 2x CPU cores)
WORKERS=8

# Enable CPU affinity
taskset -c 0-7 python -m uvicorn fedzk.mpc.server:app
```

### GPU Acceleration
```bash
# Enable GPU for ZK proof generation
ENABLE_GPU=true

# Verify CUDA availability
nvidia-smi
```

### Memory Optimization
```bash
# Limit memory usage
ulimit -m 4194304  # 4GB limit

# Set garbage collection thresholds
PYTHONOPTIMIZE=1
```

## 🛡️ Security Hardening

### 1. API Key Management
- Use cryptographically secure random keys (minimum 32 characters)
- Rotate keys regularly (monthly recommended)
- Store keys in secure key management systems (HashiCorp Vault, AWS KMS)
- Never log API keys in application logs

### 2. Network Security
```bash
# Firewall rules
iptables -A INPUT -p tcp --dport 8000 -s trusted_ip_range -j ACCEPT
iptables -A INPUT -p tcp --dport 8000 -j DROP

# TLS Configuration (use reverse proxy like nginx)
ssl_protocols TLSv1.2 TLSv1.3;
ssl_ciphers ECDHE-RSA-AES256-GCM-SHA512:DHE-RSA-AES256-GCM-SHA512;
```

### 3. Application Security
- Enable request size limits
- Implement rate limiting per user/API key
- Use secure headers (HSTS, CSP, X-Frame-Options)
- Regular security audits and dependency updates

## 📝 Operational Procedures

### Daily Operations
1. Check health endpoint status
2. Monitor error rates and response times
3. Verify ZK circuit file integrity
4. Review security logs for unusual activity

### Weekly Operations
1. Rotate log files
2. Update dependencies
3. Performance baseline review
4. Security scan execution

### Monthly Operations
1. API key rotation
2. Security configuration review
3. Capacity planning assessment
4. Disaster recovery testing

## 🚨 Troubleshooting

### Common Issues

#### ZK Toolchain Not Available
```bash
# Symptoms: 503 errors on proof generation
# Solution: Install and configure ZK toolchain
./scripts/setup_zk.sh
export PATH=$PATH:/usr/local/bin/circom:/usr/local/bin/snarkjs
```

#### High Memory Usage
```bash
# Symptoms: OOM kills, slow response times
# Solution: Optimize batch sizes and enable cleanup
MAX_GRADIENT_SIZE=10000
REQUEST_TIMEOUT=60
```

#### Rate Limiting Errors
```bash
# Symptoms: 429 status codes
# Solution: Adjust rate limits or investigate traffic patterns
RATE_LIMIT_REQUESTS=1000
RATE_LIMIT_WINDOW=60
```

## 📈 Scaling Guidelines

### Horizontal Scaling
- Deploy multiple instances behind load balancer
- Use Redis for shared rate limiting state
- Implement circuit file sharing (NFS, S3)

### Vertical Scaling
- Increase memory for larger gradient processing
- Add GPU nodes for faster proof generation
- Optimize CPU cores based on workload

### Auto-scaling Triggers
- CPU usage > 70% for 5 minutes
- Memory usage > 80% for 3 minutes
- Response time > 5 seconds average
- Error rate > 1% over 10 minutes

## ✅ Production Validation

Run this checklist before going live:

```bash
# 1. Health checks responding
curl http://localhost:8000/health

# 2. Authentication working
curl -H "x-api-key: your-key" http://localhost:8000/metrics

# 3. ZK circuits available
ls src/fedzk/zk/circuits/*.{wasm,zkey,json}

# 4. Performance baseline
ab -n 100 -c 10 -H "x-api-key: your-key" http://localhost:8000/health

# 5. Security headers present
curl -I http://localhost:8000/health

# 6. Error handling
curl -X POST http://localhost:8000/generate_proof  # Should return 401

# 7. Monitoring working
curl http://localhost:9090/metrics
```

## 🎯 Production Deployment Complete

Your FedZK instance is now production-ready with:
- ✅ Real zero-knowledge proof generation
- ✅ Enterprise-grade security and monitoring
- ✅ Scalable architecture and deployment
- ✅ Comprehensive operational procedures
- ✅ Performance optimization and tuning

For support and advanced configuration, refer to the full documentation or contact the FedZK team.
