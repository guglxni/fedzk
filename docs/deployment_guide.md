# Deployment Guide

This guide provides detailed instructions for deploying FedZK in various environments, from development to production.

## Deployment Architectures

FedZK supports several deployment architectures depending on your use case:

### 1. Single-Node Development

Suitable for development and testing:

```
┌────────────────────────────────────┐
│                                    │
│  ┌──────────┐      ┌────────────┐  │
│  │          │      │            │  │
│  │  Client  │◄────►│ Coordinator│  │
│  │          │      │            │  │
│  └──────────┘      └────────────┘  │
│                                    │
└────────────────────────────────────┘
```

### 2. Distributed Deployment

Suitable for production with multiple clients:

```
┌──────────┐     ┌──────────┐     ┌──────────┐
│          │     │          │     │          │
│  Client  │     │  Client  │     │  Client  │
│          │     │          │     │          │
└────┬─────┘     └────┬─────┘     └────┬─────┘
     │                │                │
     └────────────────┼────────────────┘
                      │
                      ▼
             ┌─────────────────┐
             │                 │
             │   Coordinator   │
             │                 │
             └─────────────────┘
```

### 3. MPC-Enabled Deployment

Suitable for enhanced privacy requirements:

```
┌──────────┐     ┌──────────┐     ┌──────────┐
│          │     │          │     │          │
│  Client  │     │  Client  │     │  Client  │
│          │     │          │     │          │
└────┬─────┘     └────┬─────┘     └────┬─────┘
     │                │                │
     └────────────────┼────────────────┘
                      │
                      ▼
             ┌─────────────────┐
             │                 │
             │   Coordinator   │
             │                 │
             └────────┬────────┘
                      │
                      ▼
             ┌─────────────────┐
             │                 │
             │    MPC Server   │
             │                 │
             └─────────────────┘
```

## Deployment Steps

### Prerequisites

- Python 3.8 or higher
- pip (Python package installer)
- Git
- Docker (optional, for containerized deployment)

### Development Deployment

1. **Clone the repository**:
   ```bash
   git clone https://github.com/guglxni/fedzk.git
   cd fedzk
   ```

2. **Set up a virtual environment**:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install the package**:
   ```bash
   pip install -e .
   ```

4. **Run the development server**:
   ```bash
   python -m fedzk.coordinator --host 127.0.0.1 --port 8000 --debug
   ```

5. **Test with a client**:
   ```bash
   python -m fedzk.client --coordinator-url http://127.0.0.1:8000
   ```

### Production Deployment

#### Coordinator Deployment

1. **Install production dependencies**:
   ```bash
   pip install "fedzk[all]" gunicorn
   ```

2. **Create a configuration file** (`coordinator_config.json`):
   ```json
   {
     "host": "0.0.0.0",
     "port": 8000,
     "min_clients": 5,
     "aggregation_threshold": 3,
     "round_timeout": 600,
     "max_rounds": 100,
     "log_level": "info"
   }
   ```

3. **Start the coordinator with Gunicorn**:
   ```bash
   gunicorn -w 4 -k uvicorn.workers.UvicornWorker fedzk.coordinator.api:create_app --bind 0.0.0.0:8000
   ```

4. **Set up a reverse proxy** (Nginx example):
   ```nginx
   server {
       listen 80;
       server_name coordinator.example.com;
       
       location / {
           proxy_pass http://127.0.0.1:8000;
           proxy_set_header Host $host;
           proxy_set_header X-Real-IP $remote_addr;
       }
   }
   ```

#### MPC Server Deployment

1. **Install MPC dependencies**:
   ```bash
   pip install "fedzk[all]" gunicorn
   ```

2. **Create a configuration file** (`mpc_config.json`):
   ```json
   {
     "host": "0.0.0.0",
     "port": 8001,
     "privacy_budget": 0.1,
     "encryption_key": "secure_random_key_here",
     "mpc_protocol": "semi_honest",
     "log_level": "info"
   }
   ```

3. **Start the MPC server**:
   ```bash
   python -m fedzk.mpc.server --config mpc_config.json
   ```

#### Client Deployment

1. **Install client dependencies**:
   ```bash
   pip install "fedzk[all]"
   ```

2. **Create a configuration file** (`client_config.json`):
   ```json
   {
     "coordinator_url": "https://coordinator.example.com",
     "client_id": "client001",
     "data_path": "/path/to/data",
     "model_config": {
       "architecture": "mlp",
       "layers": [784, 128, 10]
     },
     "training_config": {
       "batch_size": 32,
       "epochs": 5,
       "learning_rate": 0.01
     },
     "log_level": "info"
   }
   ```

3. **Start the client**:
   ```bash
   python -m fedzk.client --config client_config.json
   ```

### Docker Deployment

1. **Build the Docker image**:
   ```bash
   docker build -t fedzk:latest .
   ```

2. **Run the coordinator container**:
   ```bash
   docker run -d --name fedzk-coordinator -p 8000:8000 \
     -v /path/to/coordinator_config.json:/app/config.json \
     fedzk:latest coordinator --config /app/config.json
   ```

3. **Run the MPC server container**:
   ```bash
   docker run -d --name fedzk-mpc -p 8001:8001 \
     -v /path/to/mpc_config.json:/app/config.json \
     fedzk:latest mpc --config /app/config.json
   ```

4. **Run client containers**:
   ```bash
   docker run -d --name fedzk-client1 \
     -v /path/to/client_config.json:/app/config.json \
     -v /path/to/data:/app/data \
     fedzk:latest client --config /app/config.json
   ```

### Docker Compose Deployment

Create a `docker-compose.yml` file:

```yaml
version: '3'

services:
  coordinator:
    image: fedzk:latest
    command: coordinator --config /app/config.json
    ports:
      - "8000:8000"
    volumes:
      - ./configs/coordinator_config.json:/app/config.json
    restart: unless-stopped

  mpc-server:
    image: fedzk:latest
    command: mpc --config /app/config.json
    ports:
      - "8001:8001"
    volumes:
      - ./configs/mpc_config.json:/app/config.json
    restart: unless-stopped
    depends_on:
      - coordinator

  client1:
    image: fedzk:latest
    command: client --config /app/config.json
    volumes:
      - ./configs/client1_config.json:/app/config.json
      - ./data/client1:/app/data
    restart: unless-stopped
    depends_on:
      - coordinator

  client2:
    image: fedzk:latest
    command: client --config /app/config.json
    volumes:
      - ./configs/client2_config.json:/app/config.json
      - ./data/client2:/app/data
    restart: unless-stopped
    depends_on:
      - coordinator
```

Run with:

```bash
docker-compose up -d
```

## Security Considerations

### Network Security

1. **Use TLS/SSL**:
   - Generate SSL certificates
   - Configure the coordinator to use HTTPS
   - Update client configurations to use HTTPS URLs

2. **Firewall Configuration**:
   - Restrict access to coordinator and MPC servers
   - Only allow necessary ports (8000, 8001)
   - Use IP allowlisting for client connections

### Authentication

1. **API Key Authentication**:
   ```json
   {
     "api_key": "your_secure_api_key_here"
   }
   ```

2. **JWT Authentication**:
   ```python
   # Coordinator side
   from fedzk.security import JWTAuth
   
   auth = JWTAuth(secret_key="your_secret_key")
   coordinator = Aggregator(auth=auth)
   ```

   ```python
   # Client side
   from fedzk.security import JWTAuth
   
   auth = JWTAuth(secret_key="your_secret_key")
   client = Client(auth=auth)
   ```

### Data Security

1. **Data Encryption**:
   ```python
   from fedzk.security import DataEncryptor
   
   encryptor = DataEncryptor(key="your_encryption_key")
   encrypted_updates = encryptor.encrypt(updates)
   ```

2. **Secure Storage**:
   - Use encrypted volumes for storing model data
   - Implement secure deletion policies

## Monitoring and Maintenance

### Logging

Configure logging for all components:

```json
{
  "logging": {
    "level": "info",
    "file": "/var/log/fedzk/coordinator.log",
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "rotate": true,
    "max_size": 10485760
  }
}
```

### Health Checks

Implement health checks for all services:

```bash
# Coordinator health check
curl -X GET http://coordinator.example.com/health

# MPC server health check
curl -X GET http://mpc.example.com/health
```

### Backup and Recovery

1. **Regular Backups**:
   ```bash
   # Back up configuration
   tar -czvf fedzk_config_backup.tar.gz configs/
   
   # Back up models
   tar -czvf fedzk_models_backup.tar.gz models/
   ```

2. **Restore from Backup**:
   ```bash
   # Restore configuration
   tar -xzvf fedzk_config_backup.tar.gz
   
   # Restore models
   tar -xzvf fedzk_models_backup.tar.gz
   ```

## Scaling Considerations

### Horizontal Scaling

1. **Coordinator Load Balancing**:
   - Deploy multiple coordinator instances
   - Use a load balancer (Nginx, HAProxy)
   - Implement sticky sessions for client connections

2. **Database Scaling**:
   - Use a distributed database for larger deployments
   - Configure the coordinator to use the database backend

### Vertical Scaling

1. **Resource Allocation**:
   - Allocate more CPU/memory to coordinator and MPC servers
   - Optimize client resource usage based on model size

2. **Performance Tuning**:
   - Refer to the [Performance Tuning](performance.md) guide for specific recommendations

## Troubleshooting Deployment Issues

For common deployment issues, refer to the [Troubleshooting Guide](troubleshooting.md).

## Examples

Check the [examples](../examples) directory for:

- [Kubernetes deployment configurations](../examples/kubernetes)
- [AWS deployment scripts](../examples/aws)
- [Azure deployment templates](../examples/azure) 