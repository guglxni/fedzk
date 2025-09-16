# FEDzk Real-time Dashboard

This document provides information about the real-time visualization and monitoring dashboard for the FEDzk federated learning system. The dashboard provides a sandbox environment for demonstrating and monitoring the zero-knowledge proof generation and verification process in a federated learning context.

## Features

The FEDzk dashboard provides:

- Real-time visualization of federated learning rounds
- Live monitoring of client training and proof status
- Zero-knowledge proof generation and verification metrics
- System performance and health metrics
- Activity logging and event tracking

## Getting Started

### Prerequisites

- Python 3.8+
- Node.js and npm (for ZK proof generation)
- FEDzk correctly installed and configured

### Running the Dashboard

1. Make the dashboard script executable:
   ```bash
   chmod +x scripts/run_dashboard.sh
   ```

2. Run the dashboard script:
   ```bash
   ./scripts/run_dashboard.sh
   ```

The script will automatically:
1. Create a Python virtual environment if needed
2. Install required dependencies
3. Check if the MPC server is running and start it if necessary
4. Start the dashboard web interface

### Accessing the Dashboard

Once started, the dashboard will be available at:
```
http://localhost:5000
```

## Dashboard Components

### Federated Learning Status

Shows the current state of federated learning, including:
- Current round number and progress
- Number of participating clients
- Global model accuracy

### Zero-Knowledge Proof Metrics

Displays metrics related to ZK proofs:
- Total proofs generated
- Successful verifications
- Generation and verification times
- Proof history chart

### Client Status

Shows the status of each federated learning client:
- Training status
- Dataset size
- Local model accuracy
- Proof generation and verification status

### System Metrics

Provides system-level metrics:
- CPU and memory usage
- Active connections
- Request rates
- Error counts

## Architecture

The dashboard consists of:

1. **Backend**: A Flask server that simulates federated learning clients and integrates with the FEDzk MPC server for real proof generation and verification
2. **Frontend**: A responsive web interface built with Bootstrap, Chart.js, and custom JavaScript

Communication with the MPC server uses the production API endpoints:
- `/generate_proof` - Generates real ZK proofs
- `/verify_proof` - Verifies generated proofs
- `/health` and `/metrics` - For system monitoring

## Use Cases

### Demo Mode

The dashboard includes a simulation mode that demonstrates a complete federated learning workflow:
1. Client training on simulated data
2. Generation of real ZK proofs using the trained model gradients
3. Verification of proofs through the MPC server
4. Aggregation of model updates into a global model

### Monitoring Mode

Used to monitor:
- Proof generation and verification times
- Success rates across multiple clients
- System performance and resource usage

## Troubleshooting

If you encounter issues:

1. **Dashboard won't start**: Check that Python and required packages are installed.
2. **MPC server errors**: Check `mpc_server.log` in the project root directory.
3. **No ZK proofs generated**: Ensure ZK setup has been completed with `scripts/setup_zk.sh`.
4. **Poor performance**: Adjust client count and data sizes in the dashboard configuration.

## Extending the Dashboard

The dashboard can be extended with:
1. Additional visualizations in `dashboard/templates/index.html`
2. New metrics or simulation parameters in `scripts/dashboard.py`
3. Custom simulation logic in the `MockClient` class
