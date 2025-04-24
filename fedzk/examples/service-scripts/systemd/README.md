# FedZK Systemd Services

This directory contains systemd service files for running FedZK services as daemons on Linux systems.

## Services

1. **FedZK MPC Server** (`fedzk-mpc.service`): Runs the Multi-Party Computation server for secure proof generation and verification.
2. **FedZK Coordinator** (`fedzk-coordinator.service`): Runs the Coordinator API server that manages federated learning rounds.

## Installation

### Automatic Installation

Use the provided installation script:

```bash
# Clone the FedZK repository if you haven't already
git clone https://github.com/your-org/fedzk.git
cd fedzk

# Make the installation script executable
chmod +x examples/service-scripts/systemd/install-services.sh

# Run the installation script with sudo
sudo examples/service-scripts/systemd/install-services.sh
```

By default, the services will be installed to run under the `fedzk` user in the `/opt/fedzk` directory.

You can customize the installation with these options:

```bash
sudo examples/service-scripts/systemd/install-services.sh --install-path /custom/path --user custom_user
```

### Manual Installation

If you prefer to install the services manually:

1. Copy the service files to the systemd directory:
   ```bash
   sudo cp examples/service-scripts/systemd/fedzk-*.service /etc/systemd/system/
   ```

2. Edit the service files to set the correct paths and user:
   ```bash
   sudo nano /etc/systemd/system/fedzk-mpc.service
   sudo nano /etc/systemd/system/fedzk-coordinator.service
   ```

3. Reload the systemd daemon:
   ```bash
   sudo systemctl daemon-reload
   ```

## Managing Services

### Starting Services

```bash
sudo systemctl start fedzk-mpc.service
sudo systemctl start fedzk-coordinator.service
```

### Stopping Services

```bash
sudo systemctl stop fedzk-mpc.service
sudo systemctl stop fedzk-coordinator.service
```

### Enabling Services at Boot

```bash
sudo systemctl enable fedzk-mpc.service
sudo systemctl enable fedzk-coordinator.service
```

### Checking Service Status

```bash
sudo systemctl status fedzk-mpc.service
sudo systemctl status fedzk-coordinator.service
```

### Viewing Logs

```bash
sudo journalctl -u fedzk-mpc.service -f
sudo journalctl -u fedzk-coordinator.service -f
```

## Configuration

The services are configured to:

- Run under the `fedzk` user
- Use the shell scripts in `examples/service-scripts/` to start the servers
- Apply security hardening measures to protect the system
- Restart automatically on failure

### MPC Server API Keys

The MPC server can be configured with API keys for authentication. During installation, you'll be prompted to enter comma-separated API keys.

You can manually modify the API keys after installation:

```bash
sudo systemctl edit fedzk-mpc.service
```

Add the following under the `[Service]` section:

```
Environment="MPC_API_KEYS=your_key1,your_key2"
```

After editing, restart the service:

```bash
sudo systemctl restart fedzk-mpc.service
``` 