#!/bin/bash

# FedZK Service Installation Script
# This script installs and enables systemd services for FedZK

set -e

# Default installation path
INSTALL_PATH="/opt/fedzk"
USER="fedzk"
SERVICES_DIR=$(dirname "$0")

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --install-path)
      INSTALL_PATH="$2"
      shift 2
      ;;
    --user)
      USER="$2"
      shift 2
      ;;
    --help)
      echo "Usage: $0 [options]"
      echo ""
      echo "Options:"
      echo "  --install-path PATH  Set the installation path (default: /opt/fedzk)"
      echo "  --user USER          Set the user to run the services (default: fedzk)"
      echo "  --help               Display this help message"
      exit 0
      ;;
    *)
      echo "Unknown option: $1"
      echo "Use --help for usage information"
      exit 1
      ;;
  esac
done

# Check if running as root
if [ "$EUID" -ne 0 ]; then
  echo "This script requires sudo privileges to install systemd services."
  echo "Please run with sudo."
  exit 1
fi

# Create user if it doesn't exist
if ! id -u "$USER" &>/dev/null; then
  echo "Creating user $USER..."
  useradd -m -s /bin/bash "$USER"
fi

# Create installation directory if it doesn't exist
if [ ! -d "$INSTALL_PATH" ]; then
  echo "Creating installation directory $INSTALL_PATH..."
  mkdir -p "$INSTALL_PATH"
  chown -R "$USER:$USER" "$INSTALL_PATH"
fi

# Prompt for API keys
read -p "Enter comma-separated API keys for MPC server (leave empty for no authentication): " API_KEYS

# Copy service files to systemd directory
echo "Installing systemd service files..."
cp "$SERVICES_DIR/fedzk-mpc.service" /etc/systemd/system/
cp "$SERVICES_DIR/fedzk-coordinator.service" /etc/systemd/system/

# Replace paths and user in service files
sed -i "s|/opt/fedzk|$INSTALL_PATH|g" /etc/systemd/system/fedzk-mpc.service
sed -i "s|/opt/fedzk|$INSTALL_PATH|g" /etc/systemd/system/fedzk-coordinator.service
sed -i "s|User=fedzk|User=$USER|g" /etc/systemd/system/fedzk-mpc.service
sed -i "s|User=fedzk|User=$USER|g" /etc/systemd/system/fedzk-coordinator.service
sed -i "s|Group=fedzk|Group=$USER|g" /etc/systemd/system/fedzk-mpc.service
sed -i "s|Group=fedzk|Group=$USER|g" /etc/systemd/system/fedzk-coordinator.service

# Update API keys in MPC service
if [ -n "$API_KEYS" ]; then
  sed -i "s|%API_KEYS%|$API_KEYS|g" /etc/systemd/system/fedzk-mpc.service
else
  # If no API keys provided, remove the environment variable line
  sed -i '/Environment="MPC_API_KEYS=%API_KEYS%"/d' /etc/systemd/system/fedzk-mpc.service
fi

# Reload systemd
echo "Reloading systemd daemon..."
systemctl daemon-reload

echo "Services installed successfully!"
echo ""
echo "To start the services:"
echo "  sudo systemctl start fedzk-mpc.service"
echo "  sudo systemctl start fedzk-coordinator.service"
echo ""
echo "To enable services at boot:"
echo "  sudo systemctl enable fedzk-mpc.service"
echo "  sudo systemctl enable fedzk-coordinator.service"
echo ""
echo "To check service status:"
echo "  sudo systemctl status fedzk-mpc.service"
echo "  sudo systemctl status fedzk-coordinator.service" 