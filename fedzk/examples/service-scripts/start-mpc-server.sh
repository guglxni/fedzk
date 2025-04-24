#!/bin/bash
set -e

# Default values
PORT=8001
API_KEYS=""
ENV_FILE=""

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --port)
      PORT="$2"
      shift 2
      ;;
    --api-keys)
      API_KEYS="$2"
      shift 2
      ;;
    --env-file)
      ENV_FILE="$2"
      shift 2
      ;;
    *)
      echo "Unknown parameter: $1"
      exit 1
      ;;
  esac
done

# Load environment variables if specified
if [ -n "$ENV_FILE" ] && [ -f "$ENV_FILE" ]; then
  echo "Loading environment variables from $ENV_FILE"
  export $(grep -v '^#' $ENV_FILE | xargs)
fi

# Set API keys if provided via command line
if [ -n "$API_KEYS" ]; then
  export MPC_API_KEYS=$API_KEYS
fi

echo "Starting FedZK MPC Server on port $PORT"
echo "API authentication: $([ -n "$MPC_API_KEYS" ] && echo "enabled" || echo "disabled")"

# Start the MPC server
python -m fedzk.mpc.server --port $PORT 