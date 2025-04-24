#!/bin/bash
# run-services.sh - Start both MPC server and Coordinator for FedZK

# Default configuration
MPC_PORT=8081
COORD_PORT=8000
MPC_HOST="127.0.0.1"
COORD_HOST="127.0.0.1"
MPC_DEBUG=false
COORD_DEBUG=false
MPC_API_KEYS="test-key"
LOG_LEVEL="info"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --mpc-port)
      MPC_PORT="$2"
      shift 2
      ;;
    --coord-port)
      COORD_PORT="$2"
      shift 2
      ;;
    --mpc-host)
      MPC_HOST="$2"
      shift 2
      ;;
    --coord-host)
      COORD_HOST="$2"
      shift 2
      ;;
    --mpc-debug)
      MPC_DEBUG=true
      shift
      ;;
    --coord-debug)
      COORD_DEBUG=true
      shift
      ;;
    --log-level)
      LOG_LEVEL="$2"
      shift 2
      ;;
    --api-keys)
      MPC_API_KEYS="$2"
      shift 2
      ;;
    -h|--help)
      echo "Usage: $0 [options]"
      echo "Options:"
      echo "  --mpc-port PORT     Port for MPC server (default: 8081)"
      echo "  --coord-port PORT   Port for Coordinator server (default: 8000)"
      echo "  --mpc-host HOST     Host for MPC server (default: 127.0.0.1)"
      echo "  --coord-host HOST   Host for Coordinator server (default: 127.0.0.1)"
      echo "  --mpc-debug         Enable debug mode for MPC server"
      echo "  --coord-debug       Enable debug mode for Coordinator server"
      echo "  --log-level LEVEL   Log level (default: info)"
      echo "  --api-keys KEYS     Comma-separated list of allowed API keys for MPC server"
      exit 0
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

# Activate virtual environment if it exists
if [ -d ".venv" ]; then
  source .venv/bin/activate
fi

# Export MPC API keys
export MPC_API_KEYS="$MPC_API_KEYS"

# Start MPC server in background
echo "Starting MPC server on $MPC_HOST:$MPC_PORT..."
MPC_CMD="uvicorn fedzk.mpc.server:app --host $MPC_HOST --port $MPC_PORT --log-level $LOG_LEVEL"
if [ "$MPC_DEBUG" = true ]; then
  MPC_CMD="$MPC_CMD --reload"
fi

# Start Coordinator server in background
echo "Starting Coordinator server on $COORD_HOST:$COORD_PORT..."
COORD_CMD="uvicorn fedzk.coordinator.api:app --host $COORD_HOST --port $COORD_PORT --log-level $LOG_LEVEL"
if [ "$COORD_DEBUG" = true ]; then
  COORD_CMD="$COORD_CMD --reload"
fi

# Run both services in parallel with output prefixing
(
  $MPC_CMD 2>&1 | sed 's/^/[MPC] /' &
  MPC_PID=$!
  
  $COORD_CMD 2>&1 | sed 's/^/[COORD] /' &
  COORD_PID=$!
  
  # Handle shutdown signals
  trap "kill $MPC_PID $COORD_PID; exit" SIGINT SIGTERM
  
  # Wait for both processes to finish
  wait
) || exit $? 