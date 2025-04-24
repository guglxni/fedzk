#!/bin/bash
set -e

# Default configuration
PORT=9000
HOST="0.0.0.0"
DEBUG=false
LOG_LEVEL="info"
API_KEYS=""

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --port)
            PORT="$2"
            shift 2
            ;;
        --host)
            HOST="$2"
            shift 2
            ;;
        --debug)
            DEBUG=true
            shift
            ;;
        --log-level)
            LOG_LEVEL="$2"
            shift 2
            ;;
        --api-keys)
            API_KEYS="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $(basename "$0") [OPTIONS]"
            echo "Run the FedZK MPC server"
            echo ""
            echo "Options:"
            echo "  --port PORT       Port to run the server on (default: 9000)"
            echo "  --host HOST       Host to bind the server to (default: 0.0.0.0)"
            echo "  --debug           Enable debug mode"
            echo "  --log-level LEVEL Set log level: debug, info, warning, error (default: info)"
            echo "  --api-keys KEYS   Comma-separated list of API keys for authentication"
            echo "  --help            Show this help message and exit"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Activate virtual environment if it exists
if [[ -d ".venv" ]]; then
    source .venv/bin/activate
fi

# Set API keys environment variable if provided
if [[ -n "$API_KEYS" ]]; then
    export MPC_API_KEYS="$API_KEYS"
fi

# Additional debug arguments if debug mode is enabled
DEBUG_ARGS=""
if [[ "$DEBUG" == "true" ]]; then
    DEBUG_ARGS="--reload --workers 1"
    LOG_LEVEL="debug"
fi

# Start the MPC server
echo "Starting FedZK MPC Server on $HOST:$PORT with log level $LOG_LEVEL"
uvicorn fedzk.mpc.server:app --host "$HOST" --port "$PORT" --log-level "$LOG_LEVEL" $DEBUG_ARGS 