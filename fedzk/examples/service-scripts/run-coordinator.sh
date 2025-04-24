#!/bin/bash
set -e

# Default configuration
PORT=8000
HOST="0.0.0.0"
DEBUG=false
LOG_LEVEL="info"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
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
        --help)
        echo "Usage: $0 [OPTIONS]"
        echo "Options:"
        echo "  --port PORT       Port to run coordinator on (default: 8000)"
        echo "  --host HOST       Host to bind to (default: 0.0.0.0)"
        echo "  --debug           Enable debug mode"
        echo "  --log-level LEVEL Set log level (default: info)"
        echo "  --help            Show this help message"
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
    echo "Activating virtual environment..."
    source .venv/bin/activate
fi

# Additional args for debug mode
DEBUG_ARGS=""
if [ "$DEBUG" = true ]; then
    DEBUG_ARGS="--reload"
    LOG_LEVEL="debug"
fi

echo "Starting FedZK Coordinator server on $HOST:$PORT..."
uvicorn fedzk.coordinator.server:app --host $HOST --port $PORT --log-level $LOG_LEVEL $DEBUG_ARGS
