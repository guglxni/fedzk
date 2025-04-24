#!/bin/bash
set -e

# Default values
PORT=8000
MIN_CLIENTS=3
AGGREGATION="fedavg"
OUTPUT_DIR="./federated_models"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --port)
      PORT="$2"
      shift 2
      ;;
    --min-clients)
      MIN_CLIENTS="$2"
      shift 2
      ;;
    --aggregation)
      AGGREGATION="$2"
      shift 2
      ;;
    --output-dir)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    *)
      echo "Unknown parameter: $1"
      exit 1
      ;;
  esac
done

# Create output directory if it doesn't exist
mkdir -p $OUTPUT_DIR

echo "Starting FedZK Coordinator on port $PORT"
echo "Minimum clients required: $MIN_CLIENTS"
echo "Aggregation strategy: $AGGREGATION"
echo "Model output directory: $OUTPUT_DIR"

# Start the coordinator server
python -m fedzk.coordinator.server \
  --port $PORT \
  --min-clients $MIN_CLIENTS \
  --aggregation $AGGREGATION \
  --output-dir $OUTPUT_DIR 