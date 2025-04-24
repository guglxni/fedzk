#!/bin/bash
set -e

# Default values
COORDINATOR_URL="http://localhost:8000"
MPC_URL="http://localhost:8001"
MODEL="resnet18"
DATASET="cifar10"
EPOCHS=1
API_KEY=""

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --coordinator-url)
      COORDINATOR_URL="$2"
      shift 2
      ;;
    --mpc-url)
      MPC_URL="$2"
      shift 2
      ;;
    --model)
      MODEL="$2"
      shift 2
      ;;
    --dataset)
      DATASET="$2"
      shift 2
      ;;
    --epochs)
      EPOCHS="$2"
      shift 2
      ;;
    --api-key)
      API_KEY="$2"
      shift 2
      ;;
    *)
      echo "Unknown parameter: $1"
      exit 1
      ;;
  esac
done

# Start client training and proof generation
echo "Starting FedZK Client with model $MODEL on dataset $DATASET"
echo "Connecting to Coordinator: $COORDINATOR_URL"
echo "Connecting to MPC server: $MPC_URL"

# Build API key argument if provided
API_KEY_ARG=""
if [ -n "$API_KEY" ]; then
  API_KEY_ARG="--api-key $API_KEY"
fi

# Start training and generate proof
python -m fedzk.cli train --model $MODEL --dataset $DATASET --epochs $EPOCHS
python -m fedzk.cli generate --mpc-server $MPC_URL $API_KEY_ARG
python -m fedzk.cli submit --coordinator $COORDINATOR_URL 