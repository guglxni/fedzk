#!/bin/bash
set -e

# Default values
COORDINATOR_URL="http://localhost:8000"
MODEL_TYPE="simple"
DATASET="mnist"
SECURE=false
MPC_SERVER=""
API_KEY=""
EPOCHS=1
BATCH_SIZE=32

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --coordinator)
      COORDINATOR_URL="$2"
      shift 2
      ;;
    --model)
      MODEL_TYPE="$2"
      shift 2
      ;;
    --dataset)
      DATASET="$2"
      shift 2
      ;;
    --secure)
      SECURE=true
      shift
      ;;
    --mpc-server)
      MPC_SERVER="$2"
      shift 2
      ;;
    --api-key)
      API_KEY="$2"
      shift 2
      ;;
    --epochs)
      EPOCHS="$2"
      shift 2
      ;;
    --batch-size)
      BATCH_SIZE="$2"
      shift 2
      ;;
    *)
      echo "Unknown parameter: $1"
      exit 1
      ;;
  esac
done

echo "Starting FedZK Client"
echo "Coordinator URL: $COORDINATOR_URL"
echo "Model type: $MODEL_TYPE"
echo "Dataset: $DATASET"
echo "Epochs: $EPOCHS"
echo "Batch size: $BATCH_SIZE"

# Build the command with optional arguments
CMD="python -m fedzk.client.run --coordinator $COORDINATOR_URL --model $MODEL_TYPE --dataset $DATASET --epochs $EPOCHS --batch-size $BATCH_SIZE"

if [ "$SECURE" = true ]; then
  echo "Using secure mode"
  CMD="$CMD --secure"
fi

if [ -n "$MPC_SERVER" ]; then
  echo "Using MPC server: $MPC_SERVER"
  CMD="$CMD --mpc-server $MPC_SERVER"
  
  if [ -n "$API_KEY" ]; then
    echo "Using API key authentication"
    CMD="$CMD --api-key $API_KEY"
  fi
fi

# Run the client
echo "Executing: $CMD"
eval $CMD 