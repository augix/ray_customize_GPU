#!/bin/bash

# required environment variables
# RAY_ADDRESS

# Get the hostname to ensure unique resource names across nodes
HOSTNAME=$(hostname)

# Get the number of GPUs using nvidia-smi
NUM_GPUS=$(nvidia-smi -L | wc -l)

# Construct the resources JSON string
RESOURCES="{"
for ((i=0; i<$NUM_GPUS; i++)); do
    RESOURCES+="\"${HOSTNAME}_GPU${i}\": 1"
    if [ $i -lt $((NUM_GPUS-1)) ]; then
        RESOURCES+=", "
    fi
done
RESOURCES+="}"

# Start Ray with the custom resources (for head node)
ray start --address=$RAY_ADDRESS --num-gpus $NUM_GPUS --resources="$RESOURCES"