#!/bin/bash

#SBATCH --cpus-per-task=4               # CPU cores
#SBATCH --nodelist=rax17           # uncomment/modify to match your cluster
#SBATCH --nodes=1                   # uncomment if you need GPUs
#SBATCH --ntasks=1  


nvidia-smi
export CUDA_VISIBLE_DEVICES=1
export PYTHONUNBUFFERED=1

if [ -f "~/Documents/SC-LSTM/venv/bin/python/activate" ]; then
    source "~/Documents/SC-LSTM/venv/bin/python/activate"
fi


SCRIPT="stoc_boundary_attack.py"
if [ ! -f "$SCRIPT" ]; then
    echo "ERROR: $SCRIPT not found in $(pwd)"
    exit 3
fi

exec python -u "$SCRIPT" "$@"
