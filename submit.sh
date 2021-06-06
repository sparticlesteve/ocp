#!/bin/bash

conda activate ocp-models
export NCCL_SOCKET_IFNAME=eth

python main.py --config-yml configs/mlperf_hpc.yml \
    --mode train --distributed --submit \
    --amp \
    --num-gpus 8 \
    --num-workers 8 \
    --num-nodes 2 \
    --slurm-timeout 8
