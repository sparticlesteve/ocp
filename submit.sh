#!/bin/bash

conda activate ocp-models
export NCCL_SOCKET_IFNAME=eth
id=cgpu-003

set -x
python main.py --config-yml configs/mlperf_hpc.yml \
    --mode train --distributed --submit \
    --amp \
    --num-gpus 8 \
    --num-workers 8 \
    --num-nodes 4 \
    --identifier $id \
    --slurm-timeout 8

#--checkpoint checkpoints/2021-06-06-18-06-08/checkpoint.pt
