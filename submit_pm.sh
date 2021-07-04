#!/bin/bash

module purge
conda activate ocp-dev
module use /global/common/software/m1759/catalysis_dl_envs/cuda_modulefiles
module load cuda/11.1.1

# NCCL hangs otherwise..?
export NCCL_IB_DISABLE=1

# Testing batch size limits
id=pm-test-005

set -x
python main.py --config-yml configs/mlperf_hpc_pm.yml \
    --mode train --distributed --submit \
    --identifier $id \
    --num-gpus 4 \
    --num-workers 8 \
    --num-nodes 2 --amp \
    --slurm-timeout 12
    #--checkpoint ./checkpoints/2021-07-03-18-14-40-pm-test-004/checkpoint.pt
