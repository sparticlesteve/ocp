#!/bin/bash

module purge
conda activate ocp-dev
#module use /global/common/software/m1759/catalysis_dl_envs/cuda_modulefiles
module load cuda/11.1.1

# NCCL hangs otherwise..?
export NCCL_IB_DISABLE=1
# Enable submitit to requeue the job
export SBATCH_REQUEUE=1

# Run ID
id=pm-014-n128-b4

set -x
python main.py --config-yml configs/mlperf_hpc_pm.yml \
    --mode train --distributed --submit \
    --identifier $id \
    --num-gpus 4 \
    --num-workers 31 \
    --num-nodes 32 \
    --slurm-timeout 8
