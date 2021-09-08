#!/bin/bash

module purge
conda activate ocp-dev
module load cuda/11.1.1

# NCCL hangs otherwise..?
export NCCL_IB_DISABLE=1
# Enable submitit to requeue the job
export SBATCH_REQUEUE=1

set -x
python main.py --config-yml configs/mlperf_rcp_1024.yml \
    --mode train --distributed --submit --num-gpus 4  --num-workers 31 \
    --num-nodes 64 --slurm-timeout 4 --identifier ocp-rcp-1024-01 --seed 1
python main.py --config-yml configs/mlperf_rcp_1024.yml \
    --mode train --distributed --submit --num-gpus 4  --num-workers 31 \
    --num-nodes 64 --slurm-timeout 4 --identifier ocp-rcp-1024-02 --seed 2
python main.py --config-yml configs/mlperf_rcp_1024.yml \
    --mode train --distributed --submit --num-gpus 4  --num-workers 31 \
    --num-nodes 64 --slurm-timeout 4 --identifier ocp-rcp-1024-03 --seed 3
python main.py --config-yml configs/mlperf_rcp_1024.yml \
    --mode train --distributed --submit --num-gpus 4  --num-workers 31 \
    --num-nodes 64 --slurm-timeout 4 --identifier ocp-rcp-1024-04 --seed 4
python main.py --config-yml configs/mlperf_rcp_1024.yml \
    --mode train --distributed --submit --num-gpus 4  --num-workers 31 \
    --num-nodes 64 --slurm-timeout 4 --identifier ocp-rcp-1024-05 --seed 5
python main.py --config-yml configs/mlperf_rcp_1024.yml \
    --mode train --distributed --submit --num-gpus 4  --num-workers 31 \
    --num-nodes 64 --slurm-timeout 4 --identifier ocp-rcp-1024-06 --seed 6
python main.py --config-yml configs/mlperf_rcp_1024.yml \
    --mode train --distributed --submit --num-gpus 4  --num-workers 31 \
    --num-nodes 64 --slurm-timeout 4 --identifier ocp-rcp-1024-07 --seed 7
python main.py --config-yml configs/mlperf_rcp_1024.yml \
    --mode train --distributed --submit --num-gpus 4  --num-workers 31 \
    --num-nodes 64 --slurm-timeout 4 --identifier ocp-rcp-1024-08 --seed 8
python main.py --config-yml configs/mlperf_rcp_1024.yml \
    --mode train --distributed --submit --num-gpus 4  --num-workers 31 \
    --num-nodes 64 --slurm-timeout 4 --identifier ocp-rcp-1024-09 --seed 9
python main.py --config-yml configs/mlperf_rcp_1024.yml \
    --mode train --distributed --submit --num-gpus 4  --num-workers 31 \
    --num-nodes 64 --slurm-timeout 4 --identifier ocp-rcp-1024-10 --seed 10
