#!/bin/bash

module purge
conda activate ocp-dev
module load cuda/11.1.1

# NCCL hangs otherwise..?
export NCCL_IB_DISABLE=1
# Enable submitit to requeue the job
export SBATCH_REQUEUE=1

set -x
python main.py --config-yml configs/mlperf_rcp_256.yml \
    --mode train --distributed --submit --num-gpus 4 --num-workers 31 \
    --num-nodes 16 --slurm-timeout 12 --identifier ocp-rcp-256-02-01 --seed 1
python main.py --config-yml configs/mlperf_rcp_256.yml \
    --mode train --distributed --submit --num-gpus 4 --num-workers 31 \
    --num-nodes 16 --slurm-timeout 12 --identifier ocp-rcp-256-02-02 --seed 2
python main.py --config-yml configs/mlperf_rcp_256.yml \
    --mode train --distributed --submit --num-gpus 4 --num-workers 31 \
    --num-nodes 16 --slurm-timeout 12 --identifier ocp-rcp-256-02-03 --seed 3
python main.py --config-yml configs/mlperf_rcp_256.yml \
    --mode train --distributed --submit --num-gpus 4 --num-workers 31 \
    --num-nodes 16 --slurm-timeout 12 --identifier ocp-rcp-256-02-04 --seed 4
python main.py --config-yml configs/mlperf_rcp_256.yml \
    --mode train --distributed --submit --num-gpus 4 --num-workers 31 \
    --num-nodes 16 --slurm-timeout 12 --identifier ocp-rcp-256-02-05 --seed 5
python main.py --config-yml configs/mlperf_rcp_256.yml \
    --mode train --distributed --submit --num-gpus 4 --num-workers 31 \
    --num-nodes 16 --slurm-timeout 12 --identifier ocp-rcp-256-02-06 --seed 6
python main.py --config-yml configs/mlperf_rcp_256.yml \
    --mode train --distributed --submit --num-gpus 4 --num-workers 31 \
    --num-nodes 16 --slurm-timeout 12 --identifier ocp-rcp-256-02-07 --seed 7
python main.py --config-yml configs/mlperf_rcp_256.yml \
    --mode train --distributed --submit --num-gpus 4 --num-workers 31 \
    --num-nodes 16 --slurm-timeout 12 --identifier ocp-rcp-256-02-08 --seed 8
python main.py --config-yml configs/mlperf_rcp_256.yml \
    --mode train --distributed --submit --num-gpus 4 --num-workers 31 \
    --num-nodes 16 --slurm-timeout 12 --identifier ocp-rcp-256-02-09 --seed 9
python main.py --config-yml configs/mlperf_rcp_256.yml \
    --mode train --distributed --submit --num-gpus 4 --num-workers 31 \
    --num-nodes 16 --slurm-timeout 12 --identifier ocp-rcp-256-02-10 --seed 10
