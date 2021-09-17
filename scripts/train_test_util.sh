#!/bin/bash
#SBATCH -C gpu
#SBATCH -J ocp-test
#SBATCH -N 2
#SBATCH --ntasks-per-node=2
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=10
#SBATCH --time 10

conda activate ocp-dev
module load cuda/11.1.1

# NCCL hang workaround
#export NCCL_IB_DISABLE=1

# Run ID
id=sbatch-test-001
export MASTER_ADDR=$(hostname)
export MASTER_PORT=25207

set -x
srun -l -u python -m ocpmodels.common.nersc_launch main.py \
    --config-yml configs/mlperf_hpc.yml \
    --mode train --distributed --identifier $id --num-gpus 2
