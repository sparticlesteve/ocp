#!/bin/bash
#SBATCH -C gpu
#SBATCH -J ocp-pm
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=32
#SBATCH --time 10
#SBATCH -o logs/slurm-%x-%j.out

args=$@

module purge
source $CONDA_INIT_SCRIPT
conda activate ocp-dev
module load cuda/11.1.1

# Distributed config
export MASTER_ADDR=$(hostname)
export MASTER_PORT=29504
export NCCL_IB_DISABLE=1
export NCCL_DEBUG=WARN
export NCCL_SOCKET_IFNAME=hsn

# Run ID
id=pm-n${SLURM_NTASKS}-$SLURM_JOB_ID

set -x
srun -l -u scripts/run_training.sh \
    --config-yml configs/mlperf_hpc_pm.yml \
    --identifier $id $args
