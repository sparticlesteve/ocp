#!/bin/bash
#SBATCH -C gpu
#SBATCH -J ocp-cgpu
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=10
#SBATCH --time 4:00:00
#SBATCH -o logs/slurm-%x-%j.out

args=$@

# Default settings
: "${OCP_CONFIG:=configs/mlperf_hpc.yml}"

# Setup software
conda activate ocp-dev
module load cuda/11.1.1

# Distributed config
export MASTER_ADDR=$(hostname)
export MASTER_PORT=29504
export NCCL_DEBUG=WARN
export NCCL_SOCKET_IFNAME=eth
export NCCL_IB_HCA=mlx5_0:1,mlx5_2:1,mlx5_4:1,mlx5_6:1

set -x
srun -u -l scripts/run_training.sh --config-yml $OCP_CONFIG $args
