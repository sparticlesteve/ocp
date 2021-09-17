#!/bin/bash
#SBATCH -C gpu
#SBATCH -J ocp-cgpu
#SBATCH --image=sfarrell/mlperf-ocp:latest
#SBATCH --ntasks-per-node=2
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=10
#SBATCH --time 10
#SBATCH -o logs/slurm-%x-%j.out

args=$@

# Distributed config
export MASTER_ADDR=$(hostname)
export MASTER_PORT=29504
export NCCL_DEBUG=WARN
export NCCL_SOCKET_IFNAME=eth
export NCCL_IB_HCA=mlx5_0:1,mlx5_2:1,mlx5_4:1,mlx5_6:1

# Run ID
id=cgpu-n${SLURM_NTASKS}-$SLURM_JOB_ID

set -x
srun -l -u shifter scripts/run_training.sh \
    --config-yml configs/mlperf_hpc.yml \
    --identifier $id $args
