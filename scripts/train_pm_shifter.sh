#!/bin/bash
#SBATCH -C gpu
#SBATCH -J ocp-pm
#SBATCH --image=sfarrell/mlperf-ocp:latest
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=32
#SBATCH --time 4:00:00
#SBATCH -o logs/slurm-%x-%j.out

args=$@

# Default settings
: "${OCP_CONFIG:=configs/mlperf_hpc_pm.yml}"

# Distributed config
export MASTER_ADDR=$(hostname)
export MASTER_PORT=29504
export NCCL_SOCKET_IFNAME=hsn
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}

# Run the dummy cuda app to "fix" cuda init errors
if [ ! -f ./dummy ]; then
    echo "int main() {cudaFree(0);}" > dummy.cu && nvcc -o dummy dummy.cu
fi
srun ./dummy

set -x
srun -l -u shifter scripts/run_training.sh --config-yml $OCP_CONFIG $args
