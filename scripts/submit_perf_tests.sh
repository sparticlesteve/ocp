#!/bin/bash

# Performance test on 1gpu and 4gpus
export OMP_NUM_THREADS=1
export WANDB_MODE="dryrun"
export OCP_CONFIG=configs/perf_test_5k.yml
export ENABLE_PROFILING=1
export ENABLE_NV_BINDING=1
sbatch -J pm-prof -t 30 --ntasks-per-node=1 -d singleton --exclusive \
    scripts/train_pm_shifter.sh
# Single node profiling
sbatch -J pm-prof -t 30 --ntasks-per-node=4 -d singleton --exclusive \
    scripts/train_pm_shifter.sh

# Single GPU tests
#export OCP_CONFIG=configs/perf_test_n1.yml
#args="-J pm-perf -d singleton -t 30 -n 1"
#sbatch $args scripts/train_pm_shifter.sh
#sbatch $args scripts/train_pm_shifter.sh --name="'dpp_opts'"
#sbatch $args scripts/train_pm_shifter.sh --name="'dpp_opts'" --optimizer="'FusedAdam'"
#sbatch $args scripts/train_pm_shifter.sh --name="'dpp_opts'" --optimizer="'FusedAdam'" --amp

## Single-gpu to 4-gpu comparison on cori-gpu
#export WANDB_MODE="dryrun"
#export OCP_CONFIG=configs/perf_test_5k.yml
#export ENABLE_PROFILING=0
#export ENABLE_NV_BINDING=1
#sbatch -J cgpu-prof -t 30 --ntasks-per-node=1 -d singleton --exclusive \
#    scripts/train_cgpu_shifter.sh --max_epochs=1
#sbatch -J cgpu-prof -t 30 --ntasks-per-node=8 -d singleton --exclusive \
#    scripts/train_cgpu_shifter.sh --max_epochs=2

