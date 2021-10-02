#!/bin/bash

#-------------------------------------------------------------------------------
# Scanning for best omp_num_threads and num_workers
export OCP_CONFIG=configs/perf_test_5k.yml
export ENABLE_PROFILING=1
#for ompnt in 1 2 4 8 16; do
for ompnt in 1 32 64; do
    export OMP_NUM_THREADS=$ompnt
    #for nio in 4 8 16; do
    for nio in 1 2; do
        sbatch -J pm-perf-omp${ompnt}-io${nio} -t 10 -d singleton --exclusive \
            --ntasks-per-node=1 \
            scripts/train_pm_shifter.sh --num_workers=$nio
    done
done

#-------------------------------------------------------------------------------
# Performance test on 1gpu and 4gpus, with/without nvidia (fixed) binding
#export OMP_NUM_THREADS=1
#export OCP_CONFIG=configs/perf_test_5k.yml
##export ENABLE_PROFILING=1
#export ENABLE_NV_BINDING=0
#sbatch -J pm-prof -t 30 --ntasks-per-node=1 -d singleton --exclusive scripts/train_pm_shifter.sh
#sbatch -J pm-prof -t 30 --ntasks-per-node=4 -d singleton --exclusive scripts/train_pm_shifter.sh
#export ENABLE_NV_BINDING=1
#sbatch -J pm-prof -t 30 --ntasks-per-node=1 -d singleton --exclusive scripts/train_pm_shifter.sh
#sbatch -J pm-prof -t 30 --ntasks-per-node=4 -d singleton --exclusive scripts/train_pm_shifter.sh

#-------------------------------------------------------------------------------

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

