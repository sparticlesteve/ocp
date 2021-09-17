#!/bin/bash

# This script will run the actual training command with provided
# command line options. It is run by every rank and sets the per-rank
# environment variables needed for pytorch distributed initialization.

args=$@

export WORLD_SIZE=$SLURM_NTASKS
export RANK=$SLURM_PROCID
export LOCAL_RANK=$SLURM_LOCALID

python main.py --mode train --distributed --local_rank $LOCAL_RANK $args
