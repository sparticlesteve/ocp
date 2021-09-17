"""Launch utility for distributed training at NERSC

To be used in an sbatch script like:
    srun ... python -m ocpmodels.common.nersc_launch main.py ...
"""

import os
import sys
import argparse


def setup_distributed_env():
    """
    Setup the distributed environment for PyTorch distributed at NERSC.

    Uses command line arguments and NERSC+SLURM environment variables to
    set the environment variables usable by PyTorch's env method in
    init_process_group.
    """

    # TODO: setup command line options
    #parser = argparse.ArgumentParser()

    # Initially I will assume some env variables are pre-defined
    assert 'MASTER_ADDR' in os.environ
    assert 'MASTER_PORT' in os.environ

    # Set WORLD_SIZE and RANK variables
    os.environ['WORLD_SIZE'] = os.environ['SLURM_NTASKS']
    os.environ['RANK'] = os.environ['SLURM_PROCID']

    # Local rank is processed by argument in user script
    local_rank = os.environ['SLURM_LOCALID']
    sys.argv.append(f'--local_rank={local_rank}')

    if os.environ['RANK'] == '0':
        print('Environment configured with nersc_launch:')
        print('  MASTER_ADDR:', os.environ['MASTER_ADDR'])
        print('  MASTER_PORT:', os.environ['MASTER_PORT'])
        print('  WORLD_SIZE: ', os.environ['WORLD_SIZE'])

if __name__ == '__main__':
    setup_distributed_env()
