"""Portable Slurm settings shared by the TCGA submission commands."""
import os
from pathlib import Path
import subprocess
import sys

CODE_ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = CODE_ROOT / 'jobs/run_job.sh'


def add_arguments(parser, *, cpus, memory, time_limit, gpus=0):
    parser.add_argument('--account', default=os.environ.get('STPATH_SLURM_ACCOUNT'))
    parser.add_argument('--partition', default=os.environ.get('STPATH_SLURM_PARTITION'))
    parser.add_argument('--cpus', type=int, default=cpus)
    parser.add_argument('--mem', dest='memory', default=memory)
    parser.add_argument('--time', dest='time_limit', default=time_limit)
    parser.add_argument('--gpus', type=int, default=gpus)
    parser.add_argument('--exclude-nodes')


def command(args, *, cpus=None, memory=None, time_limit=None, gpus=None):
    cpus = args.cpus if cpus is None else cpus
    gpus = args.gpus if gpus is None else gpus
    if cpus < 1 or gpus < 0:
        raise ValueError('CPUs must be positive and GPU count nonnegative')
    python = os.environ.get('STPATH_PYTHON', sys.executable)
    result = ['sbatch', '--parsable',
        f'--export=ALL,STPATH_CODE_ROOT={CODE_ROOT},STPATH_PYTHON={python}',
        f'--chdir={CODE_ROOT}', '--nodes=1', '--ntasks=1', f'--cpus-per-task={cpus}',
        f'--mem={args.memory if memory is None else memory}',
        f'--time={args.time_limit if time_limit is None else time_limit}']
    if gpus:
        result.append(f'--gpus={gpus}')
    for value, option in ((args.account, 'account'), (args.partition, 'partition'),
                          (args.exclude_nodes, 'exclude')):
        if value:
            result.append(f'--{option}={value}')
    return result


def submit(command):
    try:
        result = subprocess.run(command, check=True, text=True, capture_output=True)
    except subprocess.CalledProcessError as error:
        if error.stderr:
            print(error.stderr, end='', flush=True)
        raise
    if result.stderr:
        print(result.stderr, end='', flush=True)
    job = result.stdout.strip().split(';')[0]
    if not job.isdigit():
        raise RuntimeError(f'Unexpected sbatch response; inspect the queue before retrying: {result.stdout!r}')
    return job
