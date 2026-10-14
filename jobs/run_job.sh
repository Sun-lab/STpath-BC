#!/usr/bin/env bash
set -euo pipefail
if (( $# == 0 )); then
    echo 'Pass a Python script or -m module followed by its arguments.' >&2
    exit 2
fi
CODE_ROOT="${STPATH_CODE_ROOT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"
export PYTHONPATH="$CODE_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-${OMP_NUM_THREADS:-1}}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS" OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
ARGS=("$@")
for i in "${!ARGS[@]}"; do
    if [[ "${ARGS[$i]}" == '{task}' ]]; then
        ARGS[$i]="${SLURM_ARRAY_TASK_ID:?The task placeholder requires a Slurm array job}"
    fi
done
cd "$CODE_ROOT"
exec "${STPATH_PYTHON:-python}" -u "${ARGS[@]}"
