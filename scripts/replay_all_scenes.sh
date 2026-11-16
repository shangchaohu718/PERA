#!/usr/bin/env bash
set -euo pipefail
PERA_REPLAY_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="$PERA_REPLAY_ROOT/src"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
cd -- "$PERA_REPLAY_ROOT"
exec "${PERA_PYTHON:-python}" tools/replay_suite.py "$@"
