#!/usr/bin/env bash
set -euo pipefail
PERA_PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="$PERA_PROJECT_ROOT/src:$PERA_PROJECT_ROOT/reproduction:$PERA_PROJECT_ROOT/integrations/HoloMotion${PYTHONPATH:+:$PYTHONPATH}"
exec "${PERA_PYTHON:-python}" -m pera "$@"
