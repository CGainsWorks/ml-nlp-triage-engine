#!/usr/bin/env bash
# Create a virtual environment and install the pinned dependencies.
#
#   bash scripts/setup.sh
#
# Installs the CPU build of PyTorch. The default PyPI wheel pulls the entire
# CUDA runtime with it, which is several gigabytes of no use on a machine
# without an NVIDIA card. If you have a GPU, install torch yourself first and
# this script will leave it alone.

set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON="${PYTHON:-python3}"
VENV="${VENV:-.venv}"

if ! "$PYTHON" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 9) else 1)'; then
    echo "Python 3.9 or newer is required; found $("$PYTHON" --version)" >&2
    exit 1
fi

if [ ! -d "$VENV" ]; then
    echo "==> creating virtual environment in $VENV"
    "$PYTHON" -m venv "$VENV"
fi

# shellcheck disable=SC1091
if [ -f "$VENV/bin/activate" ]; then
    source "$VENV/bin/activate"
else
    source "$VENV/Scripts/activate"   # Git Bash on Windows
fi

python -m pip install --upgrade pip

if python -c 'import torch' 2>/dev/null; then
    echo "==> torch already present ($(python -c 'import torch; print(torch.__version__)')), leaving it alone"
else
    echo "==> installing CPU-only torch"
    pip install --index-url https://download.pytorch.org/whl/cpu torch==2.2.2
fi

echo "==> installing the rest"
pip install -r requirements-dev.txt

cat <<'EOF'

Done. Next:

    source .venv/bin/activate     # or .venv/Scripts/activate on Windows
    make demo                     # build everything and serve it

EOF
