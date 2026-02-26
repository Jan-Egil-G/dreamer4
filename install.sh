#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   ./install.sh          # installs cu121 by default
#   ./install.sh cu118    # installs cu118 instead
#   ./install.sh cu121

CUDA_VARIANT="${1:-cu121}"

python -m pip install --upgrade pip setuptools wheel

# Ensure we're not mixing CPU and CUDA builds
python -m pip uninstall -y torch torchvision >/dev/null 2>&1 || true

case "$CUDA_VARIANT" in
  cu121)
    python -m pip install --index-url https://download.pytorch.org/whl/cu121 torch torchvision
    ;;
  cu118)
    python -m pip install --index-url https://download.pytorch.org/whl/cu118 torch torchvision
    ;;
  *)
    echo "Unknown CUDA variant: $CUDA_VARIANT"
    echo "Use: cu121 | cu118"
    exit 1
    ;;
esac

# Install the rest
python -m pip install -r requirements.txt

# Fail fast if CUDA isn't actually usable
python - <<'PY'
import torch, torchvision
print("torch:", torch.__version__)
print("torchvision:", torchvision.__version__)
ok = torch.cuda.is_available()
print("cuda available:", ok)
if not ok:
    raise SystemExit(
        "ERROR: Installed CUDA wheels but torch.cuda.is_available() is False.\n"
        "Common causes: container not started with GPU access (--gpus all),\n"
        "no NVIDIA driver on host, or incompatible driver/CUDA setup."
    )
print("gpu:", torch.cuda.get_device_name(0))
PY
