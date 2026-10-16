#!/usr/bin/env bash
# =============================================================================
# install_mbe.sh — MouseBrainExtractor (MBE) for OCEAN 2-D mouse skull-strip.
# =============================================================================
# MBE = MONAI SwinUNETR rodent brain extractor with a 2-D anisotropic model
# (invivo_aniso, -d 2 --patch_size 128) built for thick-slice data like your
# Bruker 2-D multi-slice T2 MSME.  Installed into a DEDICATED conda env; OCEAN
# calls it OUT-OF-PROCESS (keeps GPL-2.0 code at arm's length + OCEAN env clean).
# Native arm64 PyTorch — CPU inference works, no GPU needed.  License: GPL-2.0.
#
# Installs to ~/mbe :  MouseBrainExtractor/ (code) + MBE_weights/ (531 MB).
#
# Usage:
#   bash install_mbe.sh                 # env 'mbe', ~/mbe
#   bash install_mbe.sh --dir /custom   # custom install dir
# =============================================================================
set -euo pipefail

ENV_NAME="mbe"
ROOT="$HOME/mbe"
PYVER="3.11"
while [ $# -gt 0 ]; do
  case "$1" in
    --env) ENV_NAME="${2:?}"; shift 2 ;;
    --dir) ROOT="${2:?}"; shift 2 ;;
    --py)  PYVER="${2:?}"; shift 2 ;;
    -h|--help) sed -n '2,20p' "$0"; exit 0 ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done

CONDA=""
for c in /Users/cbd/miniforge3_default/bin/conda /Users/cbd/miniforge3/bin/conda \
         "$(command -v conda 2>/dev/null || true)"; do
  [ -n "$c" ] && [ -x "$c" ] && { CONDA="$c"; break; }
done
[ -z "$CONDA" ] && { echo "ERROR: conda not found." >&2; exit 1; }
command -v git >/dev/null || { echo "ERROR: git not found." >&2; exit 1; }
echo "Using conda: $CONDA"
mkdir -p "$ROOT"

# ---- 1. code ----------------------------------------------------------------
MBE_DIR="$ROOT/MouseBrainExtractor"
if [ ! -d "$MBE_DIR/.git" ]; then
  echo "Cloning MouseBrainExtractor…"
  git clone --depth 1 https://github.com/MouseSuite/MouseBrainExtractor.git "$MBE_DIR"
else
  echo "MBE code already present: $MBE_DIR"
fi

# ---- 2. env + deps ----------------------------------------------------------
if ! "$CONDA" env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  echo "Creating env '$ENV_NAME' (python $PYVER)…"
  "$CONDA" create -n "$ENV_NAME" "python=$PYVER" -y
fi
echo "Installing PyTorch + MONAI + deps (native arm64)…"
"$CONDA" run -n "$ENV_NAME" python -m pip install --upgrade pip
"$CONDA" run -n "$ENV_NAME" python -m pip install \
    torch torchvision "monai==1.3.0" "monai[einops]" \
    nibabel nilearn scikit-image scikit-learn scipy tqdm numpy

# ---- 3. weights (531 MB) ----------------------------------------------------
WT="$ROOT/MBE_weights"
if [ ! -d "$WT" ] || [ -z "$(ls -A "$WT" 2>/dev/null)" ]; then
  echo "Downloading pretrained weights (~531 MB)…"
  curl -L "https://data.mousesuite.org/mbe/MBE_weights.tar.gz" -o "$ROOT/MBE_weights.tar.gz"
  mkdir -p "$WT"
  tar xzf "$ROOT/MBE_weights.tar.gz" -C "$WT" --strip-components 0
  rm -f "$ROOT/MBE_weights.tar.gz"
fi

# ---- 4. verify --------------------------------------------------------------
echo; echo "Verifying…"
"$CONDA" run -n "$ENV_NAME" python - <<'PY'
import torch, monai
print("  torch :", torch.__version__, "| mps:", torch.backends.mps.is_available())
print("  monai :", monai.__version__)
PY
ENVP="$("$CONDA" run -n "$ENV_NAME" python -c 'import sys;print(sys.executable)')"
echo "  weight model folders:"; find "$WT" -name 'checkpoint_best.pth' 2>/dev/null | sed 's#.*/\([^/]*\)/checkpoint_best.pth#    \1#' | sort -u

cat <<EOF

=============================================================================
 MBE installed.
   env python : $ENVP
   code       : $MBE_DIR
   weights    : $WT

 OCEAN auto-discovers these under ~/mbe.  To pin explicitly:
   export OCEAN_MBE_PY="$ENVP"
   export OCEAN_MBE_DIR="$MBE_DIR"
   export OCEAN_MBE_WEIGHTS="$WT"

 Next: tell me it's installed and I'll run MBE's 2-D invivo_aniso model on your
 scan 3 and show you the mask before wiring it into OCEAN.
=============================================================================
EOF
