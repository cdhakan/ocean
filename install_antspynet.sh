#!/usr/bin/env bash
# =============================================================================
# install_antspynet.sh — ANTsPyNet mouse brain extraction for OCEAN.
# =============================================================================
# Installs ANTsPyNet (+ ANTsPy + TensorFlow) into a DEDICATED conda env so it
# never bloats OCEAN's runtime env.  OCEAN calls it OUT-OF-PROCESS (subprocess)
# via antspynet_mouse_extract.py, so TensorFlow stays out of the packaged app.
#
#   Native arm64 macOS wheels exist for antspyx (cp312) and tensorflow (cp312).
#   Model weights auto-download from figshare on first run.  License: Apache-2.0.
#
# Usage:
#   bash install_antspynet.sh                 # env 'antspynet', python 3.12
#   bash install_antspynet.sh --env myenv
# =============================================================================
set -euo pipefail

ENV_NAME="antspynet"
PYVER="3.12"
while [ $# -gt 0 ]; do
  case "$1" in
    --env) ENV_NAME="${2:?--env needs a name}"; shift 2 ;;
    --py)  PYVER="${2:?--py needs a version}";  shift 2 ;;
    -h|--help) sed -n '2,18p' "$0"; exit 0 ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done

CONDA=""
for c in /Users/cbd/miniforge3_default/bin/conda \
         /Users/cbd/miniforge3/bin/conda \
         "$(command -v conda 2>/dev/null || true)"; do
  if [ -n "$c" ] && [ -x "$c" ]; then CONDA="$c"; break; fi
done
[ -z "$CONDA" ] && { echo "ERROR: conda not found." >&2; exit 1; }
echo "Using conda: $CONDA"

if ! "$CONDA" env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  echo "Creating env '$ENV_NAME' (python $PYVER)…"
  "$CONDA" create -n "$ENV_NAME" "python=$PYVER" -y
fi

echo "Installing antspyx + antspynet + tensorflow (native arm64 wheels)…"
"$CONDA" run -n "$ENV_NAME" python -m pip install --upgrade pip
"$CONDA" run -n "$ENV_NAME" python -m pip install \
    "antspyx==0.6.3" antspynet "tensorflow==2.21.0" nibabel numpy

echo
echo "Verifying imports…"
"$CONDA" run -n "$ENV_NAME" python - <<'PY'
import ants, antspynet, tensorflow as tf, numpy, nibabel
print("  antspyx    :", ants.__version__)
print("  antspynet  :", getattr(antspynet, "__version__", "ok"))
print("  tensorflow :", tf.__version__)
print("  OK — mouse_brain_extraction available:", hasattr(antspynet, "mouse_brain_extraction"))
PY

ENVP="$("$CONDA" run -n "$ENV_NAME" python -c 'import sys;print(sys.executable)')"
cat <<EOF

=============================================================================
 ANTsPyNet installed in env '$ENV_NAME'.
   python: $ENVP

 Test it on your Bruker scan (reads scan 3, runs both models, saves overlays):

   OCEAN auto-discovers this env, but you can pin it:
     export OCEAN_ANTSPYNET_PY="$ENVP"

 Model weights (~tens of MB) download from figshare on the FIRST run.
=============================================================================
EOF
