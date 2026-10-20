#!/usr/bin/env bash
# Set up the GE Orchestra SDK for OCEAN's "GE P-file (Orchestra)" format.
#
# The SDK's compiled Python module needs Python 3.10 and NumPy 1.x, so it gets
# its OWN conda env ("orchestra"); OCEAN's own environment (cbdmrf) and the
# CEST-MRF / CEST-MRI pipelines are not touched. OCEAN runs the reconstruction
# in that env as a separate process. The SDK itself is GE-licensed and is never
# copied or bundled - OCEAN only remembers where it is.
#
# Usage:   bash install_orchestra.sh [/path/to/orchestra-sdk-2.1-1.python]
set -euo pipefail

SDK="${1:-}"
if [ -z "$SDK" ]; then
    for d in "$HOME"/orchestra-sdk* "$HOME"/Downloads/orchestra-sdk*; do
        [ -d "$d" ] && SDK="$d" && break
    done
fi
if [ -z "$SDK" ] || [ ! -d "$SDK" ]; then
    echo "Orchestra SDK folder not found - pass it as the first argument"; exit 1
fi
FOUND_LIB=""
for lib in GERecon.so.mac GERecon.so GERecon.pyd GERecon.so.python36; do
    [ -f "$SDK/$lib" ] && FOUND_LIB="$lib" && break
done
if [ -z "$FOUND_LIB" ]; then
    echo "No GERecon library (GERecon.so.mac / GERecon.so / GERecon.pyd) in $SDK"; exit 1
fi

for c in "$HOME/miniforge3_default" "$HOME/miniforge3" "$HOME/miniconda3" "$HOME/anaconda3" "$HOME/mambaforge"; do
    if [ -f "$c/etc/profile.d/conda.sh" ]; then source "$c/etc/profile.d/conda.sh"; break; fi
done
command -v conda >/dev/null 2>&1 || { echo "conda not found - install Miniforge first"; exit 1; }

# GE built the macOS / Windows libraries for Python 3.10, the Linux one
# (GERecon.so.python36) for Python 3.6 — the env must match.
case "$(uname)" in
    Linux) PYVER=3.6 ;;
    *)     PYVER=3.10 ;;
esac
if ! conda env list | grep -qE "^orchestra[[:space:]]"; then
    echo "Creating conda env 'orchestra' (Python $PYVER, NumPy 1.x)..."
    conda create -y -n orchestra -c conda-forge "python=$PYVER" "numpy<2"
fi
PY="$(conda run -n orchestra python -c 'import sys; print(sys.executable)')"

# macOS marks a downloaded library as quarantined, which can make Python refuse
# to load it ("code signature ... not valid"). Clear that flag on the SDK library.
if [ "$(uname)" = "Darwin" ] && [ -f "$SDK/GERecon.so.mac" ]; then
    xattr -d com.apple.quarantine "$SDK/GERecon.so.mac" 2>/dev/null || true
fi

mkdir -p "$HOME/.ocean"
printf '{"python": "%s", "sdk": "%s"}\n' "$PY" "$SDK" > "$HOME/.ocean/orchestra.json"

"$PY" "$(dirname "$0")/my_gui/orchestra_worker.py" --check --sdk "$SDK"
echo "Orchestra ready for OCEAN."
echo "  Python env : $PY"
echo "  SDK folder : $SDK"
echo "  (remembered in $HOME/.ocean/orchestra.json)"
