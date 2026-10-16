#!/usr/bin/env bash
# =============================================================================
# install_niftyreg.sh — install NiftyReg for OCEAN's mouse brain extraction.
# =============================================================================
# NiftyReg ships as CLI binaries (reg_aladin, reg_resample, …), NOT a Python
# package — OCEAN calls them via subprocess, so they only need to be findable,
# not imported into OCEAN's Python env.  This installs them into a DEDICATED
# conda env (clean, isolated), then prints how OCEAN finds them.
#
#   >>> You do NOT need to `conda activate` anything to USE this. <<<
#   OCEAN auto-discovers reg_aladin across your conda envs at runtime.
#
# Usage:
#   bash install_niftyreg.sh                 # dedicated env  'niftyreg'  (recommended)
#   bash install_niftyreg.sh --into-base     # install into OCEAN's base env instead
#   bash install_niftyreg.sh --env myenv     # install into an existing/other env
# =============================================================================
set -euo pipefail

ENV_NAME="niftyreg"
MODE="dedicated"

while [ $# -gt 0 ]; do
  case "$1" in
    --into-base) MODE="base"; shift ;;
    --env)       MODE="named"; ENV_NAME="${2:?--env needs a name}"; shift 2 ;;
    -h|--help)   sed -n '2,20p' "$0"; exit 0 ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done

# ---- locate a conda ----------------------------------------------------------
# Prefer the conda that owns OCEAN's runtime (miniforge3_default) so the new env
# sits alongside it; fall back to whatever `conda` is on PATH.
CONDA=""
for c in /Users/cbd/miniforge3_default/bin/conda \
         /Users/cbd/miniforge3/bin/conda \
         "$(command -v conda 2>/dev/null || true)"; do
  if [ -n "$c" ] && [ -x "$c" ]; then CONDA="$c"; break; fi
done
if [ -z "$CONDA" ]; then
  echo "ERROR: conda not found. Install miniforge/conda first." >&2
  exit 1
fi
echo "Using conda: $CONDA"

# ---- install -----------------------------------------------------------------
# niftyreg 1.5.69.x is on conda-forge as a NATIVE osx-arm64 binary (BSD-3).
case "$MODE" in
  dedicated|named)
    echo "Installing niftyreg into env '$ENV_NAME' from conda-forge…"
    if "$CONDA" env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
      "$CONDA" install  -n "$ENV_NAME" -c conda-forge niftyreg -y
    else
      "$CONDA" create   -n "$ENV_NAME" -c conda-forge niftyreg -y
    fi
    BIN="$("$CONDA" run -n "$ENV_NAME" python -c 'import shutil,os;print(os.path.dirname(shutil.which("reg_aladin") or ""))')"
    ;;
  base)
    echo "Installing niftyreg into the BASE env from conda-forge…"
    "$CONDA" install -n base -c conda-forge niftyreg -y
    BIN="$("$CONDA" run -n base python -c 'import shutil,os;print(os.path.dirname(shutil.which("reg_aladin") or ""))')"
    ;;
esac
[ -z "$BIN" ] && { echo "ERROR: reg_aladin not found after install." >&2; exit 1; }

# ---- verify ------------------------------------------------------------------
echo
echo "Verifying reg_aladin runs natively…"
"$BIN/reg_aladin" --version 2>/dev/null | head -1 || "$BIN/reg_aladin" -h 2>&1 | head -1 || true
file "$BIN/reg_aladin" | sed 's/^/  arch: /'

# ---- tell the user how OCEAN finds it ---------------------------------------
cat <<EOF

=============================================================================
 NiftyReg installed.  Binaries in:
   $BIN
   ($(ls "$BIN" | grep -c '^reg_') reg_* tools, incl. reg_aladin, reg_resample)

 OCEAN finds reg_aladin automatically (it scans your conda envs) — you do NOT
 need to activate '$ENV_NAME'.  If auto-discovery ever misses it, pin it:

   export OCEAN_NIFTYREG_BIN="$BIN"

 Mouse brain extraction also needs ONE in-vivo mouse atlas (template + mask),
 which is ~8 MB and not bundled.  Point OCEAN at it with:

   export OCEAN_MOUSE_ATLAS_DIR="/path/to/atlas"   # folder w/ template/ + mask/

 (Get a starter atlas: install_niftyreg.sh does NOT download data; see
  fetch_mouse_atlas.sh — run it separately when you're ready.)
=============================================================================
EOF
