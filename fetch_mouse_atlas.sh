#!/usr/bin/env bash
# =============================================================================
# fetch_mouse_atlas.sh — download ONE in-vivo mouse-brain atlas for OCEAN.
# =============================================================================
# OCEAN's mouse brain extraction (MASMAT-style, NiftyReg-only) needs a mouse
# atlas = a template image + its brain mask.  This grabs just the small in-vivo
# set (~8 MB) from dama-lab/mouse-brain-atlas using a sparse git checkout, so
# you do NOT clone the full ~2.4 GB repo and do NOT need Subversion.
#
# Atlas: dama-lab/mouse-brain-atlas  (atlas GPL-3.0; cite the NeAt papers if you
# use the NeAt set).  Default here is FVB_NCrl/in_vivo.
#
# Usage:
#   bash fetch_mouse_atlas.sh                      # -> ./mouse_atlas/FVB_NCrl/in_vivo
#   bash fetch_mouse_atlas.sh /custom/dest         # custom destination dir
#   ATLAS_SUBDIR=NeAt/in_vivo bash fetch_mouse_atlas.sh   # a different atlas set
# =============================================================================
set -euo pipefail

DEST="${1:-$(pwd)/mouse_atlas}"
REPO="https://github.com/dama-lab/mouse-brain-atlas.git"
SUBDIR="${ATLAS_SUBDIR:-FVB_NCrl/in_vivo}"

command -v git >/dev/null || { echo "ERROR: git not found." >&2; exit 1; }

echo "Fetching atlas '$SUBDIR' (sparse, blobless) into: $DEST"
mkdir -p "$DEST"

if [ ! -d "$DEST/.git" ]; then
  git clone --filter=blob:none --no-checkout --depth 1 "$REPO" "$DEST"
fi
cd "$DEST"
git sparse-checkout init --cone
git sparse-checkout set "$SUBDIR"
git checkout 2>/dev/null || git checkout HEAD

ATLAS_DIR="$DEST/$SUBDIR"
echo
if [ -d "$ATLAS_DIR/template" ] && [ -d "$ATLAS_DIR/mask" ]; then
  echo "OK — atlas ready:"
  echo "  template/ : $(ls "$ATLAS_DIR/template" | wc -l | tr -d ' ') file(s)"
  echo "  mask/     : $(ls "$ATLAS_DIR/mask"     | wc -l | tr -d ' ') file(s)"
else
  echo "NOTE: expected template/ and mask/ subfolders under $ATLAS_DIR"
  echo "      contents:"; ls -1 "$ATLAS_DIR" 2>/dev/null | sed 's/^/        /'
fi

cat <<EOF

=============================================================================
 Point OCEAN at this atlas (add to your shell profile to make it permanent):

   export OCEAN_MOUSE_ATLAS_DIR="$ATLAS_DIR"

 Then (with NiftyReg installed — see install_niftyreg.sh) the
 "Detect Mice Brain Outline" button will run single-atlas skull-stripping.
=============================================================================
EOF
