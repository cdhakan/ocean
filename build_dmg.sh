#!/usr/bin/env bash

set -euo pipefail

APP_NAME="OCEAN"
APP_SAFE="OCEAN"
VERSION="1.0.0"
SPEC_FILE="OCEAN.spec"
DIST_DIR="dist"
DMG_NAME="${APP_SAFE}_v${VERSION}.dmg"
APP_BUNDLE="${DIST_DIR}/${APP_NAME}.app"

echo "═══════════════════════════════════════════════"
echo "  OCEAN  —  macOS build  v${VERSION}"
echo "═══════════════════════════════════════════════"

if [ -f "$HOME/miniforge3_default/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniforge3_default/etc/profile.d/conda.sh"
    conda activate cbdmrf
elif [ -f "$HOME/miniforge3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniforge3/etc/profile.d/conda.sh"
    conda activate cbdmrf
fi

ICON_SRC="icon.png"
ICONSET_DIR="assets/AppIcon.iconset"
ICNS_PATH="assets/AppIcon.icns"

if [ -f "${ICON_SRC}" ]; then
    echo ""
    echo "▶ Converting icon.png → ${ICNS_PATH}…"
    mkdir -p "${ICONSET_DIR}"
    for size in 16 32 64 128 256 512; do
        sips -z "${size}" "${size}" "${ICON_SRC}" \
            --out "${ICONSET_DIR}/icon_${size}x${size}.png" >/dev/null
        sips -z "$((size*2))" "$((size*2))" "${ICON_SRC}" \
            --out "${ICONSET_DIR}/icon_${size}x${size}@2x.png" >/dev/null
    done
    iconutil -c icns "${ICONSET_DIR}" -o "${ICNS_PATH}"
    echo "   ✓ Icon ready: ${ICNS_PATH}"
else
    echo "   (icon.png not found — building without custom icon)"
fi

echo ""
echo "▶ Cleaning previous build artefacts…"
rm -rf build/ \
       "${DIST_DIR}/OCEAN" \
       "${DIST_DIR}/${APP_NAME}.app" \
       "${DIST_DIR}/OCEAN.app" \
       "${DIST_DIR}/${DMG_NAME}" \
       __pycache__

echo ""
echo "▶ Running PyInstaller (this takes 1–3 minutes)…"
pyinstaller "${SPEC_FILE}" \
    --clean \
    --noconfirm \
    --log-level ERROR

if [ ! -d "${APP_BUNDLE}" ]; then
    echo "ERROR: PyInstaller did not produce ${APP_BUNDLE}"
    echo "  (check that BUNDLE name in spec matches '${APP_NAME}.app')"
    exit 1
fi
echo "   ✓ App bundle: ${APP_BUNDLE}"

echo ""
echo "▶ Verifying bundle structure…"
ls "${APP_BUNDLE}/Contents/MacOS/" | head -5
echo "   ✓ Bundle structure looks good."

echo ""
echo "▶ Creating DMG…"

BUNDLED_ICNS="${APP_BUNDLE}/Contents/Resources/AppIcon.icns"
if [ ! -f "${BUNDLED_ICNS}" ]; then
    BUNDLED_ICNS="${APP_BUNDLE}/Contents/Resources/icon-windowed.icns"
fi

VOLICON_ARGS=()
if [ -f "${BUNDLED_ICNS}" ]; then
    VOLICON_ARGS=(--volicon "${BUNDLED_ICNS}")
fi

if command -v create-dmg &>/dev/null; then

    create-dmg \
        --volname "OCEAN ${VERSION}" \
        "${VOLICON_ARGS[@]}" \
        --window-pos 200 120 \
        --window-size 660 420 \
        --icon-size 100 \
        --icon "${APP_NAME}.app" 180 200 \
        --hide-extension "${APP_NAME}.app" \
        --app-drop-link 460 200 \
        "${DIST_DIR}/${DMG_NAME}" \
        "${APP_BUNDLE}" \
    || {
        echo "   (re-trying without custom volume icon)"
        create-dmg \
            --volname "OCEAN ${VERSION}" \
            --window-pos 200 120 \
            --window-size 660 420 \
            --icon-size 100 \
            --icon "${APP_NAME}.app" 180 200 \
            --hide-extension "${APP_NAME}.app" \
            --app-drop-link 460 200 \
            "${DIST_DIR}/${DMG_NAME}" \
            "${APP_BUNDLE}"
    }

else
    echo "   (create-dmg not found — using hdiutil fallback)"
    TMP_DMG="${DIST_DIR}/tmp_${APP_SAFE}.dmg"
    STAGING=$(mktemp -d)
    cp -R "${APP_BUNDLE}" "${STAGING}/"
    ln -s /Applications "${STAGING}/Applications"
    hdiutil create \
        -volname "OCEAN ${VERSION}" \
        -srcfolder "${STAGING}" \
        -ov -format UDRW \
        "${TMP_DMG}"
    hdiutil convert "${TMP_DMG}" -format UDZO -imagekey zlib-level=9 \
        -o "${DIST_DIR}/${DMG_NAME}"
    rm -f "${TMP_DMG}"
    rm -rf "${STAGING}"
fi

echo ""
echo "═══════════════════════════════════════════════"
echo "  ✓  DONE!  Output:"
echo "     ${DIST_DIR}/${DMG_NAME}"
FILESIZE=$(du -sh "${DIST_DIR}/${DMG_NAME}" 2>/dev/null | cut -f1)
echo "     Size: ${FILESIZE}"
echo ""
echo "  To distribute: send this single .dmg file."
echo "  Recipients open it, drag the app"
echo "  into their /Applications folder, done."
echo "═══════════════════════════════════════════════"
