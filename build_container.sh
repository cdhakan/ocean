#!/usr/bin/env bash

set -euo pipefail

DOCKERHUB_USER="${1:-}"
IMAGE_NAME="ocean-hpc"
TAG="latest"

if [[ -z "$DOCKERHUB_USER" ]]; then
    echo "Usage: ./build_container.sh <your_dockerhub_username>"
    echo "Example: ./build_container.sh mylab"
    exit 1
fi

FULL_IMAGE="${DOCKERHUB_USER}/${IMAGE_NAME}:${TAG}"

echo "============================================================"
echo "  OCEAN HPC Container Build"
echo "  Image: ${FULL_IMAGE}"
echo "============================================================"

echo ""
echo "[1/3] Building Docker image..."
docker build \
    --platform linux/amd64 \
    -f Dockerfile.hpc \
    -t "${FULL_IMAGE}" \
    .

echo "Build complete: ${FULL_IMAGE}"

echo ""
echo "[2/3] Pushing to DockerHub..."
echo "      (If not logged in, run: docker login)"
docker push "${FULL_IMAGE}"
echo "Pushed: https://hub.docker.com/r/${DOCKERHUB_USER}/${IMAGE_NAME}"

echo ""
SIF_NAME="${IMAGE_NAME}_${TAG}.sif"

if command -v singularity &>/dev/null; then
    echo "[3/3] Converting to Singularity SIF: ${SIF_NAME}"
    singularity pull --force "${SIF_NAME}" "docker://${FULL_IMAGE}"
    echo "SIF created: ${SIF_NAME}  ($(du -sh ${SIF_NAME} | cut -f1))"
elif command -v apptainer &>/dev/null; then
    echo "[3/3] Converting to Apptainer SIF: ${SIF_NAME}"
    apptainer pull --force "${SIF_NAME}" "docker://${FULL_IMAGE}"
    echo "SIF created: ${SIF_NAME}  ($(du -sh ${SIF_NAME} | cut -f1))"
else
    echo "[3/3] Singularity/Apptainer not found locally — skipping .sif conversion."
    echo "      To get the .sif on your HPC, run this on the cluster:"
    echo ""
    echo "        singularity pull docker://${FULL_IMAGE}"
    echo "        # → saves as ${SIF_NAME}"
fi

echo ""
echo "============================================================"
echo "  Done!  Share this pull command with your HPC users:"
echo ""
echo "    singularity pull docker://${FULL_IMAGE}"
echo ""
echo "  Or upload ${SIF_NAME} directly to shared cluster storage."
echo "============================================================"
