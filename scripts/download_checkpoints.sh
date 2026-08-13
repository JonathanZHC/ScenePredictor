#!/usr/bin/env bash
set -euo pipefail

# Download runtime checkpoints for ScenePredictor.
#
# Files are stored in the top-level repository:
#   checkpoints/sam3.pt
#   checkpoints/efficienttam_s_512x512.pt
#
# Usage:
#   ./scripts/download_checkpoints.sh
#
# Optional:
#   HF_TOKEN=hf_xxx ./scripts/download_checkpoints.sh
#   FORCE=1 ./scripts/download_checkpoints.sh
#   CONTAINER_NAME=my-container ./scripts/download_checkpoints.sh

CONTAINER_NAME="${CONTAINER_NAME:-scenepredictor}"
FORCE="${FORCE:-0}"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
CHECKPOINT_DIR="${REPO_ROOT}/checkpoints"

SAM3_FILE="${CHECKPOINT_DIR}/sam3.pt"
EFFICIENTTAM_FILE="${CHECKPOINT_DIR}/efficienttam_s_512x512.pt"

SAM3_REPO="facebook/sam3"
SAM3_FILENAME="sam3.pt"

EFFICIENTTAM_URL="https://huggingface.co/yunyangx/efficient-track-anything/resolve/main/efficienttam_s_512x512.pt"

mkdir -p "${CHECKPOINT_DIR}"

if ! docker inspect "${CONTAINER_NAME}" >/dev/null 2>&1; then
    echo "[ERROR] Docker container '${CONTAINER_NAME}' does not exist."
    echo "        Start it first with:"
    echo "          ./scripts/launch.sh"
    exit 1
fi

if [[ "$(docker inspect -f '{{.State.Running}}' "${CONTAINER_NAME}")" != "true" ]]; then
    echo "[ERROR] Docker container '${CONTAINER_NAME}' is not running."
    echo "        Start it first with:"
    echo "          ./scripts/launch.sh"
    exit 1
fi

echo "[INFO] Repository: ${REPO_ROOT}"
echo "[INFO] Checkpoints: ${CHECKPOINT_DIR}"
echo

# ---------------------------------------------------------------------------
# EfficientTAM
# ---------------------------------------------------------------------------

if [[ -s "${EFFICIENTTAM_FILE}" && "${FORCE}" != "1" ]]; then
    echo "[OK] EfficientTAM checkpoint already exists:"
    echo "     ${EFFICIENTTAM_FILE}"
else
    echo "[INFO] Downloading EfficientTAM checkpoint..."

    rm -f "${EFFICIENTTAM_FILE}"

    docker exec "${CONTAINER_NAME}" \
        curl -L --fail --retry 3 --retry-delay 2 \
        -o /workspace/checkpoints/efficienttam_s_512x512.pt \
        "${EFFICIENTTAM_URL}"

    if [[ ! -s "${EFFICIENTTAM_FILE}" ]]; then
        echo "[ERROR] EfficientTAM download failed."
        exit 1
    fi

    echo "[OK] EfficientTAM checkpoint downloaded."
fi

echo

# ---------------------------------------------------------------------------
# SAM3
# ---------------------------------------------------------------------------

if [[ -s "${SAM3_FILE}" && "${FORCE}" != "1" ]]; then
    echo "[OK] SAM3 checkpoint already exists:"
    echo "     ${SAM3_FILE}"
else
    echo "[INFO] Downloading SAM3 checkpoint..."

    rm -f "${SAM3_FILE}"

    if [[ -n "${HF_TOKEN:-}" ]]; then
        echo "[INFO] Using HF_TOKEN from the host environment."

        docker exec \
            -e HF_TOKEN="${HF_TOKEN}" \
            "${CONTAINER_NAME}" \
            /opt/tracking-venv/bin/hf download \
            "${SAM3_REPO}" \
            "${SAM3_FILENAME}" \
            --local-dir /workspace/checkpoints
    else
        if ! docker exec "${CONTAINER_NAME}" \
            /opt/tracking-venv/bin/hf auth whoami >/dev/null 2>&1; then
            echo "[ERROR] SAM3 requires Hugging Face authentication."
            echo
            echo "Either log in once inside the container:"
            echo
            echo "  docker exec -it ${CONTAINER_NAME} \\"
            echo "    /opt/tracking-venv/bin/hf auth login"
            echo
            echo "or run this script with a token:"
            echo
            echo "  HF_TOKEN=hf_xxx ./scripts/download_checkpoints.sh"
            echo
            echo "Also make sure your Hugging Face account has access to:"
            echo "  ${SAM3_REPO}"
            exit 1
        fi

        docker exec "${CONTAINER_NAME}" \
            /opt/tracking-venv/bin/hf download \
            "${SAM3_REPO}" \
            "${SAM3_FILENAME}" \
            --local-dir /workspace/checkpoints
    fi

    if [[ ! -s "${SAM3_FILE}" ]]; then
        echo "[ERROR] SAM3 download failed."
        exit 1
    fi

    echo "[OK] SAM3 checkpoint downloaded."
fi

echo
echo "[OK] All required checkpoints are available:"
ls -lh "${SAM3_FILE}" "${EFFICIENTTAM_FILE}"
