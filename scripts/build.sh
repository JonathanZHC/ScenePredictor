#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
IMAGE_NAME="${IMAGE_NAME:-scenepredictor:latest}"
DOCKERFILE="${REPO_ROOT}/Dockerfile"

required=(
  "${REPO_ROOT}/DifFlow3D/pointnet2/setup.py"
  "${REPO_ROOT}/MultiViewRGBDTracker/sam_rgbd_tracking/__init__.py"
  "${REPO_ROOT}/isaacscene/run_isaacsim.py"
)
for path in "${required[@]}"; do
  if [[ ! -e "${path}" ]]; then
    echo "[ERROR] missing dependency: ${path}" >&2
    echo "Initialize only the top-level submodules:" >&2
    echo "  git submodule update --init DifFlow3D MultiViewRGBDTracker isaacscene" >&2
    exit 1
  fi
done

if [[ ! -f "${DOCKERFILE}" ]]; then
  echo "[ERROR] Dockerfile not found: ${DOCKERFILE}" >&2
  exit 1
fi

echo "[BUILD] ${IMAGE_NAME}"
echo "  dockerfile: ${DOCKERFILE}"
echo "  context:    ${REPO_ROOT}"

docker build \
  --progress=plain \
  -f "${DOCKERFILE}" \
  -t "${IMAGE_NAME}" \
  "${REPO_ROOT}"
