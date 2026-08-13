#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE="${IMAGE:-sam-rgbd-tracking:latest}"
CONTAINER="${CONTAINER:-sam-rgbd-tracking}"
CACHE_ROOT="${REPO_ROOT}/.container-cache"

if [[ "$(docker inspect -f '{{.State.Running}}' "${CONTAINER}" 2>/dev/null || true)" == "true" ]]; then
  echo "[OK] container ${CONTAINER} is already running"
  exit 0
fi

docker rm -f "${CONTAINER}" >/dev/null 2>&1 || true

mkdir -p \
  "${CACHE_ROOT}/kit" \
  "${CACHE_ROOT}/ov" \
  "${CACHE_ROOT}/warp" \
  "${CACHE_ROOT}/matplotlib" \
  "${CACHE_ROOT}/huggingface" \
  "${CACHE_ROOT}/torch" \
  "${CACHE_ROOT}/compute" \
  "${CACHE_ROOT}/logs" \
  "${CACHE_ROOT}/config" \
  "${CACHE_ROOT}/data" \
  "${REPO_ROOT}/logs"

# The Isaac image runs as UID/GID 1234. The repo stays owned by the host user;
# only cache/output directories are made world-writable for the test branch.
chmod -R a+rwX "${CACHE_ROOT}" "${REPO_ROOT}/logs" 2>/dev/null || true

xhost +local:docker >/dev/null 2>&1 || true

DOCKER_ARGS=(
  -d
  --name "${CONTAINER}"
  --gpus all
  --network host
  --ipc host
  --shm-size=16g
  --ulimit memlock=-1
  --ulimit stack=67108864
  -e "DISPLAY=${DISPLAY:-:0}"
  -e QT_X11_NO_MITSHM=1
  -e ACCEPT_EULA=Y
  -e PRIVACY_CONSENT=Y
  -e "ROS_DOMAIN_ID=${ROS_DOMAIN_ID:-117}"
  -e RMW_IMPLEMENTATION=rmw_fastrtps_cpp
  -e NVIDIA_VISIBLE_DEVICES=all
  -e NVIDIA_DRIVER_CAPABILITIES=all
  -e XDG_RUNTIME_DIR=/tmp/runtime-1234
  -v /tmp/.X11-unix:/tmp/.X11-unix:rw
  -v "${REPO_ROOT}:/workspace:rw"
  -v "${CACHE_ROOT}/kit:/isaac-sim/kit/cache:rw"
  -v "${CACHE_ROOT}/ov:/isaac-sim/.cache/ov:rw"
  -v "${CACHE_ROOT}/warp:/isaac-sim/.cache/warp:rw"
  -v "${CACHE_ROOT}/matplotlib:/isaac-sim/.cache/matplotlib:rw"
  -v "${CACHE_ROOT}/huggingface:/isaac-sim/.cache/huggingface:rw"
  -v "${CACHE_ROOT}/torch:/isaac-sim/.cache/torch:rw"
  -v "${CACHE_ROOT}/compute:/isaac-sim/.nv/ComputeCache:rw"
  -v "${CACHE_ROOT}/logs:/isaac-sim/.nvidia-omniverse/logs:rw"
  -v "${CACHE_ROOT}/config:/isaac-sim/.nvidia-omniverse/config:rw"
  -v "${CACHE_ROOT}/data:/isaac-sim/.local/share/ov/data:rw"
)

CID="$(docker run "${DOCKER_ARGS[@]}" "${IMAGE}" sleep infinity)"
sleep 0.5
if [[ "$(docker inspect -f '{{.State.Running}}' "${CONTAINER}" 2>/dev/null || true)" != "true" ]]; then
  echo "[FAIL] container exited during startup" >&2
  docker logs "${CONTAINER}" >&2 || true
  exit 1
fi

echo "[OK] started container ${CONTAINER} (${CID:0:12})"
