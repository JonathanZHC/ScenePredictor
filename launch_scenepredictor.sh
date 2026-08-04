#!/usr/bin/env bash
set -euo pipefail

IMAGE="${IMAGE:-scenepredictor}"
CONTAINER="${CONTAINER:-scenepredictor}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CACHE_ROOT="${REPO_ROOT}/.container-cache"
DRI_RENDER_NODE="$(
    find /dev/dri -maxdepth 1 -name 'renderD*' |
    sort |
    head -n 1
)"
DRI_RENDER_GID="$(stat -c '%g' "${DRI_RENDER_NODE}")"

if [[ ! -f "${REPO_ROOT}/.gitmodules" ]]; then
    echo "Run this script from the ScenePredictor repository root." >&2
    exit 1
fi

mkdir -p \
    "${CACHE_ROOT}/kit" \
    "${CACHE_ROOT}/ov" \
    "${CACHE_ROOT}/warp" \
    "${CACHE_ROOT}/matplotlib" \
    "${CACHE_ROOT}/compute" \
    "${CACHE_ROOT}/ultralytics" \
    "${CACHE_ROOT}/huggingface" \
    "${CACHE_ROOT}/torch" \
    "${CACHE_ROOT}/logs" \
    "${CACHE_ROOT}/config" \
    "${CACHE_ROOT}/data" \
    "${REPO_ROOT}/camera_output"

# The image runs as UID/GID 1234. Cache/output mounts must be writable.
if command -v sudo >/dev/null 2>&1; then
    sudo chown -R 1234:1234 "${CACHE_ROOT}" "${REPO_ROOT}/camera_output"
else
    echo "sudo is unavailable; ensure cache/output directories are writable by UID 1234." >&2
fi

if docker ps -a --format '{{.Names}}' | grep -qx "${CONTAINER}"; then
    docker rm -f "${CONTAINER}" >/dev/null
fi

docker run -d \
    --name "${CONTAINER}" \
    --gpus all \
    --device /dev/dri:/dev/dri \
    --group-add "${DRI_RENDER_GID}" \
    --network host \
    --ipc host \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    -e DISPLAY="${DISPLAY:-:0}" \
    -e XDG_RUNTIME_DIR=/tmp/runtime-isaac-sim \
    -e QT_X11_NO_MITSHM=1 \
    -e NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics,display \
    -e __NV_PRIME_RENDER_OFFLOAD=1 \
    -e __GLX_VENDOR_LIBRARY_NAME=nvidia \
    -e __VK_LAYER_NV_optimus=NVIDIA_only \
    -e ACCEPT_EULA=Y \
    -e PRIVACY_CONSENT=Y \
    -e ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-117}" \
    -e RMW_IMPLEMENTATION=rmw_fastrtps_cpp \
    -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
    -v "${REPO_ROOT}:/workspace:rw" \
    -v "${CACHE_ROOT}/kit:/isaac-sim/kit/cache:rw" \
    -v "${CACHE_ROOT}/ov:/isaac-sim/.cache/ov:rw" \
    -v "${CACHE_ROOT}/warp:/isaac-sim/.cache/warp:rw" \
    -v "${CACHE_ROOT}/matplotlib:/isaac-sim/.cache/matplotlib:rw" \
    -v "${CACHE_ROOT}/compute:/isaac-sim/.nv/ComputeCache:rw" \
    -v "${CACHE_ROOT}/ultralytics:/isaac-sim/.config/Ultralytics:rw" \
    -v "${CACHE_ROOT}/huggingface:/isaac-sim/.cache/huggingface:rw" \
    -v "${CACHE_ROOT}/torch:/isaac-sim/.cache/torch:rw" \
    -v "${CACHE_ROOT}/logs:/isaac-sim/.nvidia-omniverse/logs:rw" \
    -v "${CACHE_ROOT}/config:/isaac-sim/.nvidia-omniverse/config:rw" \
    -v "${CACHE_ROOT}/data:/isaac-sim/.local/share/ov/data:rw" \
    "${IMAGE}" \
    sleep infinity

echo "Started container: ${CONTAINER}"
echo
echo "GPU/runtime check:"
echo "  docker exec -it ${CONTAINER} bash -lc 'source /opt/ros/jazzy/setup.bash && /isaac-sim/python.sh -c \"import torch, open_clip, pointnet2_cuda; print(torch.__version__, torch.cuda.is_available())\"'"
echo
echo "Start Isaac Sim dynamic scene:"
echo "  docker exec -it ${CONTAINER} bash -lc 'source /opt/ros/jazzy/setup.bash && /isaac-sim/python.sh /workspace/isaacscene/run_isaacsim.py --scene dynamic --width 640 --height 480 --rgbd-hz 30 --pointcloud-hz 2 --motion-speed-scale 1.0'"
echo
echo "Start RViz in another terminal:"
echo "  docker exec -it ${CONTAINER} bash -lc 'source /opt/ros/jazzy/setup.bash && rviz2 -d /workspace/isaacscene/isaacscene.rviz'"
