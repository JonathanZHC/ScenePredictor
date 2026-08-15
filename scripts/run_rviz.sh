#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONTAINER="${CONTAINER:-scenepredictor}"
MODE="${1:-predictor}"

# Default to the Intel Arrow Lake iGPU for RViz rendering. Set RVIZ_GPU=nvidia
# to deliberately use the RTX again. DRI_PRIME uses Mesa's PCI selector syntax.
RVIZ_GPU="${RVIZ_GPU:-intel}"
RVIZ_DRI_PRIME="${RVIZ_DRI_PRIME:-pci-0000_00_02_0}"

case "${MODE}" in
  predictor)
    CONFIG=/workspace/rviz/scene_pred_pipeline.rviz
    ;;
  tracker)
    CONFIG=/workspace/MultiViewRGBDTracker/rviz/tracking.rviz
    ;;
  isaac)
    CONFIG=/workspace/isaacscene/isaacscene.rviz
    ;;
  /*|*.rviz)
    CONFIG="${MODE}"
    ;;
  *)
    echo "usage: $0 [predictor|tracker|isaac|/absolute/path/to/file.rviz]" >&2
    exit 2
    ;;
esac

"${REPO_ROOT}/scripts/launch.sh"

GPU_ENV=()
case "${RVIZ_GPU}" in
  intel)
    INTEL_RENDER="$(docker exec "${CONTAINER}" printenv SCENEPRED_INTEL_RENDER 2>/dev/null || true)"
    if [[ -z "${INTEL_RENDER}" ]] || ! docker exec "${CONTAINER}" test -e "${INTEL_RENDER}"; then
      echo "[FAIL] Intel iGPU is not exposed inside ${CONTAINER}." >&2
      echo "       Recreate the container with the updated scripts/launch.sh." >&2
      exit 1
    fi
    GPU_ENV=(
      -e __GLX_VENDOR_LIBRARY_NAME=mesa
      -e "DRI_PRIME=${RVIZ_DRI_PRIME}"
    )
    echo "[INFO] RViz renderer: Intel iGPU (${RVIZ_DRI_PRIME}, ${INTEL_RENDER})"
    ;;
  nvidia)
    GPU_ENV=(
      -e __GLX_VENDOR_LIBRARY_NAME=nvidia
      -e __NV_PRIME_RENDER_OFFLOAD=1
    )
    echo "[INFO] RViz renderer: NVIDIA"
    ;;
  *)
    echo "[FAIL] RVIZ_GPU must be 'intel' or 'nvidia' (got '${RVIZ_GPU}')" >&2
    exit 2
    ;;
esac

# Verify the renderer selected inside the exact container/X11 environment that
# RViz will use. This runs only once at RViz startup and does not affect timing.
echo "[INFO] OpenGL renderer check:"
docker exec "${GPU_ENV[@]}" "${CONTAINER}" bash -lc '
  glxinfo -B 2>/dev/null | grep -E "OpenGL vendor|OpenGL renderer" || true
'

docker exec -it "${GPU_ENV[@]}" "${CONTAINER}" bash -lc "
  export QT_X11_NO_MITSHM=1
  export XDG_RUNTIME_DIR=/tmp/runtime-1234
  export XDG_CACHE_HOME=/tmp/rviz-cache
  export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
  export FASTDDS_BUILTIN_TRANSPORTS='LARGE_DATA?max_msg_size=4MB&sockets_size=8MB&non_blocking=true&tcp_negotiation_timeout=50'
  mkdir -p \"\${XDG_RUNTIME_DIR}\" \"\${XDG_CACHE_HOME}\"
  chmod 700 \"\${XDG_RUNTIME_DIR}\" 2>/dev/null || true
  source /opt/ros/jazzy/setup.bash
  exec rviz2 -d '${CONFIG}'
"
