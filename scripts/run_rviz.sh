#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONTAINER="${CONTAINER:-scenepredictor}"
MODE="${1:-predictor}"

case "${MODE}" in
  predictor)
    CONFIG=/workspace/scene_pred_pipeline.rviz
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

docker exec -it "${CONTAINER}" bash -lc "
  export __GLX_VENDOR_LIBRARY_NAME=nvidia
  export __NV_PRIME_RENDER_OFFLOAD=1
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
