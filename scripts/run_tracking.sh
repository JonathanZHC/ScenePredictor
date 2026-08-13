#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONTAINER="${CONTAINER:-scenepredictor}"
EXECUTION_MODE=""

if [[ $# -gt 0 ]]; then
  case "$1" in
    sequential|fixed_batch)
      EXECUTION_MODE="$1"
      shift
      ;;
  esac
fi

"${REPO_ROOT}/scripts/launch.sh"

NODE_ARGS=(--config /workspace/MultiViewRGBDTracker/configs/tracking.yaml)
if [[ -n "${EXECUTION_MODE}" ]]; then
  NODE_ARGS+=(--efficient-tam-execution-mode "${EXECUTION_MODE}")
fi
NODE_ARGS+=("$@")
printf -v NODE_ARGS_Q ' %q' "${NODE_ARGS[@]}"

docker exec -it "${CONTAINER}" bash -lc "
  source /opt/ros/jazzy/setup.bash
  export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
  export FASTDDS_BUILTIN_TRANSPORTS='LARGE_DATA?max_msg_size=4MB&sockets_size=8MB&non_blocking=true&tcp_negotiation_timeout=50'

  # The tracker config currently contains paths relative to its repository root.
  cd /workspace/MultiViewRGBDTracker
  exec /opt/tracking-venv/bin/python -m sam_rgbd_tracking.ros_node${NODE_ARGS_Q}
"
