#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONTAINER="${CONTAINER:-sam-rgbd-tracking}"
SCENE="${1:-dynamic}"
if [[ $# -gt 0 ]]; then shift; fi

case "${SCENE}" in
  static|dynamic|hybrid|occlusion) ;;
  *) echo "usage: $0 [static|dynamic|hybrid|occlusion] [extra run_isaacsim.py args...]" >&2; exit 2 ;;
esac


"${REPO_ROOT}/scripts/launch.sh"

docker exec -it "${CONTAINER}" bash -lc "
  source /opt/ros/jazzy/setup.bash

  export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
  export FASTDDS_BUILTIN_TRANSPORTS='LARGE_DATA?max_msg_size=4MB&sockets_size=8MB&non_blocking=true&tcp_negotiation_timeout=50'
  
  exec /isaac-sim/python.sh /workspace/isaacscene/run_isaacsim.py \\
    --scene '${SCENE}' \\
    --width 640 \\
    --height 480 \\
    --rgbd-hz 30 \\
    --pointcloud-hz 5 \\
    --corrupt \\
    --no-rgb-corruption \\
    --motion-speed-scale 1.0 \\
    $*
"
