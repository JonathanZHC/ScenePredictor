#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

CONTAINER="${CONTAINER:-scenepredictor}"
ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-117}"

if [[ $# -lt 1 ]]; then
  echo "usage: $0 <bag_path> [extra ros2 bag play args...]" >&2
  echo
  echo "examples:"
  echo "  $0 /workspace/rosbags/dynamic_test"
  echo "  $0 /workspace/rosbags/dynamic_test --loop"
  echo "  $0 /workspace/rosbags/dynamic_test --rate 0.5 --loop"
  exit 2
fi

BAG_PATH="$1"
shift

# Start the container if it is not already running.
if ! docker ps --format '{{.Names}}' | grep -qx "${CONTAINER}"; then
  echo "[rosbag] Container '${CONTAINER}' is not running."
  echo "[rosbag] Starting it..."
  "${REPO_ROOT}/scripts/launch.sh"
fi

echo "[rosbag] container : ${CONTAINER}"
echo "[rosbag] bag       : ${BAG_PATH}"
echo "[rosbag] domain id : ${ROS_DOMAIN_ID}"

docker exec -it \
  -e ROS_DOMAIN_ID="${ROS_DOMAIN_ID}" \
  -e RMW_IMPLEMENTATION=rmw_fastrtps_cpp \
  -e 'FASTDDS_BUILTIN_TRANSPORTS=LARGE_DATA?max_msg_size=4MB&sockets_size=8MB&non_blocking=true&tcp_negotiation_timeout=50' \
  "${CONTAINER}" \
  bash -lc '
    # ROS setup must be sourced before enabling nounset (-u).
    source /opt/ros/jazzy/setup.bash
    set -euo pipefail

    BAG_PATH="$1"
    shift

    if [[ ! -e "${BAG_PATH}" ]]; then
      echo "[rosbag] ERROR: bag not found inside container: ${BAG_PATH}" >&2
      exit 1
    fi

    echo "[rosbag] Playing ${BAG_PATH}"
    exec ros2 bag play "${BAG_PATH}" "$@"
  ' bash "${BAG_PATH}" "$@"