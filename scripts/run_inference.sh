#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

IMAGE="${IMAGE:-scenepredictor:latest}"
CONTAINER="${CONTAINER:-scenepredictor}"
CONFIG="${CONFIG:-/workspace/configs/default.yaml}"
ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-117}"
RMW_IMPLEMENTATION="${RMW_IMPLEMENTATION:-rmw_fastrtps_cpp}"
FASTDDS_BUILTIN_TRANSPORTS="${FASTDDS_BUILTIN_TRANSPORTS:-LARGE_DATA?max_msg_size=4MB&sockets_size=8MB&non_blocking=true&tcp_negotiation_timeout=50}"

IMAGE="${IMAGE}" \
CONTAINER="${CONTAINER}" \
"${REPO_ROOT}/scripts/launch.sh"

ARGS=(--config "${CONFIG}")
ARGS+=("$@")
printf -v ARGS_Q ' %q' "${ARGS[@]}"

docker exec -it \
  -e "ROS_DOMAIN_ID=${ROS_DOMAIN_ID}" \
  -e "RMW_IMPLEMENTATION=${RMW_IMPLEMENTATION}" \
  -e "FASTDDS_BUILTIN_TRANSPORTS=${FASTDDS_BUILTIN_TRANSPORTS}" \
  "${CONTAINER}" bash -lc "
  source /opt/ros/jazzy/setup.bash
  cd /workspace
  exec /opt/tracking-venv/bin/python \
    /workspace/scripts/run_scene_pred_pipeline.py${ARGS_Q}
"