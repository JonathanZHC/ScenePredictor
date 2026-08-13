#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONTAINER="${CONTAINER:-scenepredictor}"

for path in DifFlow3D MultiViewRGBDTracker isaacscene; do
  if [[ ! -e "${REPO_ROOT}/${path}" ]]; then
    echo "[FAIL] missing top-level dependency: ${path}" >&2
    exit 1
  fi
done

# Nested tracker isaacscene is intentionally not required.
if git -C "${REPO_ROOT}/MultiViewRGBDTracker" submodule status isaacscene >/tmp/scenepredictor_nested_submodule 2>/dev/null; then
  nested="$(cat /tmp/scenepredictor_nested_submodule)"
  if [[ "${nested}" == -* ]]; then
    echo "[OK] nested MultiViewRGBDTracker/isaacscene is not initialized"
  else
    echo "[WARN] nested MultiViewRGBDTracker/isaacscene is initialized; ScenePredictor does not use it"
  fi
fi

"${REPO_ROOT}/scripts/launch.sh"

echo "[1/6] GPU"
docker exec "${CONTAINER}" nvidia-smi --query-gpu=name,driver_version --format=csv,noheader

echo "[2/6] ROS 2 Jazzy"
docker exec "${CONTAINER}" bash -lc 'source /opt/ros/jazzy/setup.bash && ros2 --help >/dev/null && python3 -c "import rclpy; print(rclpy.__file__)"'

echo "[3/6] Isaac Python isolation"
docker exec "${CONTAINER}" bash -lc '/isaac-sim/python.sh - <<'"'"'PY'"'"'
import sys, numpy, torch, warp
print("python:", sys.executable)
print("numpy:", numpy.__version__)
print("torch:", torch.__version__, torch.version.cuda)
print("warp:", warp.__version__)
assert "/opt/tracking-venv" not in "\n".join(sys.path)
print("[OK] Isaac environment isolated")
PY'

echo "[4/6] Tracker source dependency"
docker exec "${CONTAINER}" bash -lc 'source /opt/ros/jazzy/setup.bash && cd /workspace/MultiViewRGBDTracker && /opt/tracking-venv/bin/python - <<'"'"'PY'"'"'
import torch
import sam3
import efficient_track_anything
import sam_rgbd_tracking
from sam_rgbd_tracking import MultiViewEfficientTAMComponent
print("torch:", torch.__version__, torch.version.cuda)
print("tracker:", sam_rgbd_tracking.__file__)
print("component:", MultiViewEfficientTAMComponent)
print("[OK] tracker")
PY'

echo "[5/6] DifFlow3D + PointNet++"
docker exec "${CONTAINER}" bash -lc 'source /opt/ros/jazzy/setup.bash && cd /workspace && /opt/tracking-venv/bin/python - <<'"'"'PY'"'"'
import torch
import pointnet2_cuda
from pointnet2 import pointnet2_utils
import model_difflow
print("torch:", torch.__version__, torch.version.cuda)
print("pointnet2_cuda:", pointnet2_cuda.__file__)
print("DifFlow3D:", model_difflow.__file__)
print("[OK] scene flow")
PY'

echo "[6/6] Top-level source layout"
docker exec "${CONTAINER}" bash -lc '
set -e
for path in \
  /workspace/isaacscene/run_isaacsim.py \
  /workspace/MultiViewRGBDTracker/sam_rgbd_tracking/__init__.py \
  /workspace/DifFlow3D/model_difflow.py \
  /workspace/scripts/run_scene_pred_pipeline.py; do
  test -e "$path" || { echo "[FAIL] missing $path" >&2; exit 1; }
  echo "[OK] $path"
done
'

echo "[OK] verification complete"
