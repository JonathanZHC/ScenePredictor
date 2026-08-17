#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONTAINER="${CONTAINER:-scenepredictor}"

for path in DifFlow3D MultiViewRGBDTracker isaacscene; do
  [[ -e "${REPO_ROOT}/${path}" ]] || { echo "[FAIL] missing ${path}" >&2; exit 1; }
done

"${REPO_ROOT}/scripts/launch.sh"

echo "[1/6] GPU"
docker exec "${CONTAINER}" nvidia-smi --query-gpu=name,driver_version --format=csv,noheader

echo "[2/6] ROS 2 Jazzy"
docker exec "${CONTAINER}" bash -lc 'source /opt/ros/jazzy/setup.bash && ros2 --help >/dev/null && python3 -c "import rclpy; print(rclpy.__file__)"'

echo "[3/6] Isaac Python isolation + Warp"
docker exec "${CONTAINER}" bash -lc '/isaac-sim/python.sh - <<'"'"'PY'"'"'
import sys, numpy, torch, warp
print("python:", sys.executable)
print("numpy:", numpy.__version__)
print("torch:", torch.__version__, torch.version.cuda)
print("warp:", warp.__version__)
assert warp.__version__ == "1.15.0"
assert "/opt/tracking-venv" not in "\n".join(sys.path)
print("[OK] Isaac environment isolated")
PY'

echo "[4/6] Tracker + tracking Warp"
docker exec "${CONTAINER}" bash -lc 'source /opt/ros/jazzy/setup.bash && cd /workspace/MultiViewRGBDTracker && /opt/tracking-venv/bin/python - <<'"'"'PY'"'"'
import torch, warp, sam3, efficient_track_anything, sam_rgbd_tracking
from sam_rgbd_tracking import MultiViewEfficientTAMComponent
print("torch:", torch.__version__, torch.version.cuda)
print("warp:", warp.__version__)
print("tracker:", sam_rgbd_tracking.__file__)
assert warp.__version__ == "1.15.0"
print("component:", MultiViewEfficientTAMComponent)
print("[OK] tracker")
PY'

echo "[5/6] DifFlow3D + ABI-matched PointNet++"
docker exec "${CONTAINER}" bash -lc 'source /opt/ros/jazzy/setup.bash && cd /workspace && /opt/tracking-venv/bin/python - <<'"'"'PY'"'"'
from pathlib import Path
import torch
import difflow3d
from difflow3d.runtime import DifFlow3DStreamingCudaGraphRunner, SoftmaxAnchorMotionRecoverer
from difflow3d.ops.pointnet2 import pointnet2_utils

required = (
    "gaussian_softmax_recovery_wrapper",
    "gaussian_recovery_hash_build_wrapper",
    "gaussian_softmax_recovery_local_wrapper",
    "gaussian_softmax_recovery_local_track_aware_wrapper",
)
loaded = Path(pointnet2_utils.extension_path()).resolve()
missing = [name for name in required if not hasattr(pointnet2_utils.pointnet2, name)]
print("torch:", torch.__version__, torch.version.cuda)
print("DifFlow3D:", difflow3d.__file__)
print("PointNet++:", loaded)
if not str(loaded).startswith("/opt/DifFlow3D/difflow3d/ops/pointnet2/"):
    raise RuntimeError(f"Unexpected PointNet2 extension: {loaded}")
if missing:
    raise RuntimeError(f"Missing recovery CUDA symbols: {missing}")
print("[OK] DifFlow3D")
PY'

echo "[6/6] Source layout"
docker exec "${CONTAINER}" bash -lc '
set -e
for path in \
  /workspace/isaacscene/run_isaacsim.py \
  /workspace/MultiViewRGBDTracker/sam_rgbd_tracking/__init__.py \
  /workspace/DifFlow3D/difflow3d/runtime/runner.py \
  /workspace/scripts/run_scene_pred_pipeline.py; do
  test -e "$path" || { echo "[FAIL] missing $path" >&2; exit 1; }
  echo "[OK] $path"
done
'

echo "[OK] verification complete"
