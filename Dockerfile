# syntax=docker/dockerfile:1.7
# ScenePredictor runtime:
#   - Isaac Sim / sensors:              /isaac-sim/python.sh
#   - Tracker + ScenePredictor + flow:  /opt/tracking-venv/bin/python
#
# Build context must be the ScenePredictor repository root. The following
# top-level submodules must be initialized before building:
#   DifFlow3D/
#   MultiViewRGBDTracker/
#   isaacscene/
#
# Do NOT recursively initialize MultiViewRGBDTracker/isaacscene; ScenePredictor
# uses the top-level /workspace/isaacscene checkout.

ARG ISAAC_SIM_IMAGE=nvcr.io/nvidia/isaac-sim:6.0.1
ARG CUDA_BUILDER_IMAGE=nvidia/cuda:12.8.1-cudnn-devel-ubuntu24.04

ARG ISAAC_TORCH_VERSION=2.11.0
ARG ISAAC_TORCHVISION_VERSION=0.26.0
ARG TRACKING_TORCH_VERSION=2.8.0
ARG TRACKING_TORCHVISION_VERSION=0.23.0
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cu128
ARG TORCH_CUDA_ARCH_LIST=12.0
ARG WARP_VERSION=1.15.0

ARG SAM3_REPOSITORY=https://github.com/facebookresearch/sam3.git
ARG SAM3_REF=main
ARG EFFICIENT_TAM_REPOSITORY=https://github.com/JonathanZHC/EfficientTAM.git
ARG EFFICIENT_TAM_REF=main

# =============================================================================
# Stage 1: build DifFlow3D PointNet++ against the SAME Torch/CUDA ABI used by
#          the final tracking/inference venv.
# =============================================================================
FROM ${CUDA_BUILDER_IMAGE} AS difflow-builder

ARG DEBIAN_FRONTEND=noninteractive
ARG TRACKING_TORCH_VERSION
ARG TORCH_INDEX_URL
ARG TORCH_CUDA_ARCH_LIST

SHELL ["/bin/bash", "-o", "pipefail", "-c"]
ENV LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    CUDA_HOME=/usr/local/cuda \
    TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST} \
    FORCE_CUDA=1 \
    MAX_JOBS=4

RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential ca-certificates cmake ninja-build \
      python3 python3-dev python3.12 python3.12-dev python3.12-venv \
    && rm -rf /var/lib/apt/lists/*

RUN python3.12 -m venv /opt/difflow-build-venv \
    && /opt/difflow-build-venv/bin/python -m pip install --upgrade pip setuptools wheel \
    && /opt/difflow-build-venv/bin/python -m pip install \
      torch==${TRACKING_TORCH_VERSION} \
      --index-url ${TORCH_INDEX_URL} \
    && /opt/difflow-build-venv/bin/python -m pip install numpy==1.26.4

COPY DifFlow3D/ /opt/DifFlow3D/

# The current DifFlow3D repository already contains the modern PyTorch/CUDA
# PointNet2 wrappers. Build the extension in its package-local location so the
# runtime imports exactly the ABI-matched binary from /opt/DifFlow3D.
RUN test -d /opt/DifFlow3D/difflow3d/ops/pointnet2/src \
    && test -f /opt/DifFlow3D/checkpoints/model_difflow_355_0.0114.pth \
    && ! grep -RInE 'THC/THC\.h|THCState' \
         /opt/DifFlow3D/difflow3d/ops/pointnet2/src \
         --include='*.cpp' --include='*.cu' --include='*.h'

RUN cd /opt/DifFlow3D/difflow3d/ops/pointnet2 \
    && rm -rf build pointnet2_cuda*.so \
    && /opt/difflow-build-venv/bin/python setup.py build_ext --inplace \
    && cd /opt/DifFlow3D \
    && PYTHONPATH=/opt/DifFlow3D \
       /opt/difflow-build-venv/bin/python - <<'PY'
from pathlib import Path
import torch
from difflow3d.ops.pointnet2 import pointnet2_utils

required = (
    'gaussian_softmax_recovery_wrapper',
    'gaussian_recovery_hash_build_wrapper',
    'gaussian_softmax_recovery_local_wrapper',
    'gaussian_softmax_recovery_local_track_aware_wrapper',
)
loaded = Path(pointnet2_utils.extension_path()).resolve()
expected = Path('/opt/DifFlow3D/difflow3d/ops/pointnet2').resolve()
missing = [name for name in required if not hasattr(pointnet2_utils.pointnet2, name)]
print('DifFlow builder torch:', torch.__version__, torch.version.cuda)
print('pointnet2_cuda:', loaded)
if loaded.parent != expected:
    raise RuntimeError(f'Loaded PointNet2 extension from {loaded}; expected {expected}')
if missing:
    raise RuntimeError(f'Missing DifFlow recovery CUDA symbols: {missing}')
print('[OK] DifFlow3D PointNet2 + recovery extension')
PY

RUN /opt/difflow-build-venv/bin/python - <<'PY'
from pathlib import Path
import shutil

root = Path('/usr/local/cuda')
candidates = sorted(root.rglob('libcudart.so.12'))
if not candidates:
    raise FileNotFoundError('libcudart.so.12 not found in CUDA builder')
source = candidates[0].resolve()
dst_dir = Path('/opt/difflow-cuda-runtime/lib')
dst_dir.mkdir(parents=True, exist_ok=True)
shutil.copy2(source, dst_dir / 'libcudart.so.12')
print('bundled', source)
PY

# =============================================================================
# Stage 2: Isaac Sim + ROS + isolated inference environment.
# =============================================================================
FROM ${ISAAC_SIM_IMAGE}

USER root
SHELL ["/bin/bash", "-o", "pipefail", "-c"]
ARG DEBIAN_FRONTEND=noninteractive
ARG ISAAC_TORCH_VERSION
ARG ISAAC_TORCHVISION_VERSION
ARG TRACKING_TORCH_VERSION
ARG TRACKING_TORCHVISION_VERSION
ARG TORCH_INDEX_URL
ARG WARP_VERSION
ARG SAM3_REPOSITORY
ARG SAM3_REF
ARG EFFICIENT_TAM_REPOSITORY
ARG EFFICIENT_TAM_REF

ENV LANG=en_US.UTF-8 \
    LC_ALL=en_US.UTF-8 \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONNOUSERSITE=1 \
    PIP_NO_CACHE_DIR=1 \
    ISAAC_SIM_PATH=/isaac-sim \
    DIFFLOW_REPO=/opt/DifFlow3D \
    ROS_DISTRO=jazzy \
    ROS_DOMAIN_ID=117 \
    RMW_IMPLEMENTATION=rmw_fastrtps_cpp \
    ACCEPT_EULA=Y \
    PRIVACY_CONSENT=Y \
    OMNICLIENT_HUB_MODE=disabled \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=all \
    HOME=/isaac-sim \
    HF_HOME=/isaac-sim/.cache/huggingface \
    TORCH_HOME=/isaac-sim/.cache/torch \
    MPLCONFIGDIR=/isaac-sim/.cache/matplotlib

RUN apt-get update && apt-get install -y --no-install-recommends \
      ca-certificates curl git git-lfs gnupg2 locales lsb-release \
      software-properties-common \
      python3 python3-dev python3.12 python3.12-dev python3.12-venv \
      build-essential cmake ninja-build pkg-config ffmpeg \
      libgl1 libgl1-mesa-dri libglx-mesa0 libegl-mesa0 \
      libglib2.0-0 libsm6 libxext6 libxrender1 libx11-6 \
      libxrandr2 libxinerama1 libxcursor1 libxi6 xauth mesa-utils \
      libgomp1 \
    && locale-gen en_US.UTF-8 \
    && curl -fsSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key \
      -o /usr/share/keyrings/ros-archive-keyring.gpg \
    && echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo ${UBUNTU_CODENAME}) main" \
      > /etc/apt/sources.list.d/ros2.list \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
      ros-jazzy-ros-base \
      ros-jazzy-rmw-fastrtps-cpp \
      ros-jazzy-geometry-msgs \
      ros-jazzy-sensor-msgs \
      ros-jazzy-sensor-msgs-py \
      ros-jazzy-std-msgs \
      ros-jazzy-visualization-msgs \
      ros-jazzy-message-filters \
      ros-jazzy-cv-bridge \
      ros-jazzy-tf2-ros \
      ros-jazzy-tf2-tools \
      ros-jazzy-tf2-geometry-msgs \
      ros-jazzy-rviz2 \
      ros-jazzy-image-transport-plugins \
    && git lfs install --system \
    && rm -rf /var/lib/apt/lists/*

# Isaac-only GPU packages. Never expose tracking-venv to Isaac Python.
RUN mkdir -p /opt/isaac-python-packages \
    && /isaac-sim/python.sh -m pip install --no-cache-dir \
      --target /opt/isaac-python-packages \
      torch==${ISAAC_TORCH_VERSION} \
      torchvision==${ISAAC_TORCHVISION_VERSION} \
      --index-url ${TORCH_INDEX_URL} \
    && PYTHONPATH=/opt/isaac-python-packages \
      /isaac-sim/python.sh -m pip install --no-cache-dir \
      --target /opt/isaac-python-packages \
      numpy==1.26.4 \
      warp-lang==${WARP_VERSION}

RUN /isaac-sim/python.sh - <<'PY'
from pathlib import Path
import site
site_dirs = site.getsitepackages()
if not site_dirs:
    raise RuntimeError('Isaac Sim site-packages not found')
pth = Path(site_dirs[0]) / 'scenepredictor_isaac_paths.pth'
pth.write_text(
    "import sys; sys.path.append('/opt/isaac-python-packages'); "
    "sys.path.append('/workspace')\n",
    encoding='utf-8',
)
print('created', pth)
PY

# Shared inference venv for MultiViewRGBDTracker + ScenePredictor + DifFlow3D.
RUN python3.12 -m venv /opt/tracking-venv \
    && /opt/tracking-venv/bin/python -m pip install --no-cache-dir --upgrade pip \
    && /opt/tracking-venv/bin/python -m pip install --no-cache-dir \
      "setuptools>=75,<81" "wheel>=0.45,<1" \
    && /opt/tracking-venv/bin/python -m pip install --no-cache-dir \
      torch==${TRACKING_TORCH_VERSION} \
      torchvision==${TRACKING_TORCHVISION_VERSION} \
      --index-url ${TORCH_INDEX_URL} \
    && /opt/tracking-venv/bin/python -m pip install --no-cache-dir \
      "numpy>=1.26,<2" \
      "scipy>=1.12,<2" \
      "pandas>=2.2,<3" \
      "opencv-python-headless>=4.9,<4.12" \
      "Pillow>=10,<12" \
      "PyYAML>=6,<7" \
      "tqdm>=4.66" \
      "psutil>=5.9" \
      "hydra-core>=1.3.2,<1.4" \
      "omegaconf>=2.3,<2.4" \
      "huggingface-hub>=0.27,<2" \
      "safetensors>=0.4" \
      "transformers>=4.48,<5" \
      "accelerate>=1,<2" \
      "timm>=1.0.17" \
      "ftfy==6.1.1" regex \
      "iopath>=0.1.10" \
      "portalocker>=2.10" \
      "einops>=0.8" \
      ninja \
      "matplotlib>=3.9" \
      "typing_extensions>=4.12" \
      "pycocotools>=2.0.8,<3" \
      "decord==0.6.0" \
      "scikit-image>=0.24" \
      "scikit-learn>=1.5" \
      joblib threadpoolctl cffi pycparser pypng \
      "ultralytics-opencv-headless>=8.4,<9" \
      "open_clip_torch>=3.3,<4"

# Upstream SAM3 + EfficientTAM model repositories used by the tracker source dependency.
RUN mkdir -p /opt/upstream \
    && git clone "${SAM3_REPOSITORY}" /opt/upstream/sam3 \
    && git -C /opt/upstream/sam3 checkout "${SAM3_REF}" \
    && git -C /opt/upstream/sam3 submodule update --init --recursive \
    && git clone "${EFFICIENT_TAM_REPOSITORY}" /opt/upstream/efficient-tam \
    && git -C /opt/upstream/efficient-tam checkout "${EFFICIENT_TAM_REF}" \
    && git -C /opt/upstream/efficient-tam submodule update --init --recursive \
    && /opt/tracking-venv/bin/python -m pip install --no-cache-dir --no-deps -e /opt/upstream/sam3

COPY --from=difflow-builder /opt/DifFlow3D /opt/DifFlow3D
COPY --from=difflow-builder /opt/difflow-cuda-runtime/lib/libcudart.so.12 /usr/local/lib/difflow-cuda/libcudart.so.12

RUN printf '%s\n' \
      '/usr/local/lib/difflow-cuda' \
      '/opt/tracking-venv/lib/python3.12/site-packages/torch/lib' \
      > /etc/ld.so.conf.d/scenepredictor-inference.conf \
    && ldconfig

# ROS bindings + upstream model repos + top-level tracker source + compiled DifFlow.
# Deliberately do NOT add /workspace/MultiViewRGBDTracker/isaacscene.
RUN printf '%s\n' \
      '/opt/ros/jazzy/lib/python3.12/site-packages' \
      '/usr/lib/python3/dist-packages' \
      '/opt/upstream/sam3' \
      '/opt/upstream/efficient-tam' \
      '/workspace/MultiViewRGBDTracker' \
      '/workspace' \
      '/opt/DifFlow3D' \
      > /opt/tracking-venv/lib/python3.12/site-packages/scenepredictor_paths.pth

# Build-time smoke test. Source code under /workspace is bind-mounted only at runtime,
# so this validates installed/upstream/compiled dependencies rather than project imports.
RUN source /opt/ros/jazzy/setup.bash \
    && PYTHONPATH=/opt/upstream/sam3:/opt/upstream/efficient-tam:/opt/DifFlow3D \
       /opt/tracking-venv/bin/python - <<'PY'
from pathlib import Path
import rclpy
import torch
import sam3
import efficient_track_anything
import difflow3d
from difflow3d.model import PointConvBidirection
from difflow3d.runtime import DifFlow3DStreamingCudaGraphRunner, SoftmaxAnchorMotionRecoverer
from difflow3d.ops.pointnet2 import pointnet2_utils

required = (
    'gaussian_softmax_recovery_wrapper',
    'gaussian_recovery_hash_build_wrapper',
    'gaussian_softmax_recovery_local_wrapper',
    'gaussian_softmax_recovery_local_track_aware_wrapper',
)
loaded = Path(pointnet2_utils.extension_path()).resolve()
missing = [name for name in required if not hasattr(pointnet2_utils.pointnet2, name)]
print('inference torch:', torch.__version__, torch.version.cuda)
print('SAM3:', sam3.__file__)
print('EfficientTAM:', efficient_track_anything.__file__)
print('DifFlow3D:', difflow3d.__file__)
print('PointNet++:', loaded)
if not str(loaded).startswith('/opt/DifFlow3D/difflow3d/ops/pointnet2/'):
    raise RuntimeError(f'Unexpected PointNet2 extension: {loaded}')
if missing:
    raise RuntimeError(f'Missing DifFlow recovery CUDA symbols: {missing}')
print('[OK] build-time inference dependencies')
PY

RUN cat > /usr/local/bin/scenepredictor-entrypoint <<'EOF_ENTRYPOINT'
#!/usr/bin/env bash
set -euo pipefail
if [[ -f /opt/ros/jazzy/setup.bash ]]; then
    set +u
    source /opt/ros/jazzy/setup.bash
    set -u
fi
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-117}"
export RMW_IMPLEMENTATION="${RMW_IMPLEMENTATION:-rmw_fastrtps_cpp}"
export FASTDDS_BUILTIN_TRANSPORTS="${FASTDDS_BUILTIN_TRANSPORTS:-LARGE_DATA?max_msg_size=4MB&sockets_size=8MB&non_blocking=true&tcp_negotiation_timeout=50}"
exec "$@"
EOF_ENTRYPOINT
RUN chmod +x /usr/local/bin/scenepredictor-entrypoint

RUN install -d -o 1234 -g 1234 -m 0775 \
      /workspace \
      /workspace/logs \
      /isaac-sim/kit/cache \
      /isaac-sim/.cache/ov \
      /isaac-sim/.cache/warp \
      /isaac-sim/.cache/matplotlib \
      /isaac-sim/.cache/huggingface \
      /isaac-sim/.cache/torch \
      /isaac-sim/.nv/ComputeCache \
      /isaac-sim/.nvidia-omniverse/logs \
      /isaac-sim/.nvidia-omniverse/config \
      /isaac-sim/.local/share/ov/data \
      /tmp/runtime-1234

USER 1234:1234
WORKDIR /workspace
ENTRYPOINT ["/usr/local/bin/scenepredictor-entrypoint"]
CMD ["/bin/bash"]