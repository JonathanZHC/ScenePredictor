# ScenePredictor

ScenePredictor is a real-time multi-view RGB-D scene prediction stack built around three reusable components:

- [`isaacscene`](https://github.com/JonathanZHC/isaacscene) for Isaac Sim scenes, synchronized RGB-D sensing, TF, and ROS 2 publication;
- [`MultiViewRGBDTracker`](https://github.com/JonathanZHC/MultiViewRGBDTracker) for SAM3 + EfficientTAM instance tracking, cross-view fusion, and persistent cross-frame identities;
- [`DifFlow3D`](https://github.com/JonathanZHC/DifFlow3D) for GPU scene-flow inference.

The repository is currently being refactored so that tracking/alignment is delegated completely to `MultiViewRGBDTracker`, while ScenePredictor consumes its fused persistent instances and performs scene-flow estimation and velocity recovery downstream.

> **Development status:** the Docker/runtime/dependency layout described below is the intended current setup. The legacy `scene_pred_pipeline` implementation is still being replaced by the new tracker → scene-flow integration.

## Intended pipeline

```text
Isaac Sim / RGB-D cameras
        │
        ▼
isaacscene
        │
        │ synchronized RGB-D + camera calibration + TF
        ▼
MultiViewRGBDTracker
        │
        │ fused MultiViewInstance objects
        │ persistent global_track_id
        ▼
ScenePredictor
        │
        ├── keep only instances present in both t-1 and t
        ├── sample the filtered two-frame point clouds
        ├── one combined DifFlow3D inference
        └── same-instance dense velocity recovery
        │
        ▼
ROS 2 / RViz output
```

The scene-flow stage always uses only the immediately previous frame (`t-1 → t`). A newly appearing instance is excluded from scene flow until it also exists in the previous frame.

## Repository layout

The three external repositories are intentionally kept at the same level:

```text
ScenePredictor/
├── Dockerfile
│
├── DifFlow3D/                  # top-level Git submodule
├── MultiViewRGBDTracker/       # top-level Git submodule
├── isaacscene/                 # top-level Git submodule
│
├── checkpoints/                # local model weights, ignored by Git
├── configs/
├── scene_pred_pipeline/        # ScenePredictor implementation
├── scripts/
│   ├── build.sh
│   ├── launch.sh
│   ├── download_checkpoints.sh
│   ├── run_isaac.sh
│   ├── run_tracking.sh
│   ├── run_inference.sh
│   ├── run_rviz.sh
│   ├── verify.sh
│   └── stop.sh
│
├── tests/
├── scene_pred_pipeline.rviz
└── README.md
```

`MultiViewRGBDTracker` itself also references `isaacscene` as a submodule for standalone use. ScenePredictor **does not use or initialize that nested copy**. It uses only the top-level:

```text
ScenePredictor/isaacscene
```

## Requirements

Host requirements:

- Linux with an NVIDIA GPU;
- NVIDIA driver compatible with the selected Isaac Sim / CUDA images;
- Docker;
- NVIDIA Container Toolkit;
- X11 access if Isaac Sim or RViz is run with a GUI.

The default runtime uses:

```text
Isaac Sim / sensors:
    /isaac-sim/python.sh

Tracker + ScenePredictor + DifFlow3D:
    /opt/tracking-venv/bin/python
```

The separation is intentional. Do not globally activate the tracking virtual environment inside Isaac Sim.

DifFlow3D's PointNet++ CUDA extension is built against the same PyTorch/CUDA ABI as the tracking inference environment.

## 0. Preparation on host

Run once on the host:

```bash
sudo tee /etc/sysctl.d/99-fastdds-large-data.conf >/dev/null <<'EOF2'
net.core.rmem_max=16777216
net.core.wmem_max=16777216

net.ipv4.tcp_rmem=4096 4194304 16777216
net.ipv4.tcp_wmem=4096 4194304 16777216
EOF2

sudo sysctl --system
```

Check:

```bash
sysctl net.core.rmem_max
sysctl net.core.wmem_max
sysctl net.ipv4.tcp_rmem
sysctl net.ipv4.tcp_wmem
```

Expected values include:

```text
net.core.rmem_max = 16777216
net.core.wmem_max = 16777216
net.ipv4.tcp_rmem = 4096 4194304 16777216
net.ipv4.tcp_wmem = 4096 4194304 16777216
```

## 1. Clone

Clone the parent repository first:

```bash
git clone https://github.com/JonathanZHC/ScenePredictor.git
cd ScenePredictor
```

Initialize **only the three top-level submodules**:

```bash
git submodule update --init \
  DifFlow3D \
  MultiViewRGBDTracker \
  isaacscene
```

Do **not** use:

```bash
git submodule update --init --recursive
```

because that would also initialize:

```text
MultiViewRGBDTracker/isaacscene
```

which ScenePredictor intentionally does not use.

Check the top-level dependencies:

```bash
git submodule status
```

The nested tracker copy should remain uninitialized:

```bash
git -C MultiViewRGBDTracker submodule status
```

A leading `-` before the nested `isaacscene` commit means that nested submodule is registered but not initialized.

## 2. Build the Docker image

From the repository root:

```bash
./scripts/build.sh
```

The default image is:

```text
scenepredictor:latest
```

The Dockerfile:

- uses Isaac Sim 6.0.1 as the runtime base;
- keeps Isaac Python isolated from the tracking Python environment;
- installs SAM3 and EfficientTAM runtime sources;
- builds the DifFlow3D PointNet++ CUDA extension in a matching PyTorch/CUDA environment;
- exposes ROS 2 Jazzy to both runtime paths where needed;
- bind-mounts the ScenePredictor repository at `/workspace` at runtime.

## 3. Start the persistent container

```bash
./scripts/launch.sh
```

The default container name is:

```text
scenepredictor
```

The container stays alive so Isaac Sim, tracking/inference, and RViz can run as separate processes.

If it is already running, `launch.sh` should simply reuse it.

## 4. Download checkpoints

The runtime expects:

```text
checkpoints/
├── sam3.pt
└── efficienttam_s_512x512.pt
```

Checkpoint files are ignored by Git.

SAM3 is hosted in the gated `facebook/sam3` Hugging Face repository, so the Hugging Face account associated with the token must already have access.

The recommended command is:

```bash
HF_TOKEN=hf_xxxxxxxxxxxxxxxxx ./scripts/download_checkpoints.sh
```

The script:

- skips checkpoints that already exist;
- downloads `efficienttam_s_512x512.pt`;
- downloads `sam3.pt` using `HF_TOKEN`;
- writes both weights into the top-level `checkpoints/` directory.

If the checkpoint directory is not writable from the container, fix only that runtime directory, for example:

```bash
mkdir -p checkpoints
chmod a+rwx checkpoints
```

Do not make the whole repository world-writable.

DifFlow3D's pretrained scene-flow checkpoint is provided by the `DifFlow3D` dependency and does not need to be downloaded by this script.

## 5. Verify the environment

After the image is built and the container is running:

```bash
./scripts/verify.sh
```

The verification script checks:

1. NVIDIA GPU visibility;
2. ROS 2 Jazzy;
3. Isaac Python isolation;
4. SAM3 / EfficientTAM / `MultiViewRGBDTracker` imports;
5. DifFlow3D and the PointNet++ CUDA extension;
6. the expected top-level repository layout.

It also warns if `MultiViewRGBDTracker/isaacscene` was initialized, because ScenePredictor does not use that nested dependency.

## 6. Run Isaac Sim

Open terminal 1:

```bash
./scripts/run_isaac.sh dynamic
```

Available scene modes:

```bash
./scripts/run_isaac.sh static
./scripts/run_isaac.sh dynamic
./scripts/run_isaac.sh hybrid
./scripts/run_isaac.sh occlusion
```

Additional arguments are forwarded to `isaacscene/run_isaacsim.py`, for example:

```bash
./scripts/run_isaac.sh dynamic --headless
```

or:

```bash
./scripts/run_isaac.sh dynamic --pointcloud-hz 0
```

The default launcher uses:

```text
resolution:          640 × 480
RGB-D rate:          30 Hz
PointCloud2 rate:     5 Hz
depth corruption:    enabled
RGB corruption:      disabled
motion speed scale:  1.0
```

ScenePredictor uses the top-level:

```text
/workspace/isaacscene
```

and not:

```text
/workspace/MultiViewRGBDTracker/isaacscene
```

## 8. Run ScenePredictor

Open the inference process with:

```bash
./scripts/run_inference.sh
```

By default it uses:

```text
/workspace/configs/default.yaml
```

A different config can be selected with:

```bash
CONFIG=/workspace/configs/my_config.yaml ./scripts/run_inference.sh
```

> **Current implementation note:** the legacy `scene_pred_pipeline` is still being replaced. The intended new implementation will invoke `MultiViewRGBDTracker` directly and then perform instance-filtered DifFlow3D inference and velocity recovery. The runtime interface of `run_inference.sh` is intended to remain stable across that refactor.

## 9. RViz

ScenePredictor visualization:

```bash
./scripts/run_rviz.sh
```

or explicitly:

```bash
./scripts/run_rviz.sh predictor
```

Tracker visualization:

```bash
./scripts/run_rviz.sh tracker
```

Isaac-only visualization:

```bash
./scripts/run_rviz.sh isaac
```

If the container user needs permission to save the config:

```bash
touch rviz/scene_pred_pipeline.rviz
sudo setfacl -m u:1234:rw rviz/scene_pred_pipeline.rviz
sudo setfacl -m u:1234:rwx rviz
```

## 10. Stop the container

```bash
./scripts/stop.sh
```

## Typical development workflow

After the first build:

**Terminal 1 — Isaac Sim**

```bash
./scripts/run_isaac.sh dynamic
```

**Terminal 2 — tracker-only debugging**

```bash
./scripts/run_tracking.sh
```

**Terminal 3 — tracker RViz**

```bash
./scripts/run_rviz.sh tracker
```

For the integrated ScenePredictor path, terminal 2 will instead be:

```bash
./scripts/run_inference.sh
```

with:

```bash
./scripts/run_rviz.sh predictor
```

for visualization.

## Updating the dependencies

The parent repository pins exact commits for reproducibility.

To update one dependency, enter it, update to the desired commit, and then commit the changed gitlink in ScenePredictor. For example:

```bash
cd MultiViewRGBDTracker
git pull origin main
cd ..

git add MultiViewRGBDTracker
git commit -m "Update MultiViewRGBDTracker dependency"
```

The same pattern applies to `isaacscene` and `DifFlow3D`.

Do not recursively initialize nested submodules unless a specific dependency explicitly requires them.

## Git hygiene

Runtime-generated data should not be committed.

The repository `.gitignore` should exclude at least:

```gitignore
checkpoints/
*.pt
*.pth
*.ckpt
*.engine

.home/
.cache/
.container-cache/

logs/
profiles/

__pycache__/
*.py[cod]
.pytest_cache/
```

In particular, Isaac Sim / NVIDIA runtime caches such as `.cache/`, `.container-cache/`, and `.home/` should never be added to Git.

## ROS / DDS

The Docker runtime uses ROS 2 Jazzy with:

```text
ROS_DOMAIN_ID=117
RMW_IMPLEMENTATION=rmw_fastrtps_cpp
```

The launch scripts also configure Fast DDS large-data transport for RGB-D / point-cloud traffic.

If large ROS messages are dropped, host socket buffer limits may need to be increased depending on the machine and network configuration.

## Planned ScenePredictor integration

The implementation refactor will make the interface between tracking and scene flow explicit:

```text
MultiViewRGBDTracker output at t-1
MultiViewRGBDTracker output at t
        │
        ▼
intersection of persistent global_track_id values
        │
        ▼
filter both frames to the same instance set
        │
        ▼
point sampling / FPS
        │
        ▼
one combined DifFlow3D inference
        │
        ▼
same-global-ID velocity recovery
        │
        ▼
ROS / RViz output
```

If an object first appears at frame `t`, it is excluded from scene flow at `t`. If it remains visible at `t+1`, it can participate in the `t → t+1` scene-flow estimate.

The previous-frame state is updated every frame even if no common instance exists, so scene flow never unintentionally spans more than one frame.
