# ScenePredictor

ScenePredictor is a real-time multi-view RGB-D scene prediction pipeline built from three reusable repositories:

- [`isaacscene`](https://github.com/JonathanZHC/isaacscene): optional Isaac Sim scenes, RGB-D sensing, TF, and ROS 2 publication;
- [`MultiViewRGBDTracker`](https://github.com/JonathanZHC/MultiViewRGBDTracker): SAM3 + EfficientTAM tracking, multi-view fusion, and persistent global IDs;
- [`DifFlow3D`](https://github.com/JonathanZHC/DifFlow3D): CUDA scene-flow inference and dense velocity recovery.

The design is now **frozen**. ScenePredictor invokes the tracker in-process, keeps only persistent instances shared by consecutive frames, runs one combined DifFlow3D inference, and recovers dense same-track velocity.

## Frozen runtime architecture

```text
RGB-D cameras / rosbag / Isaac Sim
                │
                ▼
        MultiViewRGBDTracker
                │
                ├─ SAM3: sparse asynchronous refresh
                ├─ EfficientTAM: fixed-batch mask propagation
                ├─ GPU mask/depth postprocess + deterministic voxelization
                ├─ CPU cross-view matching/fusion
                ├─ CPU temporal gating/Hungarian
                └─ persistent GPU cloud bank + batched Chamfer
                                │
                                │ same GPU bank is reused
                                ▼
                         ScenePredictor
                                │
                common global_track_id(t-1, t)
                                │
                                ▼
                     one DifFlow3D inference
                                │
                                ▼
                 same-track dense velocity recovery
                                │
                                ▼
                          ROS 2 / RViz
```

The scene-flow pair is always the immediately adjacent tracker pair `t-1 -> t`. New objects are excluded until they also exist in the previous frame.

### CPU/GPU split

The final split is intentional:

- **GPU data plane:** mask processing, RGB-D geometry, voxel deduplication, depth prefetch, cross-frame cloud bank, Chamfer, DifFlow3D, dense recovery;
- **CPU control plane:** cross-view gates/overlap, fusion bookkeeping, temporal centroid gate, Hungarian assignment, persistent IDs.

Full GPU alignment was benchmarked and rejected because small-tensor launch/synchronization overhead was substantially slower than the CPU control path. The only GPU alignment primitive retained is the computationally heavy Chamfer stage.

## Repository layout

```text
ScenePredictor/
├── Dockerfile
├── README.md
├── configs/
│   ├── default.yaml
│   ├── tracking.yaml
│   └── difflow.yaml
├── scene_pred_pipeline/
├── scripts/
├── DifFlow3D/                 # top-level submodule
├── MultiViewRGBDTracker/      # top-level submodule
└── isaacscene/                # top-level submodule
```

`MultiViewRGBDTracker` may also contain its own `isaacscene` submodule for standalone use. ScenePredictor uses only the top-level `ScenePredictor/isaacscene` checkout.

## Requirements

- Ubuntu/Linux host with NVIDIA GPU;
- Docker + NVIDIA Container Toolkit;
- NVIDIA driver compatible with CUDA 12.8+/Blackwell;
- X11 only when Isaac Sim or RViz GUI is required.

The tested RTX 5090 runtime uses:

```text
Isaac Sim base          nvcr.io/nvidia/isaac-sim:6.0.1
Tracking/DifFlow Torch  2.8.0 + cu128
Warp                    1.15.0
ROS 2                   Jazzy
Python                   3.12 tracking venv
```

Warp **1.15.0 is pinned explicitly in both Isaac Python and `/opt/tracking-venv`**. The Warp 1.15 device-index/codegen issues encountered during development were source-code issues and are already fixed in the frozen tracker kernels.

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

```bash
git clone https://github.com/JonathanZHC/ScenePredictor.git
cd ScenePredictor

git submodule update --init \
  DifFlow3D \
  MultiViewRGBDTracker \
  isaacscene
```

Do not use `--recursive` for the parent repository unless you explicitly want the tracker's nested standalone `isaacscene` checkout.

## 2. Build from source

```bash
./scripts/build.sh
```

Default image:

```text
scenepredictor:latest
```

The Docker build:

1. creates an isolated `/opt/tracking-venv`;
2. installs the pinned tracking Torch and Warp versions;
3. clones the SAM3 and EfficientTAM runtime sources;
4. removes any stale local DifFlow3D PointNet++ binary and **rebuilds the CUDA extension from source** against the exact tracking Torch/CUDA ABI;
5. copies the ABI-matched DifFlow3D runtime into `/opt/DifFlow3D`;
6. runs build-time import/API checks.

ScenePredictor runtime source remains bind-mounted at `/workspace`; the compiled DifFlow3D runtime is loaded from `/opt/DifFlow3D` so an old `.so` in a source checkout cannot override the ABI-matched build.

> For fully reproducible long-term images, override `SAM3_REF` and `EFFICIENT_TAM_REF` with tested commit hashes instead of their default `main` values.

## 3. Start the persistent container

```bash
./scripts/launch.sh
```

The default container name is `scenepredictor`.

## 4. Download checkpoints

The top-level runtime expects:

```text
checkpoints/
├── sam3.pt
└── efficienttam_s_512x512.pt
```

SAM3 is gated. Use:

```bash
HF_TOKEN=hf_xxxxxxxxxxxxxxxxx ./scripts/download_checkpoints.sh
```

DifFlow3D's `model_difflow_355_0.0114.pth` stays inside the DifFlow3D dependency.

## 5. Verify the environment

```bash
./scripts/verify.sh
```

Verification checks:

- NVIDIA GPU visibility;
- ROS 2 Jazzy;
- Isaac Python isolation;
- Warp 1.15.0 in both Isaac and tracking Python;
- SAM3 / EfficientTAM / MultiViewRGBDTracker imports;
- DifFlow3D and the ABI-matched PointNet++/recovery CUDA symbols;
- expected source layout.

## 6. Run with real/recorded RGB-D

Start ScenePredictor:

```bash
./scripts/run_inference.sh
```

Play a bag in another terminal:

```bash
./scripts/run_rosbag.sh /workspace/rosbags/<bag_name>
```

Additional `ros2 bag play` arguments are forwarded, for example:

```bash
./scripts/run_rosbag.sh /workspace/rosbags/<bag_name> --loop
```

This is the preferred performance benchmark because a real RGB-D camera does not consume the same NVIDIA GPU for rendering.

## 7. Run with Isaac Sim

Terminal 1:

```bash
./scripts/run_isaac.sh dynamic
```

Other scene modes supported by `isaacscene` can be forwarded in the same way. Isaac Sim shares the GPU with tracking/DifFlow and therefore produces more pessimistic inference timing than a physical camera or rosbag source.

Terminal 2:

```bash
./scripts/run_inference.sh
```

## 8. RViz

```bash
./scripts/run_rviz.sh predictor
```

If the container user needs permission to save the config:

```bash
touch rviz/tracking.rviz
sudo setfacl -m u:1234:rw rviz/tracking.rviz
sudo setfacl -m u:1234:rwx rviz
```

Tracker-only and Isaac-only configs remain available:

```bash
./scripts/run_rviz.sh tracker
./scripts/run_rviz.sh isaac
```

Visualization is lazy: masks, overlays, point-cloud messages, and markers are materialized only when required by subscribers, outside the numerical hot path where possible.

## 9. Tracker-only standalone use

`MultiViewRGBDTracker` remains a standalone repository with its own Dockerfile, scripts, config, README, and optional nested `isaacscene` dependency.

Inside ScenePredictor's container you can also run the tracker directly:

```bash
./scripts/run_tracking.sh
```

For an independent checkout, follow `MultiViewRGBDTracker/README.md` and use its own:

```bash
./scripts/build.sh
./scripts/launch.sh
./scripts/run_tracking.sh
```

The frozen tracker path keeps:

- GPU mask resize/threshold/erosion;
- asynchronous depth prefetch;
- fused Warp RGB-D/world/voxel geometry;
- deterministic per-voxel representative selection;
- compact D2H for CPU alignment;
- lazy CPU mask materialization;
- CPU cross-view alignment;
- persistent-bank GPU Chamfer.

## 10. DifFlow3D standalone use

`DifFlow3D` remains independently buildable and testable. Its source tree contains no required prebuilt PointNet++ `.so`; after a fresh checkout build the extension with:

```bash
cd DifFlow3D
bash scripts/build_pointnet2_ops.sh
```

or build the standalone image using `DifFlow3D/Dockerfile`.

See `DifFlow3D/README.md` for runtime tests and benchmarks.

## 11. Configuration ownership

`configs/default.yaml` contains only ScenePredictor integration settings.

`configs/tracking.yaml` owns tracker behavior, including prompts, EfficientTAM execution, postprocessing, voxel matching, alignment, and tracker profiling.

The production postprocess optimization bundle is intentionally frozen. `postprocess.gpu_geometry: true` enables the validated CUDA path; the former independent A/B switches for direct geometry, compact D2H, depth prefetch, and lazy masks were removed.

`configs/difflow.yaml` owns all DifFlow numerical settings. The current deployment uses 2048 sampled anchors with the configured coarse/middle/fine iterations and local track-aware CUDA recovery.

## 12. Current performance reference

On an RTX 5090, using rosbag RGB-D input without Isaac Sim rendering contention, one representative run produced:

```text
tracker_total   median 13.144 ms   p95 21.437 ms
DifFlow3D       median  9.981 ms   p95 20.696 ms
adapter_total   median  0.442 ms   p95  1.136 ms
cycle_total     median 23.363 ms   p95 56.517 ms
```

The typical-frame latency is comfortably below the 33.33 ms budget for 30 Hz. Tail latency remains workload/GPU-scheduling dependent, so compare changes using the same rosbag and profiling window.

## 13. Design invariants

These are intentional and should not be changed casually during cleanup:

1. EfficientTAM state has one owner thread; SAM3 refresh is asynchronous.
2. Scene flow uses only `t-1 -> t`.
3. DifFlow inference is one combined call over all common persistent instances.
4. CPU alignment consumes compact voxel data; do not reintroduce full raw-cloud D2H.
5. Cross-frame Chamfer uploads each fused cloud once and retains the previous bank on GPU.
6. ScenePredictor reuses the already-uploaded CrossFrame bank and never silently performs a second fused-cloud CPU->GPU copy.
7. RViz/debug output must not move expensive materialization back into the numerical critical path.
8. Runtime cleanup must not add per-frame allocations/synchronizations to the hot path without benchmark evidence.

## 14. Stop

```bash
./scripts/stop.sh
```

## Development hygiene

Do not commit generated runtime/build data such as:

```text
checkpoints/
.container-cache/
.home/
.cache/
logs/
profiles/
__pycache__/
*.pyc
*.so
*.engine
```

The PointNet++ `.so` is a build artifact and should be regenerated for the active Torch/CUDA ABI.
