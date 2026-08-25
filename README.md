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
touch rviz/scene_pred_pipeline.rviz iz
sudo setfacl -m u:1234:rw rviz/scene_pred_pipeline.rviz 
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

`configs/default.yaml` owns ScenePredictor integration settings and semantic prompt ownership. `tracker.tracked_prompts` enter the full 3-D tracking/DifFlow path; `tracker.excluded_prompts` still use SAM3 + EfficientTAM in 2-D but stop after a dilated GPU exclusion mask.

`configs/tracking.yaml` owns native tracker/model behavior, including EfficientTAM execution, postprocessing, voxel matching, alignment, and tracker profiling. ScenePredictor injects `tracked_prompts + excluded_prompts` into the native detector at startup.

The production postprocess optimization bundle is intentionally frozen. `postprocess.gpu_geometry: true` enables the validated CUDA path; the former independent A/B switches for direct geometry, compact D2H, depth prefetch, and lazy masks were removed. Two newer switches sit on top of it:

- `postprocess.depth_boundary_filter` (`enabled`, `erosion_width_px`, `mad_multiplier`, `min_threshold_m`): CUDA depth-boundary flyer filter for tracked masks. It runs after the residual `tracking_erosion_pixels` erosion, erodes a w-pixel 4-connected core, estimates a per-instance threshold `T = median + k * 1.4826 * MAD` from the core's local depth differences and grows the boundary band back only where `|dD| < T` (sharp boundaries lose their flyers, curved surfaces keep their points). Kernels are JIT-compiled with NVRTC from the torch-bundled `libnvrtc`; no `nvcc` is needed at runtime. When enabled, set `tracking_erosion_pixels` to 0 or 1 so the filter decides the boundary.
- `postprocess.gpu_alignment` (default `true`): cross-view fusion and Chamfer bank staging stay on CUDA (no voxel-cloud D2H, NumPy fusion or re-upload). With one camera no CPU voxel data is copied at all; with several cameras cross-view *matching* still runs on CPU keys. Multi-camera groups are concatenated without cross-view voxel dedup (DifFlow's voxel-2 dedups on its own grid). `false` restores the legacy host round trip, which is also used automatically when `enable_visualization: true`.

`configs/difflow.yaml` owns all DifFlow numerical settings. The current deployment uses 1024 sampled anchors (the checkpoint's native level-1 count, which lets the encoder reuse one KNN across levels 0/1) with the configured coarse/middle/fine iterations and local track-aware CUDA recovery. DifFlow's decode path uses two NVRTC-JIT kernels (`difflow3d/ops/fused_knn.py`, exact register-heap KNN; `difflow3d/ops/fused_cross_block.py`, fused gather+add+activation prologue) with pure-PyTorch fallbacks; `USE_FUSED_KNN` / `USE_FUSED_CROSS_BLOCK` in `difflow3d/model/pointconv.py` toggle them, and `IDENTITY_UPSAMPLE_L1_TO_L0` in `difflow3d/model/difflow.py` controls the identity shortcut for the level-1 -> level-0 upsample at 1024 points.

Runtime plumbing worth knowing when editing: all tracker/DifFlow work runs on one shared highest-priority CUDA stream (`scene_pred_pipeline/cuda_streams.py`, bound on the tracker owner thread and the GPU worker) while SAM3 keeps its own default-priority stream; ROS message building runs on a latest-only publisher thread (`_PublishWorker` in `scripts/run_scene_pred_pipeline.py`) and is reported as `publish_total` under the async diagnostics of the periodic summary; Python GC is frozen and disabled after warm-up with an explicit collect on the publisher thread; the summary also prints input drop counters (`dropped_bundles`, `unmatched_rgbd`, `unmatched_multiview`).

## 12. Current performance reference

On an RTX 5090 with a live ZED stream, one camera and three tracked instances (human + 2 mice), a representative run produced:

```text
cycle_total     median 20.4 ms   p95 33.9 ms   max 40 ms
  tracker_total   median 11.3 ms   p95 20.3 ms   (tracking_model 6.4, postprocess 1.5, alignment 2.3)
  difflow_total   median  5.6 ms   p95 10.5 ms   (decode 3.3, encode 1.1, voxel-2 1.1)
  velocity_recovery, instance_filter, cycle_other: < 0.1 / < 0.1 / ~0.5 ms
sam3_async (off the cycle)   ~130 ms per refresh
```

With two cameras and three instances the same pipeline measured ~29 ms median / ~53 ms p95. For reference, the pre-optimization baseline for that two-camera case was 45 ms median / 73 ms p95 (see the profiler summary printed every `output.profile_interval_frames`).

The remaining tail is dominated by frames that overlap the asynchronous SAM3 refresh on the same GPU (every GPU stage shows p95 ~2x median on those frames); stream priority helps only partially because that contention is memory-bandwidth bound. Compare changes using the same rosbag and profiling window.

## 13. Design invariants

These are intentional and should not be changed casually during cleanup:

1. EfficientTAM state has one owner thread; SAM3 refresh is asynchronous.
2. Scene flow uses only `t-1 -> t`.
3. DifFlow inference is one combined call over all common persistent instances.
4. Alignment consumes compact per-record metadata (counts, centroids, and CPU voxel keys only when several cameras must be matched); fused clouds stay on CUDA and are never copied to the host on the production path. Do not reintroduce full raw-cloud D2H.
5. Cross-frame Chamfer stages each fused cloud once (device-to-device from the geometry buffers) and retains the previous bank on GPU.
6. ScenePredictor reuses the already-staged CrossFrame bank and never silently performs a second fused-cloud CPU->GPU copy.
7. RViz/debug output must not move expensive materialization back into the numerical critical path; publishing runs on its own thread.
8. Runtime cleanup must not add per-frame allocations/synchronizations to the hot path without benchmark evidence. In particular, avoid adding small GPU kernels followed by a host sync inside CPU stages: while SAM3 occupies the GPU each such kernel can wait milliseconds for SM slots.
9. Tracker, postprocess, alignment and DifFlow share one high-priority CUDA stream; any new stream must synchronize with it through events, not `torch.cuda.synchronize()`.

## 14. Stop

```bash
./scripts/stop.sh
```

## Development hygiene

Unit tests run inside the container with the inference venv (GPU tests skip themselves without CUDA):

```bash
docker exec scenepredictor bash -lc 'cd /workspace && PYTHONPATH=/workspace:/workspace/MultiViewRGBDTracker:/opt/DifFlow3D \
  /opt/tracking-venv/bin/python -m unittest discover -s tests -p "test_*.py"'
```

`tests/test_depth_boundary_filter.py`, `tests/test_gpu_alignment.py`, `tests/test_gpu_geometry_centroids.py`, `tests/test_difflow_fused_ops.py` and `tests/test_perf_fixes.py` check the CUDA paths above against NumPy/legacy references. Note that DifFlow runs from the image copy in `/opt/DifFlow3D`; after editing the `DifFlow3D` submodule either rebuild the image or copy the changed files there.

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
