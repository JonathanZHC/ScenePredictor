# scene_pred_pipeline

Minimal, GPU-oriented implementation of:

1. Batched YOLO instance segmentation and batched OpenCLIP embeddings.
2. Multi-view centroid gating plus occlusion-aware sparse reprojection.
3. Fixed world-frame voxel downsampling.
4. Simple previous-frame tracking and multi-lag motion classification.
5. Moving-point preselection, CUDA FPS, DifFlow3D inference and same-track
   distance-softmax velocity recovery.

## Install into the repository

Copy the folders and files into the ScenePredictor repository root.

## Run

Start Isaac Sim first, preferably with low-rate visualization point clouds:

```bash
/isaac-sim/python.sh /workspace/isaacscene/run_isaacsim.py \
  --scene dynamic \
  --width 640 \
  --height 480 \
  --rgbd-hz 30 \
  --pointcloud-hz 2
```

Start ScenePredictor in a second terminal:

```bash
source /opt/ros/jazzy/setup.bash
/isaac-sim/python.sh /workspace/scripts/run_scene_pred_pipeline.py \
  --config /workspace/configs/default.yaml
```

Start RViz:

```bash
rviz2 -d /workspace/scene_pred_pipeline.rviz
```

## Main outputs

- `/scene_predictor/background_points`
- `/scene_predictor/static_object_points`
- `/scene_predictor/moving_object_points`
- `/scene_predictor/scene_velocity`
- `/scene_predictor/velocity_markers`
- `/scene_predictor/<camera>/moving_mask`

The `scene_velocity` PointCloud2 contains:
`x, y, z, vx, vy, vz, track_id`.
