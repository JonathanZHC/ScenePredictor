# Clone the repo:

```bash
git clone git@github.com:JonathanZHC/ScenePredictor.git
git submodule update --init --recursive
```

# Build the docker:

```bash
docker build --progress=plain -f .docker/dockerfile -t scenepredictor .
```

# Run isaacsim to generate the scene:

```bash
docker exec -it scenepredictor bash -lc '
source /opt/ros/jazzy/setup.bash

/isaac-sim/python.sh \
  /workspace/isaacscene/run_isaacsim.py \
  --scene dynamic \
  --width 640 \
  --height 480 \
  --rgbd-hz 30 \
  --pointcloud-hz 5 \
  --corrupt \
  --no-rgb-corruption \
  --motion-speed-scale 1.0
'
```

# Download YOLO weight:

```bash
docker exec -it scenepredictor bash -lc '
mkdir -p /workspace/weights
cd /workspace/weights

/isaac-sim/python.sh - <<'"'"'PY'"'"'
from ultralytics import YOLOE

model = YOLOE(
    "yoloe-26x-seg.pt",
    task="segment",
)

print("YOLOE model loaded successfully")
print("Model:", model)
PY

ls -lh /workspace/weights/yoloe-26x-seg.pt
'
```

# Export TensorRT engine file:

```bash
docker exec -it scenepredictor bash -lc '
source /opt/ros/jazzy/setup.bash

/isaac-sim/python.sh \
  /workspace/scripts/export_yoloe_tensorrt.py \
  --weights /workspace/weights/yoloe-26x-seg.pt \
  --labels /workspace/configs/object_labels.txt \
  --output /workspace/weights/yoloe-26x-seg.engine \
  --imgsz 640 \
  --batch 2 \
  --device 0 \
  --workspace 4
'
```

# Run ScenePredictor:

```bash
docker exec -it scenepredictor bash -lc '
source /opt/ros/jazzy/setup.bash

/isaac-sim/python.sh \
  /workspace/scripts/run_scene_pred_pipeline.py \
  --config /workspace/configs/default.yaml
'
```

# Run Rviz for visualization:

```bash
docker exec -it scenepredictor bash -lc '
export XDG_RUNTIME_DIR=/tmp/runtime-1234
mkdir -p "${XDG_RUNTIME_DIR}"
chmod 700 "${XDG_RUNTIME_DIR}"

source /opt/ros/jazzy/setup.bash

rviz2 -d /workspace/scene_pred_pipeline.rviz
'
```


```bash
touch scene_pred_pipeline.rviz
sudo setfacl -m u:1234:rw scene_pred_pipeline.rviz
sudo setfacl -m u:1234:rwx .


# Check ROS topics:

```bash
docker exec -it scenepredictor bash -lc '
source /opt/ros/jazzy/setup.bash
ros2 topic list | sort
'
```






## The rest is for old pipeline

# Run the docker:

docker run --rm -it \
  --gpus all \
  --network host \
  --ipc=host \
  -e DISPLAY="$DISPLAY" \
  -e ROS_DOMAIN_ID=117 \
  -v "$PWD:/workspace" \
  difflow3d-test


# Run the script:

rviz2 -d test_scene_flow.rviz

python3 test_difflow3d_superquadrics.py \
  --difflow-repo /workspace \
  --checkpoint /opt/DifFlow3D/pretrain_weights/model_difflow_355_0.0114.pth \
  --frames 300 \
  --sensor-hz 30 \
  --difflow-num-points 4096 \
  --difflow-iters 4 \
  --difflow-uncertainty 0.2 \
  --cuda-graph-warmup 10 \
  --warmup 1 \
  --rviz 

Note: 
frames=300, sensor-hz=30, difflow-num-points=1024/2048, difflow-iters=4 
frames=300, sensor-hz=30, difflow-num-points=4096, difflow-iters=2/4
frames=100, sensor-hz=10, difflow-num-points=8192, difflow-iters=2 


# For Cuda profiler:

python3 test_difflow3d_superquadrics_profiled.py \
  --difflow-repo /workspace \
  --model-module model_difflow_profiled \
  --checkpoint /opt/DifFlow3D/pretrain_weights/model_difflow_355_0.0114.pth \
  --frames 100 \
  --sensor-hz 30 \
  --difflow-num-points 2048 \
  --difflow-iters 2 \
  --difflow-uncertainty 0.2 \
  --execution-backend cuda-graph \
  --cuda-graph-warmup 10 \
  --cuda-graph-no-fallback \
  --enable-tf32 \
  --warmup 1 \
  --rviz \
  --profile-cuda \
  --profile-only \
  --profile-output-dir ./profiles/difflow_2048_iters4 \
  --profile-warmup 10 \
  --profile-wait 1 \
  --profile-schedule-warmup 2 \
  --profile-active 5 \
  --profile-repeat 1 \
  --profile-row-limit 100




With distance-based softmax:

python3 test_voxel_fps_difflow3d.py \
    --difflow-repo /workspace \
    --model-module model_difflow \
    --checkpoint /opt/DifFlow3D/pretrain_weights/model_difflow_355_0.0114.pth \
    --all-points 300000 \
    --voxel-resolution 0.020 \
    --enable-second-downsample \
    --second-voxel-resolution 0 \
    --second-candidate-ratio 2.5 \
    --fps-points 2048 \
    --difflow-iters 4 \
    --recovery-method softmax \
    --recovery-softmax-sigma 0.05 \
    --frames 300 \
    --sensor-hz 30 \
    --warmup 2 \
    --rviz 

Or with inverse-distance weighted sum:

python3 test_voxel_fps_difflow3d.py \
    --difflow-repo /workspace \
    --model-module model_difflow \
    --checkpoint /opt/DifFlow3D/pretrain_weights/model_difflow_355_0.0114.pth \
    --all-points 300000 \
    --voxel-resolution 0.010 \
    --enable-second-downsample \
    --second-voxel-resolution 0 \
    --second-candidate-ratio 2.5 \
    --fps-points 2048 \
    --difflow-iters 4 \
    --recovery-method inverse-distance \
    --recovery-idw-power 2.0 \
    --recovery-idw-epsilon 1e-5 \
    --recovery-chunk-size 4096 \
    --frames 300 \
    --sensor-hz 30 \
    --warmup 2 \
    --rviz 

For visualization:

rviz2 -d voxel_fps_difflow3d.rviz