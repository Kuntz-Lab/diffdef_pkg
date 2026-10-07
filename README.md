# diffdef_pkg

ROS 2 (Jazzy) nodes that predict a goal point cloud with DiffDef:

- `diffdef_node` subscribes to start and context point clouds and publishes a goal for each pair.
- `diffdef_service_node` returns a goal for each `GenerateGoalPointCloud` service request.

## Prediction

Both nodes share [diffdef_pkg/goal_generator.py](diffdef_pkg/goal_generator.py). For each start (tissue) and context cloud, it:

1. down-samples both clouds to `sample_num_points` (512) with farthest point sampling,
2. runs the DiffDef `BaoFlowVAE` model,
3. returns the predicted goal cloud, which uses the start cloud's header (`frame_id` and timestamp).

By default the model uses the prior mean latent (all zeros). The diffusion noise and the farthest point sampling start point are still random, so the goal can vary slightly between calls with the same input. Pass `--sample-latent` to draw a random latent per prediction instead.

The model code is imported from a DiffDef-core checkout (`models/vae_flow_3.py`). By default this is `~/retraction_ws/src/DiffDef-core`.

## Setup

The service type comes from the `retraction_interface` package, which must be in the same workspace.

ROS dependencies are in `package.xml`:

```bash
cd ~/retraction_ws
rosdep install --from-paths src/diffdef_pkg --ignore-src -y
```

torch is installed with pip into a venv that can also see the system ROS packages. Deactivate conda first, so the venv is built from the system Python:

```bash
conda deactivate   # if a conda env is active
/usr/bin/python3 -m venv --system-site-packages ~/venvs/diffdef
source ~/venvs/diffdef/bin/activate
pip install -r ~/retraction_ws/src/diffdef_pkg/requirements.txt
```

Build with plain `colcon`. The venv doesn't need to be active, but conda must be deactivated, or CMake uses conda's Python to generate `retraction_interface`:

```bash
source /opt/ros/jazzy/setup.bash
cd ~/retraction_ws
colcon build --symlink-install --packages-up-to diffdef_pkg
```

The launchers' shebang is `#!/usr/bin/env python3`, so they run on whichever `python3` is first on `PATH` at run time. `setup.py` sets this, because `--symlink-install` would otherwise hard-code the Python that ran colcon.

## Running

Always activate the venv first.

```bash
source /opt/ros/jazzy/setup.bash
source ~/venvs/diffdef/bin/activate
source ~/retraction_ws/install/setup.bash

ros2 run diffdef_pkg diffdef_node            # topics
ros2 run diffdef_pkg diffdef_service_node    # service
```

The checkpoint path is set by `CKPT_PATH` at the top of [diffdef_pkg/goal_generator.py](diffdef_pkg/goal_generator.py). Override it with `--ckpt /path/to/checkpoint.pt`. With `--symlink-install`, edits to the nodes take effect without rebuilding.

Run either node with `--help` to see its options. Paths may use `~`.

### Common options

| Option | Default | Description |
| --- | --- | --- |
| `--ckpt` | `CKPT_PATH` | Trained checkpoint (`.pt`). |
| `--diffdef-root` | `~/retraction_ws/src/DiffDef-core` | DiffDef-core checkout that the model code is imported from. |
| `--device` | `cuda` | `cuda` or `cpu`. Falls back to CPU if CUDA is unavailable. |
| `--sample-num-points` | `512` | Points per cloud after down-sampling, and in the predicted goal. |
| `--sample-latent` | off | Draw a random latent per prediction instead of using the prior mean (all zeros). |

## diffdef_node

Subscribes to a start point cloud and a context point cloud. An `ApproximateTimeSynchronizer` pairs the two clouds by timestamp, and the node publishes a goal cloud for each pair.

| Option | Default | Description |
| --- | --- | --- |
| `--start-topic` | `/tissue_pointcloud` | Start (current shape) point cloud. |
| `--context-topic` | `/context_pointcloud` | Context point cloud. |
| `--goal-topic` | `/goal_pointcloud` | Output goal point cloud. |
| `--sync-slop` | `0.2` | Max timestamp difference (s) between start and context clouds. |
| `--queue-size` | `5` | Synchronizer queue size. |
| `--debug-pickle` | off | If given, pickle the first raw, down-sampled, and goal clouds to this path. |

Each input subscription has a QoS depth of 1. Inference blocks the node, so when it finishes it works on the newest clouds rather than a backlog of old ones.

## diffdef_service_node

Serves `retraction_interface/srv/GenerateGoalPointCloud`. The request has a `tissue_pointcloud` and a `context_pointcloud`. The response has the `goal_pointcloud`, plus `success` and `message`. If either input cloud is empty or inference fails, `success` is false, `message` gives the reason, and `goal_pointcloud` is empty.

| Option | Default | Description |
| --- | --- | --- |
| `--service-name` | `/generate_goal_pointcloud` | Service to serve. |

Requests are handled one at a time.
