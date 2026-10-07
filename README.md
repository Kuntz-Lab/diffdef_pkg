# diffdef_pkg

ROS 2 (Jazzy) node that predicts a goal point cloud with DiffDef.

## diffdef_node

Subscribes to a start point cloud (default `/tissue_pointcloud`) and a context point cloud (default `/context_pointcloud`). An `ApproximateTimeSynchronizer` pairs the two clouds by timestamp. For each pair, the node:

1. down-samples both clouds to `sample_num_points` (512) with farthest point sampling,
2. runs the DiffDef `BaoFlowVAE` model,
3. publishes the predicted goal cloud on `/goal_pointcloud`.

The goal cloud uses the start cloud's `frame_id` and timestamp.

The model code is imported from a DiffDef-core checkout (`models/vae_flow_3.py`). By default this is `~/retraction_ws/src/DiffDef-core`.

### Setup

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

Build with plain `colcon`. The venv doesn't need to be active:

```bash
source /opt/ros/jazzy/setup.bash
cd ~/retraction_ws
colcon build --symlink-install --packages-select diffdef_pkg
```

The `diffdef_node` launcher's shebang is `#!/usr/bin/env python3`, so it runs on whichever `python3` is first on `PATH` at run time. `setup.py` sets this, because `--symlink-install` would otherwise hard-code the Python that ran colcon.

### Running

Always activate the venv first.

```bash
source /opt/ros/jazzy/setup.bash
source ~/venvs/diffdef/bin/activate
source ~/retraction_ws/install/setup.bash

ros2 run diffdef_pkg diffdef_node
```

The checkpoint path is set by `CKPT_PATH` at the top of [diffdef_pkg/diffdef_node.py](diffdef_pkg/diffdef_node.py). Override it with `--ckpt /path/to/checkpoint.pt`. With `--symlink-install`, edits to the node take effect without rebuilding.

### Options

| Option | Default | Description |
| --- | --- | --- |
| `--ckpt` | `CKPT_PATH` | Trained checkpoint (`.pt`). |
| `--diffdef-root` | `~/retraction_ws/src/DiffDef-core` | DiffDef-core checkout that the model code is imported from. |
| `--device` | `cuda` | `cuda` or `cpu`. Falls back to CPU if CUDA is unavailable. |
| `--sample-num-points` | `512` | Points per cloud after down-sampling, and in the predicted goal. |
| `--start-topic` | `/tissue_pointcloud` | Start (current shape) point cloud. |
| `--context-topic` | `/context_pointcloud` | Context point cloud. |
| `--goal-topic` | `/goal_pointcloud` | Output goal point cloud. |
| `--sync-slop` | `0.2` | Max timestamp difference (s) between start and context clouds. |
| `--queue-size` | `5` | Synchronizer queue size. |
| `--debug-pickle` | off | If given, pickle the first raw, down-sampled, and goal clouds to this path. |

Run with `--help` to see the same list. Paths may use `~`.

Each input subscription has a QoS depth of 1. Inference blocks the node, so when it finishes it works on the newest clouds rather than a backlog of old ones.
