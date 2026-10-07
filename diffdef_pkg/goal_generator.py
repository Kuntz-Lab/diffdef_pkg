"""DiffDef goal point cloud prediction, shared by the topic and service nodes."""

import argparse
import os
import sys
from typing import NamedTuple

import numpy as np
from rclpy.logging import get_logger
from rclpy.utilities import remove_ros_args
from sensor_msgs_py import point_cloud2 as pc2
import torch

# Trained BaoFlowVAE checkpoint (.pt). Can also be passed with --ckpt.
CKPT_PATH = '~/retraction_ws/src/DiffDef-core/weights_deploy/ckpt_60.399836_30000.pt'
# DiffDef-core checkout. Its models/ package is imported from here.
DIFFDEF_ROOT = '~/retraction_ws/src/DiffDef-core'


def farthest_point_sampling(points, num_points):
    """
    Down-sample an (N, 3) point cloud to (num_points, 3) with farthest point sampling.

    Copied from DiffDef-core utils/misc.py down_sampling(), which the training data was
    processed with. Importing it would pull in that module's open3d dependency.
    """
    n = points.shape[0]
    indices = np.zeros(num_points, dtype=np.int64)
    distance = np.full(n, np.inf)
    farthest = np.random.randint(n)
    for i in range(num_points):
        indices[i] = farthest
        dist = np.sum((points - points[farthest]) ** 2, axis=1)
        np.minimum(distance, dist, out=distance)
        farthest = int(np.argmax(distance))
    return points[indices]


def pointcloud2_to_numpy(msg):
    """Return the finite xyz points of a PointCloud2 as an (N, 3) float32 array."""
    points = pc2.read_points_numpy(msg, field_names=('x', 'y', 'z'), skip_nans=True)
    points = points.astype(np.float32, copy=False)
    return points[np.isfinite(points).all(axis=1)]


def numpy_to_pointcloud2(points, header):
    """Return an (N, 3) array as an xyz PointCloud2 with the given header."""
    return pc2.create_cloud_xyz32(header, points.astype(np.float32, copy=False))


class GoalPrediction(NamedTuple):
    goal: np.ndarray
    start_downsampled: np.ndarray
    context_downsampled: np.ndarray


class GoalGenerator:
    """Predict a goal point cloud from start and context point clouds with DiffDef."""

    def __init__(self,
                 ckpt_path=CKPT_PATH,
                 diffdef_root=DIFFDEF_ROOT,
                 device='cuda',
                 sample_num_points=512,
                 sample_latent=False):
        self.logger = get_logger('diffdef_goal_generator')
        # expanduser: torch.load() does not expand '~' itself
        self.ckpt_path = os.path.expanduser(ckpt_path)
        self.diffdef_root = os.path.expanduser(diffdef_root)
        self.device = device
        self.sample_num_points = sample_num_points
        # Draw a random latent per prediction instead of using the prior mean
        self.sample_latent = sample_latent

        if not self.ckpt_path:
            raise ValueError('No checkpoint given. Set CKPT_PATH or pass --ckpt.')
        if not os.path.isfile(self.ckpt_path):
            raise FileNotFoundError(f'Checkpoint not found: {self.ckpt_path}')

        self.model, self.ckpt_args = self.load_model(self.ckpt_path)

    def load_model(self, ckpt_path):
        if self.device == 'cuda' and not torch.cuda.is_available():
            self.logger.warn('CUDA requested but not available. Falling back to CPU.')
            self.device = 'cpu'

        if self.diffdef_root not in sys.path:
            sys.path.insert(0, self.diffdef_root)
        from models.vae_flow_3 import BaoFlowVAE

        self.logger.info(f'Loading checkpoint: {ckpt_path}')
        # weights_only=False because ckpt['args'] is a pickled argparse.Namespace
        ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)

        model = BaoFlowVAE(ckpt['args']).to(self.device)
        model.load_state_dict(ckpt['state_dict'])
        model.eval()

        self.logger.info(f'Model loaded successfully on {self.device}')
        return model, ckpt['args']

    def predict(self, start_pc, context_pc):
        """
        Down-sample raw (N, 3) start and context clouds and predict the goal cloud.

        Raises ValueError if either cloud is empty.
        """
        if start_pc.shape[0] == 0:
            raise ValueError('Start point cloud is empty.')
        if context_pc.shape[0] == 0:
            raise ValueError('Context point cloud is empty.')

        start_ds = farthest_point_sampling(start_pc, self.sample_num_points)
        context_ds = farthest_point_sampling(context_pc, self.sample_num_points)
        goal = self.predict_downsampled(start_ds, context_ds)
        return GoalPrediction(goal, start_ds, context_ds)

    def predict_downsampled(self, start_pc, context_pc):
        # Normalize both clouds by the start cloud's centroid and spread
        shift = start_pc.mean(axis=0)
        scale = (start_pc - shift).reshape(-1).std() + 1e-8

        start_n = (start_pc - shift) / scale
        context_n = (context_pc - shift) / scale

        start_t = torch.from_numpy(start_n).float().unsqueeze(0).to(self.device)
        context_t = torch.from_numpy(context_n).float().unsqueeze(0).to(self.device)

        with torch.no_grad():
            # Latent in the flow's N(0, I) base space. sample() maps it through the flow.
            if self.sample_latent:
                z = torch.randn(1, self.ckpt_args.latent_dim, device=self.device)
            else:
                z = torch.zeros(1, self.ckpt_args.latent_dim, device=self.device)
            pred_n = self.model.sample(
                z,
                context_t,
                start_t,
                self.sample_num_points,
                flexibility=getattr(self.ckpt_args, 'flexibility', 0.0),
            )[0].cpu().numpy()

        pred_pc = pred_n * scale + shift
        return pred_pc.astype(np.float32)


def make_arg_parser(description):
    """Return an ArgumentParser with the GoalGenerator options. Nodes add their own."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--ckpt', default=CKPT_PATH, help='Trained BaoFlowVAE checkpoint (.pt).')
    parser.add_argument('--diffdef-root', default=DIFFDEF_ROOT,
                        help='DiffDef-core checkout to import the model code from.')
    parser.add_argument('--device', choices=['cuda', 'cpu'], default='cuda')
    parser.add_argument('--sample-num-points', type=int, default=512,
                        help='Points per cloud after down-sampling, and in the predicted goal.')
    parser.add_argument('--sample-latent', action='store_true',
                        help='Draw a random latent per prediction instead of using the '
                             'prior mean (all zeros).')
    return parser


def parse_node_args(parser, args=None):
    """Parse the command line, ignoring ROS arguments such as --ros-args."""
    return parser.parse_args(remove_ros_args(args if args is not None else sys.argv)[1:])


def generator_from_args(cli_args):
    return GoalGenerator(
        ckpt_path=cli_args.ckpt,
        diffdef_root=cli_args.diffdef_root,
        device=cli_args.device,
        sample_num_points=cli_args.sample_num_points,
        sample_latent=cli_args.sample_latent,
    )
