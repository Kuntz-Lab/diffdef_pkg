#!/usr/bin/env python3
"""Predict a goal point cloud from time-synchronized start and context point clouds."""

import argparse
import os
import pickle
import sys

from message_filters import ApproximateTimeSynchronizer, Subscriber
import numpy as np
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import QoSProfile
from rclpy.utilities import remove_ros_args
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2 as pc2
from std_msgs.msg import Header
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


class DiffDefGoalPCNode(Node):

    def __init__(self,
                 ckpt_path=CKPT_PATH,
                 diffdef_root=DIFFDEF_ROOT,
                 device='cuda',
                 sample_num_points=512,
                 start_topic='/tissue_pointcloud',
                 context_topic='/context_pointcloud',
                 goal_topic='/goal_pointcloud',
                 sync_slop=0.2,
                 queue_size=5,
                 mean_latent=False,
                 debug_pickle_path=''):
        super().__init__('diffdef_goal_pc_node')

        # expanduser: open() and torch.load() do not expand '~' themselves
        self.ckpt_path = os.path.expanduser(ckpt_path)
        self.diffdef_root = os.path.expanduser(diffdef_root)
        self.device = device
        self.sample_num_points = sample_num_points
        self.start_topic = start_topic
        self.context_topic = context_topic
        self.goal_topic = goal_topic
        self.slop = sync_slop
        self.queue_size = queue_size
        # Use the prior mean (w = 0) instead of a random latent
        self.mean_latent = mean_latent
        # Pickle the first set of clouds here, if given
        self.debug_pickle_path = os.path.expanduser(debug_pickle_path)
        self.has_saved_debug_pickle = False

        if not self.ckpt_path:
            raise ValueError('No checkpoint given. Set CKPT_PATH or pass --ckpt.')
        if not os.path.isfile(self.ckpt_path):
            raise FileNotFoundError(f'Checkpoint not found: {self.ckpt_path}')

        self.model, self.ckpt_args = self.load_model(self.ckpt_path)

        self.pub_goal = self.create_publisher(PointCloud2, self.goal_topic, 1)

        # Depth 1: inference blocks the executor, so only the newest clouds should be waiting
        # when it finishes, not a backlog of stale ones.
        input_qos = QoSProfile(depth=1)
        self.sub_start = Subscriber(self, PointCloud2, self.start_topic, qos_profile=input_qos)
        self.sub_context = Subscriber(
            self, PointCloud2, self.context_topic, qos_profile=input_qos)

        self.sync = ApproximateTimeSynchronizer(
            [self.sub_start, self.sub_context],
            queue_size=self.queue_size,
            slop=self.slop,
        )
        self.sync.registerCallback(self.synced_callback)

        self.get_logger().info('DiffDef goal PC node is ready.')
        self.get_logger().info(
            f'Subscribing to: {self.start_topic} and {self.context_topic}')
        self.get_logger().info(f'Publishing generated goal PC to: {self.goal_topic}')

    def load_model(self, ckpt_path):
        if self.device == 'cuda' and not torch.cuda.is_available():
            self.get_logger().warn('CUDA requested but not available. Falling back to CPU.')
            self.device = 'cpu'

        if self.diffdef_root not in sys.path:
            sys.path.insert(0, self.diffdef_root)
        from models.vae_flow_3 import BaoFlowVAE

        self.get_logger().info(f'Loading checkpoint: {ckpt_path}')
        # weights_only=False because ckpt['args'] is a pickled argparse.Namespace
        ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)

        model = BaoFlowVAE(ckpt['args']).to(self.device)
        model.load_state_dict(ckpt['state_dict'])
        model.eval()

        self.get_logger().info(f'Model loaded successfully on {self.device}')
        return model, ckpt['args']

    def pointcloud2_to_numpy(self, msg):
        points = pc2.read_points_numpy(msg, field_names=('x', 'y', 'z'), skip_nans=True)
        points = points.astype(np.float32, copy=False)
        return points[np.isfinite(points).all(axis=1)]

    def numpy_to_pointcloud2(self, points, header):
        return pc2.create_cloud_xyz32(header, points.astype(np.float32, copy=False))

    def save_pointclouds_before_after_downsampling(
        self,
        start_pc_raw,
        context_pc_raw,
        start_pc_ds,
        context_pc_ds,
        goal_pc,
    ):
        if not self.debug_pickle_path or self.has_saved_debug_pickle:
            return

        save_dir = os.path.dirname(self.debug_pickle_path)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)

        save_dict = {
            'start_pc_before_downsampling': start_pc_raw,
            'context_pc_before_downsampling': context_pc_raw,
            'start_pc_after_downsampling_512': start_pc_ds,
            'context_pc_after_downsampling_512': context_pc_ds,
            'goal_pc': goal_pc,
        }

        with open(self.debug_pickle_path, 'wb') as handle:
            pickle.dump(save_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

        self.has_saved_debug_pickle = True
        self.get_logger().info(f'Saved DiffDef debug point clouds to {self.debug_pickle_path}')

    def synced_callback(self, start_msg, context_msg):
        try:
            start_pc_raw = self.pointcloud2_to_numpy(start_msg)
            context_pc_raw = self.pointcloud2_to_numpy(context_msg)

            if start_pc_raw.shape[0] == 0:
                self.get_logger().warn('Received empty start point cloud. Skipping.')
                return

            if context_pc_raw.shape[0] == 0:
                self.get_logger().warn('Received empty context point cloud. Skipping.')
                return

            start_pc_ds = farthest_point_sampling(start_pc_raw, self.sample_num_points)
            context_pc_ds = farthest_point_sampling(context_pc_raw, self.sample_num_points)
            self.get_logger().info(
                f'PC sizes | start raw: {start_pc_raw.shape[0]}, '
                f'start after ds: {start_pc_ds.shape[0]}, '
                f'context raw: {context_pc_raw.shape[0]}, '
                f'context after ds: {context_pc_ds.shape[0]}')

            goal_pc = self.generate_goal_pc(start_pc_ds, context_pc_ds)
            self.save_pointclouds_before_after_downsampling(
                start_pc_raw,
                context_pc_raw,
                start_pc_ds,
                context_pc_ds,
                goal_pc,
            )

            # Stamp the goal with the start cloud's time so it lines up with its inputs
            header = Header(stamp=start_msg.header.stamp, frame_id=start_msg.header.frame_id)
            self.pub_goal.publish(self.numpy_to_pointcloud2(goal_pc, header))

            self.get_logger().info(
                f'Published goal PC with {goal_pc.shape[0]} points. '
                f'Start: {start_pc_ds.shape[0]}, Context: {context_pc_ds.shape[0]}',
                throttle_duration_sec=1.0)

        except Exception as e:
            self.get_logger().error(f'Failed to generate goal point cloud: {e}')

    def generate_goal_pc(self, start_pc, context_pc):
        shift = start_pc.mean(axis=0)
        scale = (start_pc - shift).reshape(-1).std() + 1e-8

        start_n = (start_pc - shift) / scale
        context_n = (context_pc - shift) / scale

        start_t = torch.from_numpy(start_n).float().unsqueeze(0).to(self.device)
        context_t = torch.from_numpy(context_n).float().unsqueeze(0).to(self.device)

        with torch.no_grad():
            # Latent in the flow's N(0, I) base space. sample() maps it through the flow.
            if self.mean_latent:
                z = torch.zeros(1, self.ckpt_args.latent_dim, device=self.device)
            else:
                z = torch.randn(1, self.ckpt_args.latent_dim, device=self.device)
            pred_n = self.model.sample(
                z,
                context_t,
                start_t,
                self.sample_num_points,
                flexibility=getattr(self.ckpt_args, 'flexibility', 0.0),
            )[0].cpu().numpy()

        pred_pc = pred_n * scale + shift
        return pred_pc.astype(np.float32)


def parse_args(argv):
    parser = argparse.ArgumentParser(
        description='Predict a goal point cloud from start and context point clouds with DiffDef.')
    parser.add_argument('--ckpt', default=CKPT_PATH, help='Trained BaoFlowVAE checkpoint (.pt).')
    parser.add_argument('--diffdef-root', default=DIFFDEF_ROOT,
                        help='DiffDef-core checkout to import the model code from.')
    parser.add_argument('--device', choices=['cuda', 'cpu'], default='cuda')
    parser.add_argument('--sample-num-points', type=int, default=512,
                        help='Points per cloud after down-sampling, and in the predicted goal.')
    parser.add_argument('--start-topic', default='/tissue_pointcloud')
    parser.add_argument('--context-topic', default='/context_pointcloud')
    parser.add_argument('--goal-topic', default='/goal_pointcloud')
    parser.add_argument('--sync-slop', type=float, default=0.2,
                        help='Max timestamp difference (s) between start and context clouds.')
    parser.add_argument('--queue-size', type=int, default=5, help='Synchronizer queue size.')
    parser.add_argument('--mean-latent', action='store_true',
                        help='Use the prior mean latent (all zeros) instead of a random one.')
    parser.add_argument('--debug-pickle', default='',
                        help='If given, pickle the first raw, down-sampled, and goal clouds here.')
    return parser.parse_args(argv)


def main(args=None):
    rclpy.init(args=args)
    cli_args = parse_args(remove_ros_args(args if args is not None else sys.argv)[1:])

    node = DiffDefGoalPCNode(
        ckpt_path=cli_args.ckpt,
        diffdef_root=cli_args.diffdef_root,
        device=cli_args.device,
        sample_num_points=cli_args.sample_num_points,
        start_topic=cli_args.start_topic,
        context_topic=cli_args.context_topic,
        goal_topic=cli_args.goal_topic,
        sync_slop=cli_args.sync_slop,
        queue_size=cli_args.queue_size,
        mean_latent=cli_args.mean_latent,
        debug_pickle_path=cli_args.debug_pickle,
    )
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
