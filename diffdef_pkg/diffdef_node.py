#!/usr/bin/env python3
"""Predict a goal point cloud from time-synchronized start and context point clouds."""

import os
import pickle

from diffdef_pkg.goal_generator import (
    generator_from_args, make_arg_parser, numpy_to_pointcloud2, parse_node_args,
    pointcloud2_to_numpy)
from message_filters import ApproximateTimeSynchronizer, Subscriber
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import QoSProfile
from sensor_msgs.msg import PointCloud2


class DiffDefGoalPCNode(Node):

    def __init__(self,
                 generator,
                 start_topic='/tissue_pointcloud',
                 context_topic='/context_pointcloud',
                 goal_topic='/goal_pointcloud',
                 sync_slop=0.2,
                 queue_size=5,
                 debug_pickle_path=''):
        super().__init__('diffdef_goal_pc_node')

        self.generator = generator
        self.start_topic = start_topic
        self.context_topic = context_topic
        self.goal_topic = goal_topic
        self.slop = sync_slop
        self.queue_size = queue_size
        # Pickle the first set of clouds here, if given.
        # expanduser: open() does not expand '~' itself.
        self.debug_pickle_path = os.path.expanduser(debug_pickle_path)
        self.has_saved_debug_pickle = False

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

    def save_pointclouds_before_after_downsampling(
        self,
        start_pc_raw,
        context_pc_raw,
        prediction,
    ):
        if not self.debug_pickle_path or self.has_saved_debug_pickle:
            return

        save_dir = os.path.dirname(self.debug_pickle_path)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)

        save_dict = {
            'start_pc_before_downsampling': start_pc_raw,
            'context_pc_before_downsampling': context_pc_raw,
            'start_pc_after_downsampling_512': prediction.start_downsampled,
            'context_pc_after_downsampling_512': prediction.context_downsampled,
            'goal_pc': prediction.goal,
        }

        with open(self.debug_pickle_path, 'wb') as handle:
            pickle.dump(save_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

        self.has_saved_debug_pickle = True
        self.get_logger().info(f'Saved DiffDef debug point clouds to {self.debug_pickle_path}')

    def synced_callback(self, start_msg, context_msg):
        try:
            start_pc_raw = pointcloud2_to_numpy(start_msg)
            context_pc_raw = pointcloud2_to_numpy(context_msg)

            prediction = self.generator.predict(start_pc_raw, context_pc_raw)
            self.get_logger().info(
                f'PC sizes | start raw: {start_pc_raw.shape[0]}, '
                f'start after ds: {prediction.start_downsampled.shape[0]}, '
                f'context raw: {context_pc_raw.shape[0]}, '
                f'context after ds: {prediction.context_downsampled.shape[0]}')

            self.save_pointclouds_before_after_downsampling(
                start_pc_raw, context_pc_raw, prediction)

            # Stamp the goal with the start cloud's header so it lines up with its inputs
            self.pub_goal.publish(numpy_to_pointcloud2(prediction.goal, start_msg.header))

            self.get_logger().info(
                f'Published goal PC with {prediction.goal.shape[0]} points.',
                throttle_duration_sec=1.0)

        except ValueError as e:
            self.get_logger().warn(f'{e} Skipping.')
        except Exception as e:
            self.get_logger().error(f'Failed to generate goal point cloud: {e}')


def main(args=None):
    rclpy.init(args=args)

    parser = make_arg_parser(
        'Predict a goal point cloud from start and context point clouds with DiffDef.')
    parser.add_argument('--start-topic', default='/tissue_pointcloud')
    parser.add_argument('--context-topic', default='/context_pointcloud')
    parser.add_argument('--goal-topic', default='/goal_pointcloud')
    parser.add_argument('--sync-slop', type=float, default=0.2,
                        help='Max timestamp difference (s) between start and context clouds.')
    parser.add_argument('--queue-size', type=int, default=5, help='Synchronizer queue size.')
    parser.add_argument('--debug-pickle', default='',
                        help='If given, pickle the first raw, down-sampled, and goal clouds here.')
    cli_args = parse_node_args(parser, args)

    node = DiffDefGoalPCNode(
        generator_from_args(cli_args),
        start_topic=cli_args.start_topic,
        context_topic=cli_args.context_topic,
        goal_topic=cli_args.goal_topic,
        sync_slop=cli_args.sync_slop,
        queue_size=cli_args.queue_size,
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
