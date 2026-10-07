#!/usr/bin/env python3
"""Serve goal point cloud predictions from start (tissue) and context point clouds."""

from diffdef_pkg.goal_generator import (
    generator_from_args, make_arg_parser, numpy_to_pointcloud2, parse_node_args,
    pointcloud2_to_numpy)
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from retraction_interface.srv import GenerateGoalPointCloud


class DiffDefGoalPCService(Node):

    def __init__(self, generator, service_name='/generate_goal_pointcloud'):
        super().__init__('diffdef_goal_pc_service')

        self.generator = generator
        self.service_name = service_name
        self.srv = self.create_service(
            GenerateGoalPointCloud, self.service_name, self.handle_request)

        self.get_logger().info(f'DiffDef goal PC service is ready on: {self.service_name}')

    def handle_request(self, request, response):
        try:
            prediction = self.generator.predict(
                pointcloud2_to_numpy(request.tissue_pointcloud),
                pointcloud2_to_numpy(request.context_pointcloud))
        except Exception as e:
            response.success = False
            response.message = f'Failed to generate goal point cloud: {e}'
            self.get_logger().error(response.message)
            return response

        # Stamp the goal with the tissue cloud's header so it lines up with its inputs
        response.goal_pointcloud = numpy_to_pointcloud2(
            prediction.goal, request.tissue_pointcloud.header)
        response.success = True
        response.message = f'Generated goal PC with {prediction.goal.shape[0]} points.'
        self.get_logger().info(response.message)
        return response


def main(args=None):
    rclpy.init(args=args)

    parser = make_arg_parser(
        'Serve DiffDef goal point cloud predictions from tissue and context point clouds.')
    parser.add_argument('--service-name', default='/generate_goal_pointcloud')
    cli_args = parse_node_args(parser, args)

    node = DiffDefGoalPCService(
        generator_from_args(cli_args), service_name=cli_args.service_name)
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
