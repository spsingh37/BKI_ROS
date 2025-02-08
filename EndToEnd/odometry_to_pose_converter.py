import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped
import math
from tf_transformations import quaternion_from_euler, euler_from_quaternion


class OdometryToPoseConverter(Node):
    def __init__(self):
        super().__init__('odometry_to_pose_converter')

        # Variables to store the initial pose
        self.initial_pose = None

        # Subscriber for Odometry messages
        self.odometry_subscription = self.create_subscription(
            Odometry,
            '/wamv/sensors/position/ground_truth_odometry',  # Change this to your Odometry topic
            self.odometry_callback,
            10
        )

        # Publisher for PoseStamped messages
        self.pose_publisher = self.create_publisher(PoseStamped, '/gt_pose', 10)
        self.get_logger().info('Odometry to PoseStamped Converter Node Initialized')

    def odometry_callback(self, msg: Odometry):
        # If the initial pose is not set, use the first odometry data as the origin
        if self.initial_pose is None:
            self.initial_pose = msg.pose.pose

        # Compute relative position
        relative_x = msg.pose.pose.position.x - self.initial_pose.position.x
        relative_y = msg.pose.pose.position.y - self.initial_pose.position.y
        relative_z = msg.pose.pose.position.z - self.initial_pose.position.z

        # Rotate the position by 90 degrees clockwise around the Z-axis
        rotated_x = relative_y  # Clockwise rotation: x' = y
        rotated_y = -relative_x  # Clockwise rotation: y' = -x

        # Convert quaternion to Euler angles
        quaternion = [
            msg.pose.pose.orientation.x,
            msg.pose.pose.orientation.y,
            msg.pose.pose.orientation.z,
            msg.pose.pose.orientation.w,
        ]
        roll, pitch, yaw = euler_from_quaternion(quaternion)

        # Add a 90-degree clockwise rotation (negative 90 degrees in radians)
        rotated_yaw = yaw - math.pi / 2

        # Normalize yaw to stay within [-pi, pi]
        # rotated_yaw = (rotated_yaw + math.pi) % (2 * math.pi) - math.pi

        # Convert back to quaternion
        rotated_quaternion = quaternion_from_euler(roll, pitch, rotated_yaw)

        # Populate PoseStamped message
        relative_pose = PoseStamped()
        relative_pose.header.stamp = msg.header.stamp  # Use timestamp from Odometry data
        relative_pose.header.frame_id = msg.header.frame_id  # Use same frame as odometry
        relative_pose.pose.position.x = rotated_x
        relative_pose.pose.position.y = rotated_y
        relative_pose.pose.position.z = relative_z
        relative_pose.pose.orientation.x = rotated_quaternion[0]
        relative_pose.pose.orientation.y = rotated_quaternion[1]
        relative_pose.pose.orientation.z = rotated_quaternion[2]
        relative_pose.pose.orientation.w = rotated_quaternion[3]
        # relative_pose.pose.orientation = msg.pose.pose.orientation

        # Publish the PoseStamped message
        self.pose_publisher.publish(relative_pose)
        self.get_logger().info(f'Published PoseStamped: {relative_pose}')


def main(args=None):
    rclpy.init(args=args)
    node = OdometryToPoseConverter()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Node terminated by user')
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
