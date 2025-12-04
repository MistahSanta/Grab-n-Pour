#!/usr/bin/env python3
"""
MoveIt2 Integration for Container 6D Pose Estimation (ROS2 Jazzy)
Designed for SO101 robotic arm liquid pouring application

This module handles:
1. Publishing detected poses to ROS2 topics
2. Transforming poses between frames (camera → robot base)
3. Planning and executing motions to reach the container (via MoveIt2)
4. Grasp pose generation for picking up containers

IMPORTANT: This uses MoveIt2's new Python API (moveit_py)

For detailed setup instructions, see ROS2_JAZZY_GUIDE.md
"""

import numpy as np
from scipy.spatial.transform import Rotation as R
from dataclasses import dataclass
from typing import Optional, List, Tuple
import time
import yaml
import threading

# ROS2 imports
try:
    import rclpy
    from rclpy.node import Node
    from rclpy.duration import Duration
    
    import tf2_ros
    from tf2_ros import TransformException
    from geometry_msgs.msg import Pose, PoseStamped, Point, Quaternion
    from std_msgs.msg import Header
    
    # MoveIt2 imports
    try:
        #from moveit.planning import MoveItPy, PlanningComponent 
        from pymoveit2 import MoveIt2
        MOVEIT_AVAILABLE = True
    except ImportError:
        MOVEIT_AVAILABLE = False
        print("MoveIt2 Python interface not available.")
        print("Install with: sudo apt install ros-jazzy-moveit-py")
    
    ROS_AVAILABLE = True
except ImportError:
    ROS_AVAILABLE = False
    MOVEIT_AVAILABLE = False
    print("ROS2 not available. Install ROS2 Jazzy.")


@dataclass
class GraspConfig:
    """Configuration for grasp planning"""
    approach_direction: np.ndarray = None
    grasp_offset: np.ndarray = None
    pre_grasp_distance: float = 0.10
    gripper_orientation: np.ndarray = None
    
    def __post_init__(self):
        if self.approach_direction is None:
            self.approach_direction = np.array([0, 0, -1])
        if self.grasp_offset is None:
            self.grasp_offset = np.array([0, 0, 0.05])
        if self.gripper_orientation is None:
            self.gripper_orientation = R.from_euler('xyz', [180, 0, 0], degrees=True).as_quat()


class CameraToRobotTransform(Node):
    """Handles transformation from camera frame to robot base frame using TF2."""
    
    def __init__(self, transform_matrix: np.ndarray = None,
                 camera_frame: str = "camera_frame",
                 robot_frame: str = "base_link"):
        super().__init__('camera_to_robot_transform')
        
        self.camera_frame = camera_frame
        self.robot_frame = robot_frame
        
        self.transform_matrix = transform_matrix if transform_matrix is not None else np.eye(4)
        if transform_matrix is None:
            self.get_logger().warning('Using identity transform. Calibrate camera-to-robot!')
        
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
    
    def transform_pose(self, pose_in_camera: PoseStamped) -> Optional[PoseStamped]:
        """Transform pose from camera to robot frame"""
        try:
            transform = self.tf_buffer.lookup_transform(
                self.robot_frame,
                self.camera_frame,
                rclpy.time.Time(),
                timeout=Duration(seconds=1.0)
            )
            
            # Manual transformation
            pos = pose_in_camera.pose.position
            ori = pose_in_camera.pose.orientation
            
            trans = transform.transform.translation
            rot = transform.transform.rotation
            
            tf_matrix = np.eye(4)
            tf_matrix[:3, :3] = R.from_quat([rot.x, rot.y, rot.z, rot.w]).as_matrix()
            tf_matrix[:3, 3] = [trans.x, trans.y, trans.z]
            
            pose_matrix = np.eye(4)
            pose_matrix[:3, :3] = R.from_quat([ori.x, ori.y, ori.z, ori.w]).as_matrix()
            pose_matrix[:3, 3] = [pos.x, pos.y, pos.z]
            
            transformed_matrix = tf_matrix @ pose_matrix
            
            transformed_pos = transformed_matrix[:3, 3]
            transformed_rot = R.from_matrix(transformed_matrix[:3, :3]).as_quat()
            
            pose_in_robot = PoseStamped()
            pose_in_robot.header.frame_id = self.robot_frame
            pose_in_robot.header.stamp = self.get_clock().now().to_msg()
            pose_in_robot.pose.position = Point(x=float(transformed_pos[0]),
                                                y=float(transformed_pos[1]),
                                                z=float(transformed_pos[2]))
            pose_in_robot.pose.orientation = Quaternion(x=float(transformed_rot[0]),
                                                        y=float(transformed_rot[1]),
                                                        z=float(transformed_rot[2]),
                                                        w=float(transformed_rot[3]))
            
            return pose_in_robot
            
        except TransformException as e:
            self.get_logger().warning(f'TF lookup failed: {e}')
            return None


class MoveIt2Controller(Node):
    """MoveIt2 interface using PyMoveit2"""
    
    def __init__(self, arm_group_name: str = "arm",
                 gripper_group_name: str = "gripper"):
        super().__init__('moveit2_controller')
        
        if not MOVEIT_AVAILABLE:
            raise RuntimeError("MoveIt2 not available")
        
        self.get_logger().info('Initializing MoveIt2...')

        robot_name = "so101"
        package_name = "lerobot_moveit"




        self.moveit = MoveItPy(
            node_name="moveitpy",
        )
        self.arm = self.moveit.get_planning_component(arm_group_name)
        
        try:
            self.gripper = self.moveit.get_planning_component(gripper_group_name)
            self.gripper_available = True
        except:
            self.gripper_available = False
            self.get_logger().warning(f'Gripper "{gripper_group_name}" not found')
        
        self.get_logger().info(f'MoveIt2 ready - Arm: {arm_group_name}')
    
    def plan_and_execute(self, target_pose: PoseStamped) -> bool:
        """Plan and execute to target pose"""
        self.arm.set_goal_state(pose_stamped_msg=target_pose)
        plan_result = self.arm.plan()
        
        if plan_result:
            return self.arm.execute()
        return False
    
    def go_to_named_target(self, target_name: str) -> bool:
        """Move to named target"""
        self.arm.set_goal_state(configuration_name=target_name)
        plan_result = self.arm.plan()
        if plan_result:
            return self.arm.execute()
        return False
    
    def open_gripper(self) -> bool:
        if not self.gripper_available:
            return False
        self.gripper.set_goal_state(configuration_name="open")
        plan_result = self.gripper.plan()
        return self.gripper.execute() if plan_result else False
    
    def close_gripper(self) -> bool:
        if not self.gripper_available:
            return False
        self.gripper.set_goal_state(configuration_name="closed")
        plan_result = self.gripper.plan()
        return self.gripper.execute() if plan_result else False


class ContainerGraspPlanner:
    """Plans grasp poses"""
    
    def __init__(self, config: GraspConfig = None):
        self.config = config or GraspConfig()
    
    def compute_grasp_pose(self, container_pose: PoseStamped) -> PoseStamped:
        pos = np.array([container_pose.pose.position.x,
                        container_pose.pose.position.y,
                        container_pose.pose.position.z])
        ori = container_pose.pose.orientation
        container_rot = R.from_quat([ori.x, ori.y, ori.z, ori.w])
        
        offset_world = container_rot.apply(self.config.grasp_offset)
        grasp_pos = pos + offset_world
        grasp_quat = self.config.gripper_orientation
        
        grasp_pose = PoseStamped()
        grasp_pose.header = container_pose.header
        grasp_pose.pose.position = Point(x=float(grasp_pos[0]),
                                         y=float(grasp_pos[1]),
                                         z=float(grasp_pos[2]))
        grasp_pose.pose.orientation = Quaternion(x=float(grasp_quat[0]),
                                                 y=float(grasp_quat[1]),
                                                 z=float(grasp_quat[2]),
                                                 w=float(grasp_quat[3]))
        return grasp_pose
    
    def compute_pre_grasp_pose(self, grasp_pose: PoseStamped) -> PoseStamped:
        pos = np.array([grasp_pose.pose.position.x,
                        grasp_pose.pose.position.y,
                        grasp_pose.pose.position.z])
        
        pre_pos = pos - self.config.approach_direction * self.config.pre_grasp_distance
        
        pre_grasp_pose = PoseStamped()
        pre_grasp_pose.header = grasp_pose.header
        pre_grasp_pose.pose.position = Point(x=float(pre_pos[0]),
                                             y=float(pre_pos[1]),
                                             z=float(pre_pos[2]))
        pre_grasp_pose.pose.orientation = grasp_pose.pose.orientation
        return pre_grasp_pose


class PoseEstimatorMoveIt2Bridge(Node):
    """Main integration bridging pose estimation with MoveIt2"""
    
    def __init__(self, arm_group: str = "arm", gripper_group: str = "gripper",
                 camera_frame: str = "camera_link", robot_frame: str = "base_link"):
        super().__init__('pose_estimator_moveit2_bridge')
        
        self.get_logger().info('Initializing MoveIt2 bridge...')
        
        self.transform_handler = CameraToRobotTransform(
            camera_frame=camera_frame,
            robot_frame=robot_frame
        )
        
        self.moveit = MoveIt2Controller(arm_group, gripper_group)
        self.grasp_planner = ContainerGraspPlanner()
        
        self.latest_pose = None
        self.pose_lock = threading.Lock()
        
        self.pose_sub = self.create_subscription(
            PoseStamped,
            '/container_pose',
            self._pose_callback,
            10
        )
        
        self.get_logger().info('MoveIt2 bridge ready')
    
    def _pose_callback(self, msg: PoseStamped):
        with self.pose_lock:
            self.latest_pose = self.transform_handler.transform_pose(msg)
    
    def get_latest_pose(self) -> Optional[PoseStamped]:
        with self.pose_lock:
            return self.latest_pose
    
    def go_home(self) -> bool:
        self.get_logger().info("Going home...")
        return self.moveit.go_to_named_target("home")
    
    def pick_container(self) -> bool:
        pose = self.get_latest_pose()
        if pose is None:
            self.get_logger().error("No pose available")
            return False
        
        grasp = self.grasp_planner.compute_grasp_pose(pose)
        pre_grasp = self.grasp_planner.compute_pre_grasp_pose(grasp)
        
        self.moveit.open_gripper()
        
        if not self.moveit.plan_and_execute(pre_grasp):
            return False
        
        if not self.moveit.plan_and_execute(grasp):
            return False
        
        self.moveit.close_gripper()
        self.get_logger().info("Pick complete")
        return True


def main():
    rclpy.init()
    
    try:
        bridge = PoseEstimatorMoveIt2Bridge(
            arm_group="arm",
            gripper_group="gripper"
        )
        
        bridge.go_home()
        
        print("Waiting for container...")
        while rclpy.ok() and bridge.get_latest_pose() is None:
            rclpy.spin_once(bridge, timeout_sec=0.1)
        
        if bridge.get_latest_pose():
            bridge.pick_container()
    
    except KeyboardInterrupt:
        pass
    finally:
        rclpy.shutdown()


if __name__ == "__main__":
    main()