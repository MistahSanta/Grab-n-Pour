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
from main_irl import (
    get_current_joint_states, 
    get_message_once, 
    extract_specific_joints, 
    rotX, rotY, rotZ, 
    mat2quat # Temporary
)
from threading import Thread
from typing import Any 
import time
import yaml
import threading

# ROS2 imports
try:
    import rclpy
    from rclpy.node import Node
    from rclpy.duration import Duration
    from rclpy.callback_groups import ReentrantCallbackGroup
    from rclpy.task import Future
    
    import tf2_ros
    from tf2_ros import TransformException, TransformBroadcaster
    from geometry_msgs.msg import Pose, PoseStamped, Point, Quaternion, TransformStamped
    from std_msgs.msg import Header
    
    from lerobot_robot_ros.config import SO101ROSConfig
    from lerobot.robots.so101_follower import SO101Follower, SO101FollowerConfig
    # MoveIt2 imports
    try:
        #from moveit.planning import MoveItPy, PlanningComponent 
        from pymoveit2 import MoveIt2
        from pymoveit2 import GripperInterface
        
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






class CameraToRobotTransform(Node):
    """Handles transformation from camera frame to robot base frame using TF2."""
    
    def __init__(self, transform_matrix: np.ndarray = None,
                 camera_frame: str = "camera_frame",
                 robot_frame: str = "base"):
        
        super().__init__('camera_to_robot_transform')
        
        self.static_broadcaster = tf2_ros.StaticTransformBroadcaster(self)

        # give time for TF to populate 
        # for _ in range(20):
        #     rclpy.spin_once(self, timeout_sec=0.1)

        self.camera_frame = camera_frame
        self.robot_frame = robot_frame

        # self.static_broadcaster = tf2_ros.StaticTransformBroadcaster(self)


        # # Calibrate the camera 
        translation = [-0.2032, -0.2032, 0.381] # in meters 
        rotation = [0, 180, 180]

        self._publish_camera_tf(translation, rotation)

 


        self.transform_matrix = transform_matrix if transform_matrix is not None else np.eye(4)
        if transform_matrix is None:
            self.get_logger().warning('Using identity transform. Calibrate camera-to-robot!')
        
        
        self.tf_buffer = tf2_ros.Buffer()
        print(self.get_clock().now())
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.get_logger().info(f'Camera TF published:')
        self.get_logger().info(f'  Translation: {translation}')
        self.get_logger().info(f'  Rotation (RPY deg): {rotation}')
    
    def _publish_camera_tf(self, translation: list, rotation_euler_deg: list):
        """Publish static transform from base to camera"""
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = "base"
        t.child_frame_id = self.camera_frame
        
        # Set translation
        t.transform.translation.x = translation[0]
        t.transform.translation.y = translation[1]
        t.transform.translation.z = translation[2]
        
        # Convert euler to quaternion
        quat = R.from_euler("xyz", rotation_euler_deg, degrees=True).as_quat()
        t.transform.rotation.x = quat[0]
        t.transform.rotation.y = quat[1]
        t.transform.rotation.z = quat[2]
        t.transform.rotation.w = quat[3]
        
        self.static_broadcaster.sendTransform(t)

    def transform_loc(self, loc_in_camera: Point) -> Point:
        x = -loc_in_camera.x * 0.5
        y = -(1.0 - loc_in_camera.y) * 0.3

        ret = Point()
        ret.x = x
        ret.y = y

        return ret

    def transform_pose(self, pose_in_camera: PoseStamped) -> Optional[PoseStamped]:
        """Transform pose from camera to robot frame"""
        try:

            transform = self.tf_buffer.lookup_transform(
                self.robot_frame,
                self.camera_frame,
                rclpy.time.Time(),
                #self.get_clock().now(),
                timeout=Duration(seconds=1.0)
            )
            print("transform: ", transform, flush=True)
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

        self.node_arm = Node("Grab_N_Pour")
        irl_robot_config = SO101FollowerConfig(
            port="/dev/ttyACM0",
            id="my_awesome_follower_arm",
            use_degrees=True,
            max_relative_target=200.0,
        )
        self.irl_robot = SO101Follower(irl_robot_config)
        self.irl_robot.connect()

        self.robot_config = SO101ROSConfig()

        self.arm_joint_names = self.robot_config.ros2_interface.arm_joint_names

        # Planner ID
        self.node_arm.declare_parameter("planner_id", "RRTConnectkConfigDefault")

        # Create callback group that allows execution of callbacks in parallel without restrictions
        callback_group = ReentrantCallbackGroup()
  

        self.moveit2 = MoveIt2(
            node=self.node_arm,
            joint_names=self.arm_joint_names,
            base_link_name=self.robot_config.ros2_interface.base_link,
            end_effector_name="jaw",
            group_name="arm",
            callback_group=callback_group,
        )

        # Create gripper interface
        # node_gripper = Node("fetch_gripper")
        # self.gripper = GripperInterface(
        #     node=node_gripper,
        #     gripper_joint_names=[self.robot_config.ros2_interface.gripper_joint_name],
        #     open_gripper_joint_positions=[
        #         self.robot_config.ros2_interface.gripper_open_position
        #     ],
        #     closed_gripper_joint_positions=[
        #         self.robot_config.ros2_interface.gripper_close_position
        #     ],
        #     gripper_group_name="gripper",
        #     callback_group=callback_group,
        #     gripper_command_action_name="gripper_controller/gripper_cmd",
        # )

        #self.arm = arm_group_name
        


    def move_robot_to(
        self, 
        joint_angles_rad: List[float],
    ):
        """Moves the SO101 robot to the provided joint angles, in radians.

        `arm_configuration` should be a list of 5 floats.
        """

        # Move robot in gazebo
        # self.moveit2.move_to_configuration(arm_configuration)
        # self.moveit2.wait_until_executed()


        # time.sleep(3)

        # #get the current robot joints
        # joint_positions = get_current_joint_states(self.node_arm)
        # print("Joint position", joint_positions)

        (j1, j2, j3, j4, j5) = joint_angles_rad
        print("joint_angles_rad", joint_angles_rad)

        action = {
            "shoulder_pan.pos": np.rad2deg(j1),
            "shoulder_lift.pos": np.rad2deg(j2),
            "elbow_flex.pos": np.rad2deg(j3),
            "wrist_flex.pos": np.rad2deg(j4),
            "wrist_roll.pos": np.rad2deg(j5),
        }

        self.irl_robot.send_action(action)

        time.sleep(5.0)
    
    def get_robot_current_pos(self) -> dict[str, Any]: 
        return self.irl_robot.get_observation()
    
    def open_gripper(self):
        # ! This only open the gripper in sim, not irl 
        # if not self.gripper:
        #     print('no gripper interface! ')
        #     return False
        # self.gripper.open()
        # self.gripper.wait_until_executed()

        self.irl_robot.send_action({"gripper.pos": 90})
        time.sleep(1) 
    
    def close_gripper(self):
        # if not self.gripper:
        #     print('no gripper interface! ')
        #     return False
        # self.gripper.close()
        # self.gripper.wait_until_executed()
        
        self.irl_robot.send_action({"gripper.pos": 0})
        time.sleep(1) 


    def convert_joint_angle_rad_2_deg(self, joint_angles_rad: dict[str, Any]):
        return {j: np.rad2deg(v) for j,v in joint_angles_rad.items()}

    def move_joint_slowly(
        self, 
        joint_name: str,
        joint_degrees: int, 
        step_deg=1.0,
        delay=0.03,
    ):
        
        current_joints =  self.get_robot_current_pos()
        current_joint_deg = current_joints[joint_name]

        gap = abs(current_joint_deg - joint_degrees) 
        step_dir  = -1 if joint_degrees < current_joint_deg else 1 

        num_steps = int(gap / step_deg)

        for i in range(abs(num_steps)):
            current_joints[joint_name] = current_joint_deg + (i * step_dir)
            step_command = current_joints
            #print("sending step command: ", step_command, flush=True)
            self.irl_robot.send_action(step_command)
            time.sleep(delay)
    
    def move_multiple_joints_slowly(
        self, 
        target_joints_rad: list[float], # Expect joints to be in rad!
        step_deg=1.0,
        delay=0.03,
    ):
        print('target joint', target_joints_rad)
        j1, j2, j3, j4, j5 = target_joints_rad 

        target_joints = {
            "shoulder_pan.pos": np.rad2deg(j1),
            "shoulder_lift.pos": np.rad2deg(j2),
            "elbow_flex.pos": np.rad2deg(j3),
            "wrist_flex.pos": np.rad2deg(j4),
            "wrist_roll.pos": np.rad2deg(j5),
        }


        current_joints =  self.get_robot_current_pos()
        print("current joints", current_joints)        
        # Find the joints that are different 
        tolerance = 2
        joints_that_changed = [
            j for j in target_joints 
            if abs(target_joints[j] - current_joints[j]) > tolerance
        ] 
        
        if not joints_that_changed:
            print("no joints need to move.")
            return 

        max_gap = max(abs(target_joints[j] - current_joints[j]) for j in joints_that_changed)
        num_steps = max(1, int(max_gap / step_deg))

        # perform smooth movement 
        for step in range(1, num_steps + 1 ): 
            step_cmd_deg = current_joints.copy()

            for j in joints_that_changed:
                start = current_joints[j]
                target = target_joints[j]

                t = step / num_steps
                step_cmd_deg[j] = start + t * (target - start)

            print("sending: ", step_cmd_deg)
            # Send to robot
            self.irl_robot.send_action(step_cmd_deg)
            time.sleep(delay)

class PoseEstimatorMoveIt2Bridge(Node):
    """Main integration bridging pose estimation with MoveIt2"""
    
    def __init__(self, arm_group: str = "arm", gripper_group: str = "gripper",
                 camera_frame: str = "camera_link", robot_frame: str = "base_link"):
        super().__init__('pose_estimator_moveit2_bridge')
        
        self.get_logger().info('Initializing MoveIt2 bridge...')
        
        self.transform_handler = CameraToRobotTransform(
            camera_frame="camera_frame",
            robot_frame="base"
        )
        
        self.moveit = MoveIt2Controller(arm_group, gripper_group)
        #self.grasp_planner = ContainerGraspPlanner()
        
        self.latest_pose = None
        self.pose_lock = threading.Lock()

        self.latest_loc = None
        self.loc_lock = threading.Lock()
        
        self.pose_sub = self.create_subscription(
            PoseStamped,
            '/container_pose',
            self._pose_callback,
            10
        )

        self.loc_sub = self.create_subscription(
            Point,
            "/container_loc",
            self._loc_callback,
            10
        )
        
        self.get_logger().info('MoveIt2 bridge ready')
    


    def compute_above_cup(self, bottle_pos: Point): 
        x_cup, y_cup, _ = bottle_pos.x, bottle_pos.y, bottle_pos.z 

        x_offset = (x_cup) * 0.65
        y_offset = (y_cup) * 0.75

        p = np.array([x_offset, y_offset, .275])

        rotation = np.matmul(rotX(np.deg2rad(-90)), rotY(np.deg2rad(-90)))
        R_standoff = np.matmul(rotZ(-(np.deg2rad(90) - np.atan2(x_cup, -y_cup))), rotation)[
            :3, :3
        ]

        q_wxyz = mat2quat(R_standoff)
        q_xyzw = np.array([q_wxyz[1], q_wxyz[2], q_wxyz[3], q_wxyz[0]])

        return (p, q_xyzw)

    def compute_pre_grasp(self, bottle_pos: Point, z_offset=0.075): 
        print("bottle pos: ", bottle_pos)
        x_cup, y_cup, _ = bottle_pos.x, bottle_pos.y, bottle_pos.z 
        
        x_offset = (x_cup) * 0.60
        y_offset = (y_cup) * 0.6 + 0.03

        p = np.array([x_offset, y_offset, z_offset])

        rotation = np.matmul(rotX(np.deg2rad(-90)), rotY(np.deg2rad(-90)))
        R_standoff = np.matmul(rotZ(-(np.deg2rad(90) - np.atan2(x_cup, -y_cup))), rotation)[
            :3, :3
        ]

        print("R:", R_standoff, flush=True)
        q_wxyz = mat2quat(R_standoff)
        q_xyzw = np.array([q_wxyz[1], q_wxyz[2], q_wxyz[3], q_wxyz[0]])

        return (p, q_xyzw)


    def compute_place_beside_cup(self, bottle_pos: Point, z_offset=0.075): 
        print("bottle pos: ", bottle_pos)
        x_cup, y_cup, _ = bottle_pos.x, bottle_pos.y, bottle_pos.z 
        
        x_offset = (x_cup) * 0.20
        y_offset = (y_cup) * 0.28 - 0.2 

        p = np.array([x_offset, y_offset, z_offset])

        rotation = np.matmul(rotX(np.deg2rad(-90)), rotY(np.deg2rad(-90)))
        R_standoff = np.matmul(rotZ(-(np.deg2rad(90) - np.atan2(x_cup, -y_cup))), rotation)[
            :3, :3
        ]

        print("R:", R_standoff, flush=True)
        q_wxyz = mat2quat(R_standoff)
        q_xyzw = np.array([q_wxyz[1], q_wxyz[2], q_wxyz[3], q_wxyz[0]])

        return (p, q_xyzw)


    def compute_grasp(self, bottle_pos: Point, raise_up: bool = False): 
        x_cup, y_cup, _ = bottle_pos.x, bottle_pos.y, bottle_pos.z 

        x_offset = (x_cup) * 0.7
        y_offset = (y_cup) * 0.7 + 0.03

        p = np.array([x_offset, y_offset, .245 if raise_up else 0.075])

        rotation = np.matmul(rotX(np.deg2rad(-90)), rotY(np.deg2rad(-90)))
        R_standoff = np.matmul(rotZ(-(np.deg2rad(90) - np.atan2(x_cup, -y_cup))), rotation)[
            :3, :3
        ]

        q_wxyz = mat2quat(R_standoff)
        q_xyzw = np.array([q_wxyz[1], q_wxyz[2], q_wxyz[3], q_wxyz[0]])

        return (p, q_xyzw)

    def compute_ik(self, p, q_xyzw):
        retval_ik = self.moveit.moveit2.compute_ik(position=p, quat_xyzw=q_xyzw)
            
        if retval_ik is None:
            print("Inverse kinematics failed.")
            return 
        else:
            print("Inverse kinematics succeeded. Result: " + str(retval_ik))

            # extract the arm joints
            names = self.moveit.arm_joint_names
            joint_positions_ik = extract_specific_joints(retval_ik, names)

            print("joint_position_ik", joint_positions_ik)
            return joint_positions_ik


    def pick_container(self) -> bool:
        # pose = self.get_latest_pose()
        loc = self.get_latest_loc()
        # if pose is None:
        if loc is None:
            self.get_logger().error("No pose available")
            return False
        

        # First, to prevent the arm hitting the cup, we move the shoulder 
        # to point to the cup 
        p, q_xyzw = self.compute_pre_grasp(loc)
        joint_angles_rad = self.compute_ik(p, q_xyzw)
        shoulder_pan_deg = np.rad2deg(joint_angles_rad[0])
        self.moveit.move_joint_slowly("shoulder_pan.pos", shoulder_pan_deg)

        # Move before the bottle 
        self.moveit.open_gripper()
        self.moveit.move_robot_to(joint_angles_rad)

        # Grasp the bottle  - need to move slowly 
        p, q_xyzw = self.compute_grasp(loc)
        joint_angles_rad = self.compute_ik(p, q_xyzw)
        self.moveit.move_multiple_joints_slowly(joint_angles_rad, 0.8, 0.03)
        time.sleep(1)
        self.moveit.close_gripper() 

        # Now we have the cup, move it up slowly 
        p, q_xyzw = self.compute_grasp(loc, raise_up=True)
        joint_angles_rad = self.compute_ik(p, q_xyzw)
        self.moveit.move_multiple_joints_slowly(joint_angles_rad, 0.8, 0.03)

        # move cup out of frame
        out_of_frame = -20 
        self.moveit.move_joint_slowly("shoulder_pan.pos", out_of_frame )
        
        self.get_logger().info("Pick complete")
        return True


    def pour_into_cup(self) -> bool:
        # pose = self.get_latest_pose()
        loc = self.get_latest_loc()
        # if pose is None:
        if loc is None:
            self.get_logger().error("No pose available")
            return False
        

        # Move above the cup 
        p, q_xyzw = self.compute_above_cup(loc)
        joint_angles_rad = self.compute_ik(p, q_xyzw)
        self.moveit.move_multiple_joints_slowly(joint_angles_rad, 0.8, 0.03)


        input("pour?")
        # slowly pour into the cup 
        self.moveit.move_joint_slowly("wrist_roll.pos", -20, 0.8, 0.03)
        
        
        print("Pour complete.")
        time.sleep(0.5)
        self.moveit.move_joint_slowly("wrist_roll.pos", 65, 0.8, 0.03)

        # Put the bottle down beside the cup 
        p, q_xyzw = self.compute_place_beside_cup(loc)
        joint_angles_rad = self.compute_ik(p, q_xyzw)
        self.moveit.move_joint_slowly("shoulder_pan.pos", joint_angles_rad[0], 0.8, 0.03)
#        self.moveit.move_joint_slowly("wrist_roll.pos", joint_angles_rad[0], 0.8, 0.03)
        
        self.moveit.move_multiple_joints_slowly(joint_angles_rad, 0.8, 0.03)
        self.moveit.open_gripper()

        # move a little to the right 
        joints = self.moveit.get_robot_current_pos()
        self.moveit.move_joint_slowly("shoulder_pan.pos", (joints["shoulder_pan.pos"] + 0.06981317),  1, 0.03)
        
        # Move above the cup to avoid hitting the cup 
        p, q_xyzw = self.compute_place_beside_cup(loc, z_offset=0.275)
        joint_angles_rad = self.compute_ik(p, q_xyzw)
        joint_angles_rad[0] += + 0.06981317
        self.moveit.move_multiple_joints_slowly(joint_angles_rad, 1, 0.03)


        # move to home position 
        home = [0.0,0.0,1.413,-1.570796,1.570796]
        self.moveit.move_multiple_joints_slowly(home, 0.8, 0.03)

        return True
    
    def _pose_callback(self, msg: PoseStamped):
        with self.pose_lock:
            # print("new pose: ", msg)
            self.latest_pose = self.transform_handler.transform_pose(msg)
    
    def _loc_callback(self, loc: Point):
        print(f"got a new loc: {loc}")
        with self.loc_lock:
            self.latest_loc = self.transform_handler.transform_loc(loc)
    
    def get_latest_pose(self) -> Optional[PoseStamped]:
        with self.pose_lock:
            return self.latest_pose
    
    def get_latest_loc(self) -> Optional[Point]:
        with self.loc_lock:
            return self.latest_loc
    
    def go_home(self) -> bool:
        self.get_logger().info("Going home...")
        return self.moveit.go_to_named_target("home")
    
    def spin(self):
        rclpy.spin_once(self, timeout_sec=0.1)
        rclpy.spin_once(self.transform_handler, timeout_sec=0.1)
        rclpy.spin_once(self.moveit, timeout_sec=0.1)





def main():
    rclpy.init()

    try:
        

        bridge = PoseEstimatorMoveIt2Bridge(
            arm_group="arm",
            gripper_group="gripper"
        )

        home = [0.0,0.0,1.413,-1.570796,1.570796]
        bridge.moveit.move_robot_to(home)
        
        print("Waiting for container...")
        # # while rclpy.ok() and bridge.get_latest_pose() is None:
        while rclpy.ok() and bridge.get_latest_loc() is None:
            bridge.spin()
   
        
        if bridge.get_latest_loc():
            print("trying to pick up bottle", flush=True)
            bridge.pick_container()

        input("move pour into cup")
        if bridge.get_latest_loc():
            print("Trying to pour into cup", flush=True)
            bridge.pour_into_cup()

        print("Pour complete") 

    except KeyboardInterrupt:
        pass
    finally:
        rclpy.shutdown()


if __name__ == "__main__":
    main()