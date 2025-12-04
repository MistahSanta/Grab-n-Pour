#!/usr/bin/env python

import time
import json
import math
import rclpy
from rclpy.task import Future

from typing import List, Tuple

from lerobot.robots.so101_follower import SO101Follower, SO101FollowerConfig

from threading import Thread

from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup

from geometry_msgs.msg import Pose, PoseStamped, TransformStamped
from sensor_msgs.msg import JointState
from tf2_ros import TransformBroadcaster

from pymoveit2 import MoveIt2
from pymoveit2 import GripperInterface


import numpy as np
from transforms3d.quaternions import mat2quat, quat2mat
from lerobot_robot_ros.config import SO101ROSConfig
from dataclasses import field
from moveit_msgs.msg import Constraints, JointConstraint


# Convert quaternion and translation to a 4x4 tranformation matrix
# See Appendix B.3 in Lynch and Park, Modern Robotics for the definition of quaternion
def ros_qt_to_rt(rot, trans):
    qt = np.zeros((4,), dtype=np.float32)
    qt[0] = rot[3]
    qt[1] = rot[0]
    qt[2] = rot[1]
    qt[3] = rot[2]
    obj_T = np.eye(4)
    obj_T[:3, :3] = quat2mat(qt)
    obj_T[:3, 3] = trans

    return obj_T


# Convert a ROS pose message to a 4x4 tranformation matrix
def ros_pose_to_rt(pose):
    qarray = [0, 0, 0, 0]
    qarray[0] = pose.orientation.x
    qarray[1] = pose.orientation.y
    qarray[2] = pose.orientation.z
    qarray[3] = pose.orientation.w

    t = [0, 0, 0]
    t[0] = pose.position.x
    t[1] = pose.position.y
    t[2] = pose.position.z

    return ros_qt_to_rt(qarray, t)


def get_message_once(node, topic_name, msg_type, timeout=5.0):
    future = Future()

    def callback(msg):
        if not future.done():
            future.set_result(msg)
            node.destroy_subscription(sub)

    sub = node.create_subscription(msg_type, topic_name, callback, 10)

    # Spin this node until a message is received or timeout
    rclpy.spin_until_future_complete(node, future, timeout_sec=timeout)

    return future.result() if future.done() else None


MODEL = "rect_cup"
ROBOT = "so101"


# Query pose of frames from the Gazebo environment
def get_pose_gazebo(node):

    pose_model = get_message_once(
        node, f"/model/{MODEL}/pose", PoseStamped, timeout=2.0
    ).pose

    # convert the cube pose in world frame T_wo
    T_wo = ros_pose_to_rt(pose_model)
    print("T_wo", T_wo)

    pose_robot = get_message_once(
        node, f"/model/{ROBOT}/pose", PoseStamped, timeout=2.0
    ).pose
    # convert the robot pose in world frame T_wb
    T_wb = ros_pose_to_rt(pose_robot)
    print("T_wb", T_wb)

    ################ TO DO: query cube pose ##########################
    # compute the object pose in robot base link T_bo: 4x4 transformation matrix
    T_bw = np.linalg.inv(T_wb)
    T_bo = np.matmul(T_bw, T_wo)

    ################ TO DO: query cube pose ##########################
    return T_bo


def robot_world_pose(node):
    pose_robot = get_message_once(
        node, f"/model/{ROBOT}/pose", PoseStamped, timeout=2.0
    ).pose
    T_wb = ros_pose_to_rt(pose_robot)
    return T_wb


def get_current_joint_states(node_arm) -> Tuple[List[float], float]:
    """Returns the current joint configuration including the gripper."""

    robot = SO101ROSConfig().ros2_interface
    while 1:
        msg = get_message_once(node_arm, "/joint_states",
                               JointState, timeout=2.0)
        print("get_current_joint_states: " + str(msg))

        # get the joint names in the Fetch robot group
        names = robot.arm_joint_names
        gripper_name = robot.gripper_joint_name

        joint_positions = extract_specific_joints(msg, [*names, gripper_name])
        if joint_positions is not None:
            break
    return (joint_positions[:5], joint_positions[5])


# extract the joint positions from the msg according to the joint names
def extract_specific_joints(msg, names):
    joint_positions = np.zeros((len(names),), np.float64)
    for i, name in enumerate(names):
        joint_positions[i] = msg.position[msg.name.index(name)]
        print(name, joint_positions[i])
    return joint_positions


# rotation matrix around x-axis
def rotX(rotx):
    RotX = np.array(
        [
            [1, 0, 0, 0],
            [0, np.cos(rotx), -np.sin(rotx), 0],
            [0, np.sin(rotx), np.cos(rotx), 0],
            [0, 0, 0, 1],
        ]
    )
    return RotX


# rotation matrix around y-axis
def rotY(roty):
    RotY = np.array(
        [
            [np.cos(roty), 0, np.sin(roty), 0],
            [0, 1, 0, 0],
            [-np.sin(roty), 0, np.cos(roty), 0],
            [0, 0, 0, 1],
        ]
    )
    return RotY


# rotation matrix around z-axis
def rotZ(rotz):
    RotZ = np.array(
        [
            [np.cos(rotz), -np.sin(rotz), 0, 0],
            [np.sin(rotz), np.cos(rotz), 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ]
    )
    return RotZ


def ros_quat(tf_quat):  # wxyz -> xyzw
    quat = np.zeros(4)
    quat[-1] = tf_quat[0]
    quat[:-1] = tf_quat[1:]
    return quat


# broadcast a TF frame for debugging
class FrameBroadcaster(Node):
    def __init__(self, p=(0.5, 0, 0.2), q=(0, 0, 0, 1)):
        super().__init__("frame_broadcaster")
        self.br = TransformBroadcaster(self)
        self.timer = self.create_timer(0.05, self.timer_cb)
        self.p = p
        self.q = q

    def timer_cb(self):
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = "base_link"
        t.child_frame_id = "my_frame"
        (
            t.transform.translation.x,
            t.transform.translation.y,
            t.transform.translation.z,
        ) = self.p
        (
            t.transform.rotation.x,
            t.transform.rotation.y,
            t.transform.rotation.z,
            t.transform.rotation.w,
        ) = self.q
        self.br.sendTransform(t)


def move_robot_to(
    irl_robot,
    moveit2,
    gripper_interface,
    node_arm,
    arm_configuration: List[float],
    gripper_open: bool,
):
    """Moves the SO101 robot to the provided joint angles, in radians.

    `arm_configuration` should be a list of 5 floats.
    """

    moveit2.move_to_configuration(arm_configuration)
    moveit2.wait_until_executed()

    if gripper_open:
        gripper_interface.open()
    else:
        gripper_interface.close()

    gripper_interface.wait_until_executed()

    time.sleep(3)

    # get the current robot joints
    joint_positions = get_current_joint_states(node_arm)
    print("Joint position", joint_positions)

    ((j1, j2, j3, j4, j5), j6) = joint_positions


    action = {
        "shoulder_pan.pos": np.rad2deg(j1),
        "shoulder_lift.pos": np.rad2deg(j2),
        "elbow_flex.pos": np.rad2deg(j3),
        "wrist_flex.pos": np.rad2deg(j4),
        "wrist_roll.pos": np.rad2deg(j5),
        "gripper.pos": np.rad2deg(j6),
    }

    irl_robot.send_action(action)

    time.sleep(5.0)


def main():
    rclpy.init()

    node_arm = Node("Grab_N_Pour")

    irl_robot_config = SO101FollowerConfig(
        port="/dev/ttyACM0",
        id="my_awesome_follower_arm",
        use_degrees=True,
        max_relative_target=200.0,
    )
    irl_robot = SO101Follower(irl_robot_config)
    irl_robot.connect()

    robot_config = SO101ROSConfig()

    arm_joint_names = robot_config.ros2_interface.arm_joint_names

    # Planner ID
    node_arm.declare_parameter("planner_id", "RRTConnectkConfigDefault")

    # Create callback group that allows execution of callbacks in parallel without restrictions
    callback_group = ReentrantCallbackGroup()

    # Create MoveIt 2 interface
    moveit2 = MoveIt2(
        node=node_arm,
        joint_names=arm_joint_names,
        base_link_name=robot_config.ros2_interface.base_link,
        end_effector_name="jaw",
        group_name="arm",
        callback_group=callback_group,
    )

    moveit2.planner_id = (
        node_arm.get_parameter("planner_id").get_parameter_value().string_value
    )
    # Create gripper interface
    node_gripper = Node("fetch_gripper")
    gripper_interface = GripperInterface(
        node=node_gripper,
        gripper_joint_names=[robot_config.ros2_interface.gripper_joint_name],
        open_gripper_joint_positions=[
            robot_config.ros2_interface.gripper_open_position
        ],
        closed_gripper_joint_positions=[
            robot_config.ros2_interface.gripper_close_position
        ],
        gripper_group_name="gripper",
        callback_group=callback_group,
        gripper_command_action_name="gripper_controller/gripper_cmd",
    )

    executor = rclpy.executors.MultiThreadedExecutor()
    executor.add_node(node_arm)
    executor.add_node(node_gripper)
    # node_tf = FrameBroadcaster(p, q_xyzw)
    # executor.add_node(node_tf)
    executor_thread = Thread(target=executor.spin, daemon=True, args=())
    executor_thread.start()

    # Scale down velocity and acceleration of joints (percentage of maximum)
    moveit2.max_velocity = 0.5
    moveit2.max_acceleration = 0.5

    # Sleep a while in order to get the first joint state
    node_arm.create_rate(3.0).sleep()
    node_gripper.create_rate(3.0).sleep()

    home = [0.0, 0.0, 0.0, 0.0, 0.0]

    move_robot_to(irl_robot, moveit2, gripper_interface, node_arm, home, False)

    # query the pose of the cube
    while 1:
        T_bo = get_pose_gazebo(node_arm)
        if T_bo is not None:
            break
    print("T_bo", T_bo)

    while 1:
        T_robot = robot_world_pose(node_arm)
        if T_robot is not None:
            break
    print("T_robot", T_robot)

    x_robot, y_robot, z_robot = T_robot[0, 3], T_robot[1, 3], T_robot[2, 3]

    x_cup, y_cup, z_cup = T_bo[0, 3], T_bo[1, 3], T_bo[2, 3]

    x_offset = (x_cup) * 0.600
    y_offset = (y_cup) * 0.600

    p = np.array([x_offset, y_offset, z_cup + 0.075])

    R_standoff = np.eye(3)

    rotation = np.matmul(rotX(np.deg2rad(90)), rotY(np.deg2rad(-90)))
    R_standoff = np.matmul(rotZ(-(np.deg2rad(90) - np.atan2(x_cup, -y_cup))), rotation)[
        :3, :3
    ]

    print("R:", R_standoff, flush=True)
    q_wxyz = mat2quat(R_standoff)
    q_xyzw = np.array([q_wxyz[1], q_wxyz[2], q_wxyz[3], q_wxyz[0]])

    retval_ik = moveit2.compute_ik(position=p, quat_xyzw=q_xyzw)

    if retval_ik is None:
        print("Inverse kinematics failed.")
    else:
        print("Inverse kinematics succeeded. Result: " + str(retval_ik))

        # extract the arm joints
        print("---------------------------")
        names = arm_joint_names
        joint_positions_ik = extract_specific_joints(retval_ik, names)
        print("---------------------------")

        # compute the distances between joints
        # distance = np.linalg.norm(joint_positions - joint_positions_ik)
        # print("joint distance:", distance)

        # use moveit2 to move to the joint position from IK
        input("Press ENTER to move...")

        move_robot_to(
            irl_robot, moveit2, gripper_interface, node_arm, joint_positions_ik, True
        )

    rclpy.shutdown()
    executor.shutdown()
    executor_thread.join()


if __name__ == "__main__":
    main()