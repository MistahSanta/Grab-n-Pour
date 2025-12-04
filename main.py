#!/usr/bin/env python

import time, json, math
import rclpy


from lerobot.robots.so101_follower import SO101Follower, SO101FollowerConfig

from threading import Thread

from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup

from geometry_msgs.msg import Pose
from sensor_msgs.msg import JointState
from geometry_msgs.msg import TransformStamped
from tf2_ros import TransformBroadcaster

from pymoveit2 import MoveIt2
from pymoveit2 import GripperInterface


import numpy as np
from transforms3d.quaternions import mat2quat, quat2mat
from gazebo_func import get_current_joint_states, get_pose_gazebo, extract_specific_joints, ros_pose_to_rt, robot_world_pose
from lerobot_robot_ros.config import SO101ROSConfig
from dataclasses import field
from moveit_msgs.msg import Constraints, JointConstraint

# * This part of code is for main robot rather than simulation 
# robot_config = SO101FollowerConfig(port="/dev/ttyACM0", id="my_awesome_follower_arm", use_degrees=True, max_relative_target=200.0)
# robot = SO101Follower(robot_config)
# robot.connect()


# define the target pose 
# For now, assume bottle position is fixed 
# TODO get the actual position of the bottle from camera and offset it a little to get the target position 

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


def ros_quat(tf_quat): #wxyz -> xyzw
    quat = np.zeros(4)
    quat[-1] = tf_quat[0]
    quat[:-1] = tf_quat[1:]
    return quat

# broadcast a TF frame for debugging
class FrameBroadcaster(Node):
    def __init__(self, p=(0.5,0,0.2), q=(0,0,0,1)):
        super().__init__('frame_broadcaster')
        self.br = TransformBroadcaster(self)
        self.timer = self.create_timer(0.05, self.timer_cb)
        self.p = p
        self.q = q

    def timer_cb(self):
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = 'base_link'
        t.child_frame_id = 'my_frame'
        t.transform.translation.x, t.transform.translation.y, t.transform.translation.z = self.p
        t.transform.rotation.x, t.transform.rotation.y, t.transform.rotation.z, t.transform.rotation.w = self.q
        self.br.sendTransform(t)



def main():
    rclpy.init()
    
    node_moveit = Node("Grab_N_Pour")    
    
    
    # Create Robot interface 
    robot_config = SO101ROSConfig()

    robot_joint_names = robot_config.ros2_interface.arm_joint_names
    
    # Planner ID
    node_moveit.declare_parameter("planner_id", "RRTConnectkConfigDefault")
    
    # Create callback group that allows execution of callbacks in parallel without restrictions
    callback_group = ReentrantCallbackGroup()

    # Create MoveIt 2 interface
    moveit2 = MoveIt2(
        node=node_moveit,
        joint_names= robot_joint_names,
        base_link_name=robot_config.ros2_interface.base_link,
        end_effector_name="jaw",
        group_name="arm", 
        callback_group=callback_group,
    )    



    moveit2.planner_id = (
        node_moveit.get_parameter("planner_id").get_parameter_value().string_value
    )
    # Create gripper interface
    node_gripper = Node("fetch_gripper")
    gripper_interface = GripperInterface(
        node=node_gripper,
        gripper_joint_names=[robot_config.ros2_interface.gripper_joint_name],
        open_gripper_joint_positions=[robot_config.ros2_interface.gripper_open_position],
        closed_gripper_joint_positions=[robot_config.ros2_interface.gripper_close_position],
        gripper_group_name="gripper", 
        callback_group=callback_group,
        gripper_command_action_name="gripper_controller/gripper_cmd",
    )

    executor = rclpy.executors.MultiThreadedExecutor()
    executor.add_node(node_moveit)
    executor.add_node(node_gripper)
    #node_tf = FrameBroadcaster(p, q_xyzw)
    #executor.add_node(node_tf)
    executor_thread = Thread(target=executor.spin, daemon=True, args=())
    executor_thread.start()    
    
    # Scale down velocity and acceleration of joints (percentage of maximum)
    moveit2.max_velocity = 0.5
    moveit2.max_acceleration = 0.5

    # Sleep a while in order to get the first joint state
    node_moveit.create_rate(3.0).sleep()
    node_gripper.create_rate(3.0).sleep()        
    
    print("Testing gripping opening close")
    print()
    # test gripper open/close
    # gripper_interface.toggle()
    # gripper_interface.wait_until_executed()
    # gripper_interface.open()
    # gripper_interface.wait_until_executed()

    HOME_POSITION = [0.0,0.0,0.0,0.0,0.0]

    moveit2.move_to_configuration(HOME_POSITION)
    moveit2.wait_until_executed()

    # make sure end effector z-axis is parallel to the table 
    # 90 deg rotation around X-axis
    # roll = math.pi/2
    # pitch = 0
    # yaw = 0

    # qx = math.sin(roll/2)
    # qy = 0
    # qz = 0
    # qw = math.cos(roll/2)
    # oc = [qx,qy,qz,qw]
    # oc_tolerance = [1.05, 1.05, 3.14159]
    # moveit2.set_path_orientation_constraint(
    #     quat_xyzw=oc,
    #     tolerance=oc_tolerance,
    #     parameterization=1,
    # )



    # query the pose of the cube
    while 1:
        T_bo = get_pose_gazebo(node_moveit)
        if T_bo is not None:
            break
    print('T_bo', T_bo)


    while 1:
        T_robot = robot_world_pose(node_moveit)
        if T_robot is not None:
            break
    print('T_robot', T_robot)


    # Since the SO101 arm is limited in mobility with MoveIt, we will
    # First go to all 0 position, so we can move step by step

    
    # get the current robot joints
    joint_positions = get_current_joint_states(node_moveit)
    print("JOINT POsition", joint_positions)

    # Grab current position via FK for debugging if needed 
    FK = moveit2.compute_fk(
        joint_state=joint_positions,
        fk_link_names=["wrist"]
    )[0]
    if FK is None: 
        exit()
    print("FK: ", FK)
    print()

    robot_current_rt = ros_pose_to_rt(FK.pose)
    robot_current_position = robot_current_rt[:3, 3]# + np.array([0.1, 0.1, 0.1 ])
    print("cur_position: ", robot_current_position)
    # # compute standoff position of the rect_cup 

    x_robot, y_robot, z_robot = T_robot[0, 3], T_robot[1, 3], T_robot[2, 3]
    
    x_cup, y_cup, z_cup = T_bo[0,3], T_bo[1,3], T_bo[2,3]
    #standoff position 
    # p = np.array([x_cup, y_cup, z_cup + 0.1])
    # ratio = math.fabs(x_cup / y_cup)

    # x_off = math.fabs(x_cup / y_cup) * 0.1
    # if x_cup > 0:
    #     x_off = -x_off
    
    # y_off = ratio * 0.1

    # x_offset = (x_cup) - ratio * 0.1
    # y_offset = (y_cup) - ratio * 0.1



    x_offset = (x_cup) * 0.600
    y_offset = (y_cup) * 0.600

    p = np.array([x_offset, y_offset, z_cup + 0.075])
    # p = np.array([x_cup, y_cup + 0.21648056, z_cup + 0.2])

    #print( "P: ", p, flush=True)

    R_standoff = np.array(
        [[1,0,0],
        [0,1,0],
        [0,0,1]]
    ) # no rotation

    rotation = np.matmul(rotX(np.deg2rad(90)), rotY(np.deg2rad(-90)))
    R_standoff = np.matmul(rotZ(-(np.deg2rad(90) - np.atan2(x_cup, -y_cup))), rotation)[:3,:3]
    print("R:", R_standoff, flush=True)
    q_wxyz = mat2quat(R_standoff)
    q_xyzw = np.array([q_wxyz[1], q_wxyz[2], q_wxyz[3], q_wxyz[0]])


    # instead of using compute_ik use move_to_pose
    #moveit2.move_to_pose( position=p, quat_xyzw=q_xyzw_current, cartesian=False) 



    # To simplify the inverse_ik, we will add constraints that lock Joint 4 and 5
    # constaints = Constraints()

    # lock_j5 = JointConstraint()
    # lock_j5.joint_name = "5"
    # lock_j5.position = 1
    # lock_j5.tolerance_above = 0.2
    # lock_j5.tolerance_below = 0.2
    # lock_j5.weight = 1.0

    # constaints.joint_constraints.append(lock_j5)
    
    # print("CURRENT OIIR", q_xyzw_current)
    # moveit2.set_pose_goal(position=p, quat_xyzw=q_xyzw, target_link="gripper")

    # plan = moveit2.plan()

    # if plan:
    #     print("Found a valid plan")
    #     moveit2.execute(plan)
    # else:
    #     print("no plan found!")

    retval_ik = moveit2.compute_ik(position=p, quat_xyzw=q_xyzw) 

    if retval_ik is None:
        print("Inverse kinematics failed.")
    else:
        print("Inverse kinematics succeeded. Result: " + str(retval_ik))

        # extract the arm joints
        print("---------------------------")
        names = robot_joint_names
        joint_positions_ik = extract_specific_joints(retval_ik, names)
        print("---------------------------")

        # compute the distances between joints
        distance = np.linalg.norm(joint_positions - joint_positions_ik)
        print('joint distance:', distance)

        # use moveit2 to move to the joint position from IK
        input('move?')

        moveit2.move_to_configuration(joint_positions_ik)
        moveit2.wait_until_executed()


    rclpy.shutdown()
    executor.shutdown()
    executor_thread.join()
    



if __name__ == "__main__":
    main()