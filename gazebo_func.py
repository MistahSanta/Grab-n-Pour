import rclpy
import numpy as np


from rclpy.task import Future
from geometry_msgs.msg import Pose, PoseStamped
from sensor_msgs.msg import JointState

from transforms3d.quaternions import mat2quat, quat2mat
from lerobot_robot_ros.config import SO101ROSConfig


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

MODEL = 'rect_cup'
ROBOT = 'so101'

# Query pose of frames from the Gazebo environment
def get_pose_gazebo(node):

    pose_model = get_message_once(node, f'/model/{MODEL}/pose', PoseStamped, timeout=2.0).pose
    
    # convert the cube pose in world frame T_wo
    T_wo = ros_pose_to_rt(pose_model)
    print('T_wo', T_wo)

    pose_robot = get_message_once(node, f'/model/{ROBOT}/pose', PoseStamped, timeout=2.0).pose
    # convert the robot pose in world frame T_wb
    T_wb = ros_pose_to_rt(pose_robot)
    print('T_wb', T_wb)
    
    ################ TO DO: query cube pose ##########################
    # compute the object pose in robot base link T_bo: 4x4 transformation matrix
    T_bw = np.linalg.inv(T_wb)
    T_bo = np.matmul(T_bw, T_wo)

    ################ TO DO: query cube pose ##########################
    return T_bo


# query the joint states of the robot
def get_current_joint_states(node):
    """Return one JointState message from /joint_states, or None if timeout."""
    robot = SO101ROSConfig().ros2_interface
    while 1:
        msg = get_message_once(node, '/joint_states', JointState, timeout=2.0)
        print(msg)

        # get the joint names in the Fetch robot group
        names = robot.arm_joint_names

        joint_positions = extract_specific_joints(msg, names)
        if joint_positions is not None:
            break
    return joint_positions


# extract the joint positions from the msg according to the joint names
def extract_specific_joints(msg, names):
    joint_positions = np.zeros((len(names), ), np.float64)
    for (i, name) in enumerate(names):
        joint_positions[i] = msg.position[msg.name.index(name)]
        print(name, joint_positions[i])
    return joint_positions
