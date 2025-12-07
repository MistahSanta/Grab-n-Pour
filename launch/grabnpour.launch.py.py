import os
from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import Command, LaunchConfiguration

from launch_ros.actions import Node
from moveit_configs_utils import MoveItConfigsBuilder


def generate_launch_description():

    

    moveit_config = (
            MoveItConfigsBuilder("so101", package_name="lerobot_moveit")
            .robot_description(file_path=so101_urdf_path)
            .robot_description_semantic(file_path="config/so101.srdf")
            .trajectory_execution(file_path="config/moveit_controllers.yaml")
            .to_moveit_configs()
            )


    example_file = DeclareLaunchArgument(
        "example_file",
        default_value="motion_planning_python_api_tutorial.py",
        description="Python API tutorial file name",
    )

    moveit_py_node = Node(
        name="moveit_py",
        package="moveit2_tutorials",
        executable=LaunchConfiguration("example_file"),
        output="both",
        parameters=[moveit_config.to_dict()],
    )



    return LaunchDescription([
        is_sim_arg,
        move_group_node,
        rviz_node,
        moveit_py_node
    ])
