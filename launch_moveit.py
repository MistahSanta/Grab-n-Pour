    


def generate_launch_description():

    # create a runtime lauch argument
    is_sim_arg = DeclareLaunchArgument(name="is_sim", default_value="True")

    # get the argument value at runtime
    is_sim = LaunchConfiguration("is_sim")

    # URDF
    lerobot_description_dir = get_package_share_directory("lerobot_description")
    so101_urdf_path = os.path.join(lerobot_description_dir, "urdf", "so101.urdf.xacro")

    

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