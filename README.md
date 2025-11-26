
## Build lerobot-ros
- conda create -y -n lerobot-ros python=3.12
- conda activate lerobot-ros # Activate the virtual env
- conda install -c conda-forge libstdcxx-ng -y # needed as rclpy requires GLIBCXX_3.4.30 symbols
- source /opt/ros/jazzy/setup.sh
- cd src/lerobot-ros
- pip install -e lerobot_robot_ros lerobot_teleoperator_devices (if this doesn't work try: pip install -e lerobot_robot_ros -e lerobot_teleoperator_devices
)


## Build lerobot_ws
- source /opt/ros/jazzy/setup.bash
- cd src/lerobot_ws 
- rosdep update 
- rosdep install --from-paths src --ignore-src -r -y
- colcon build


# Run command (Need atleast 2 terminals)

1) In terminal 1, launch the simulation: 
    - ros2 launch lerobot_description so101_gazebo.launch.py
2) In terminal 2, run the robot controller:
    - ros2 launch lerobot_controller so101_controller.launch.py && ros2 launch lerobot_moveit so101_moveit.launch.py


# To test it works:

    - python3 simulasimulate_main.py (You should see the SO101 arm move around)