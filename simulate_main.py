# Move the robot using Joints space 

import time, json, math
from lerobot_robot_ros.config import SO101ROSConfig
from lerobot_robot_ros.robot import SO101ROS

robot_config = SO101ROSConfig()
robot = SO101ROS(robot_config)
robot.connect()

position_arr = [
    {
        '1.pos': math.radians(0.0),
        '2.pos': math.radians(-30.0),
        '3.pos': math.radians(30.0),
        '4.pos': math.radians(70.0),
        '5.pos': math.radians(0.0),
        'gripper.pos': 1.0,
    },
    {
        '1.pos': math.radians(45.0),
        '2.pos': math.radians(-20.0),
        '3.pos': math.radians(30.0),
        '4.pos': math.radians(60.0),
        '5.pos': math.radians(-30.0),
        #'gripper.pos': math.radians(50.0),
        'gripper.pos': 0.5,
    },
    {
        '1.pos': math.radians(-45.0),
        '2.pos': math.radians(-10.0),
        '3.pos': math.radians(0.0),
        '4.pos': math.radians(-30.0),
        '5.pos': math.radians(30.0),
        #'gripper.pos': math.radians(30.0),
        'gripper.pos': 0.7,
    },
]
# position_arr = [
#     {
#         'shoulder_pan.pos': 0.0,
#         'shoulder_lift.pos': -30.0,
#         'elbow_flex.pos': 30.0,
#         'wrist_flex.pos': 70.0,
#         'wrist_roll.pos': 0.0,
#         'gripper.pos': 0.0
#     },
#     {
#         'shoulder_pan.pos': 45.0,
#         'shoulder_lift.pos': -20.0,
#         'elbow_flex.pos': 30.0,
#         'wrist_flex.pos': 60.0,
#         'wrist_roll.pos': -30.0,
#         'gripper.pos': 50.0
#     },
#     {
#         'shoulder_pan.pos': -45.0,
#         'shoulder_lift.pos': -0.0,
#         'elbow_flex.pos': 0.0,
#         'wrist_flex.pos': -30.0,
#         'wrist_roll.pos': 30.0,
#         'gripper.pos': 30.0
#     }
# ]


def check_dict(actual_pos: dict, target_pos: dict ) -> bool: 
    # Assume actual_pos and target_pos have the same key for simplicity 
    for key in actual_pos:
        value1 = actual_pos[key]
        if key == "6.pos":
            value2 = target_pos["gripper.pos"]
            difference = abs(value1 - math.radians(100.0 - value2 * 100.0))
        else:
            value2 = target_pos[key]
            difference = abs(value1 - value2)


        print("actual_pos = " + str(actual_pos))
        print("diff for joint " + key + " is " + str(difference))

        if difference > math.radians(1.0): 
            return False 
    
    return True

# robot.send_action({
#     '1.pos': math.radians(0.0),
#     '2.pos': math.radians(0.0),
#     '3.pos': math.radians(0.0),
#     '4.pos': math.radians(0.0),
#     '5.pos': math.radians(45.0),
#     'gripper.pos': 0.0
# })


# sys.exit()


for index, position in enumerate(position_arr):
    start_time = time.time()
    # move to position  
    robot.send_action(position)

    # Reading joint posiitons
    joint_pos = robot.get_observation()

    print(joint_pos)
    

    while not check_dict(joint_pos, position):
        joint_pos = robot.get_observation() # update actual position 
    
    print(f"Time take to get to position {index + 1} = {time.time() - start_time} seconds")    

    print(f"pos{index + 1}: ", json.dumps(joint_pos, indent=4)) # pretty print json 

    print("-" * 100)

    time.sleep(3) # hold position for 3 seconds
