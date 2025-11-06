import time, json 
from lerobot.robots.so101_follower import SO101Follower, SO101FollowerConfig

robot_config = SO101FollowerConfig(port="/dev/ttyACM0", id="my_awesome_follower_arm", use_degrees=True, max_relative_target=200.0)
robot = SO101Follower(robot_config)
robot.connect()

position_arr = [
    {
        'shoulder_pan.pos': 0.0,
        'shoulder_lift.pos': -30.0,
        'elbow_flex.pos': 30.0,
        'wrist_flex.pos': 70.0,
        'wrist_roll.pos': 0.0,
        'gripper.pos': 0.0
    },
    {
        'shoulder_pan.pos': 45.0,
        'shoulder_lift.pos': -20.0,
        'elbow_flex.pos': 30.0,
        'wrist_flex.pos': 60.0,
        'wrist_roll.pos': -30.0,
        'gripper.pos': 50.0
    },
    {
        'shoulder_pan.pos': -45.0,
        'shoulder_lift.pos': -0.0,
        'elbow_flex.pos': 0.0,
        'wrist_flex.pos': -30.0,
        'wrist_roll.pos': 30.0,
        'gripper.pos': 30.0
    }
]


def check_dict(actual_pos: dict, target_pos: dict ) -> bool: 
    # Assume actual_pos and target_pos have the same key for simplicity 
    for key in actual_pos:
        value1 = int(actual_pos[key])
        value2 = int(target_pos[key])

        difference = abs(value1 - value2)

        if difference > 1: 
            return False 
    
    return True

for index, position in enumerate(position_arr):
    start_time = time.time()
    # move to position  
    robot.send_action(position)

    # Reading joint posiitons
    joint_pos = robot.get_observation()
    

    while not check_dict(joint_pos, position):
        joint_pos = robot.get_observation() # update actual position 
    
    print(f"Time take to get to position {index + 1} = {time.time() - start_time} seconds")    

    print(f"pos{index + 1}: ", json.dumps(joint_pos, indent=4)) # pretty print json 

    print("-" * 100)

    time.sleep(3) # hold position for 3 seconds