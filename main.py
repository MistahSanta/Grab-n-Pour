import time, json 
from lerobot.robots.so101_follower import SO101Follower, SO101FollowerConfig


robot_config = SO101FollowerConfig(port="/dev/ttyACM0", id="my_awesome_follower_arm", use_degrees=True, max_relative_target=200.0)
robot = SO101Follower(robot_config)
robot.connect()


# define the target pose 
# For now, assume bottle position is fixed 
# TODO get the actual position of the bottle from camera and offset it a little to get the target position 
target_pose = 


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