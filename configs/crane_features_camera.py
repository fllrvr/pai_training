# Camera version of crane_features.py: joint data + one camera image

# Image size stored in the dataset (height, width). Images are resized to this when converting.
# Set to None to keep the original size (640x480 from Gazebo) and update the shape below to match.
image_size = (240, 320)

features = {
    "observation.state": {
        "dtype": "float32",
        "shape": (5,),
        "names": ["joint1", "joint2", "joint3", "joint4", "gripper"],
    },
    "observation.environment_state": {
        "dtype": "float32",
        "shape": (5,),
        "names": ["joint1", "joint2", "joint3", "joint4", "gripper"],
    },
    "observation.images.camera": {
        "dtype": "video",
        "shape": (image_size[0], image_size[1], 3),
        "names": ["height", "width", "channels"],
    },
    "action": {
        "dtype": "float32",
        "shape": (5,),
        "names": ["joint1", "joint2", "joint3", "joint4", "gripper"],
    },
}
# Below is used for the dataset conversion
robot_type = "crane_plus"
repo_id = "crane_plus_pick_and_move_camera"
root = "/workspace/dataset/lerobot_dataset/crane_plus_pick_and_move_camera"
task_name = "pick_and_move"
# Camera topic in the rosbag and the feature key it is stored under
camera_topic = "/image_raw"
image_key = "observation.images.camera"
