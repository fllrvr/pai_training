#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
import torch
import argparse
import numpy as np

from lerobot.common.policies.act.modeling_act import ACTPolicy
# Same image preprocessing as the dataset conversion
from rosbag2lerobot import decode_image

ARM_JOINT_NAMES = ['crane_plus_joint1', 'crane_plus_joint2', 'crane_plus_joint4', 'crane_plus_joint3']
GRIPPER_JOINT_NAME = 'crane_plus_joint_hand'

class InferenceNode(Node):
    def __init__(self, model_path, camera_topic):
        super().__init__('crane_plus_act_infer')
        self.arm_pub = self.create_publisher(JointTrajectory, '/crane_plus_arm_controller/joint_trajectory', 10)
        self.gripper_pub = self.create_publisher(JointTrajectory, '/crane_plus_gripper_controller/joint_trajectory', 10)
        self.subscription = self.create_subscription(JointState, '/joint_states', self.joint_state_callback, 10)
        self.policy = ACTPolicy.from_pretrained(model_path)
        self.policy.eval()
        self.device = torch.device('cpu')
        self.policy.to(self.device)
        self.last_action = np.zeros(5, dtype=np.float32)
        self.timer = self.create_timer(0.05, self.publish_action)  # 20Hz
        self.latest_joint_state = None

        # Use the camera only if the model was trained with images.
        # The image key and size are taken from the model, so they always match the training data.
        self.image_key = None
        self.latest_image = None
        image_features = self.policy.config.image_features
        if len(image_features) > 1:
            raise ValueError(f"Only one camera is supported, but the model uses {list(image_features)}")
        if image_features:
            self.image_key, feature = next(iter(image_features.items()))
            self.image_size = tuple(feature.shape[1:])  # (C, H, W) -> (H, W)
            self.image_sub = self.create_subscription(Image, camera_topic, self.image_callback, 1)
            print(f"Camera: {camera_topic} -> {self.image_key}, image_size={self.image_size}")

    def joint_state_callback(self, msg):
        # Extract joint positions in the order of ARM_JOINT_NAMES + [GRIPPER_JOINT_NAME]
        name_to_pos = dict(zip(msg.name, msg.position))
        arm_pos = [name_to_pos.get(j, 0.0) for j in ARM_JOINT_NAMES]
        gripper_pos = name_to_pos.get(GRIPPER_JOINT_NAME, 0.0)
        self.latest_joint_state = np.array(arm_pos + [gripper_pos], dtype=np.float32)

    def image_callback(self, msg):
        self.latest_image = decode_image(msg, self.image_size)  # (H, W, 3) uint8 RGB

    def publish_action(self):
        if self.latest_joint_state is None:
            return
        if self.image_key and self.latest_image is None:
            return
        # Prepare observation dict for ACTPolicy
        obs = {
            'observation.state': torch.tensor(self.latest_joint_state, dtype=torch.float32).unsqueeze(0),  # (1, 5)
            'observation.environment_state': torch.tensor(self.latest_joint_state, dtype=torch.float32).unsqueeze(0),  # (1, 5)
        }
        if self.image_key:
            # Same format as the dataset: float32 in [0, 1], channel first
            img = torch.from_numpy(self.latest_image).permute(2, 0, 1).float() / 255.0
            obs[self.image_key] = img.unsqueeze(0)  # (1, 3, H, W)
        with torch.no_grad():
            action = self.policy.select_action(obs).cpu().numpy().flatten()  # (5,)
        self.last_action = action
        # Publish to arm and gripper
        jt = JointTrajectory()
        jt.joint_names = ARM_JOINT_NAMES
        pt = JointTrajectoryPoint()
        pt.positions = action[:4].tolist()
        pt.time_from_start.sec = 0
        pt.time_from_start.nanosec = int(0.1 * 1e9)
        jt.points.append(pt)
        self.arm_pub.publish(jt)
        jt_g = JointTrajectory()
        jt_g.joint_names = [GRIPPER_JOINT_NAME]
        pt_g = JointTrajectoryPoint()
        pt_g.positions = [float(action[4])]
        pt_g.time_from_start.sec = 0
        pt_g.time_from_start.nanosec = int(0.1 * 1e9)
        jt_g.points.append(pt_g)
        self.gripper_pub.publish(jt_g)
        print("action:", action[:4], "gripper:", action[4])

def main():
    parser = argparse.ArgumentParser(description='ACTPolicy inference node for CRANE+ V2 (ROS2)')
    parser.add_argument('--model-path', type=str, required=True, help='Path to trained ACTPolicy model directory')
    parser.add_argument('--camera-topic', type=str, default='/image_raw',
                        help='Camera topic (used only if the model was trained with images)')
    args = parser.parse_args()
    rclpy.init()
    node = InferenceNode(args.model_path, args.camera_topic)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
