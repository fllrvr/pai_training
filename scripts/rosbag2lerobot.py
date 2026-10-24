#!/usr/bin/env python3

import cv2
import numpy as np
import torch
from pathlib import Path
from rosbags.rosbag2 import Reader
from rosbags.serde import deserialize_cdr
import yaml
import importlib.util

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

def decode_image(msg, image_size=None):
    """Convert a sensor_msgs/Image to an RGB uint8 array (H, W, 3), resized to image_size (height, width) if set.

    Also used by infer.py so that inference applies exactly the same preprocessing as the dataset.
    """
    channels = {'rgb8': 3, 'bgr8': 3, 'rgba8': 4, 'bgra8': 4, 'mono8': 1}
    if msg.encoding not in channels:
        raise ValueError(f"Unsupported image encoding: {msg.encoding}")
    c = channels[msg.encoding]
    img = np.asarray(msg.data, dtype=np.uint8).reshape(msg.height, msg.step)[:, :msg.width * c]
    img = img.reshape(msg.height, msg.width, c)
    if msg.encoding == 'bgr8':
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    elif msg.encoding == 'rgba8':
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
    elif msg.encoding == 'bgra8':
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
    elif msg.encoding == 'mono8':
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    if image_size and img.shape[:2] != tuple(image_size):
        h, w = image_size
        img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
    return img

class RosbagToLeRobot:
    def __init__(self, bag_dir, output_dir, config_path, target_freq=None):
        self.bag_path = Path(bag_dir).resolve()
        self.output_dir = Path(output_dir).resolve()
        self.target_freq = target_freq

        # Find all episode directories (containing .db3 and metadata.yaml)
        self.episode_dirs = sorted(p.parent for p in self.bag_path.rglob('*.db3'))
        if not self.episode_dirs:
            raise FileNotFoundError(f"No rosbag2 episode directories found in {bag_dir}")

        print(f"Found {len(self.episode_dirs)} episode directories")
        for ep_dir in self.episode_dirs:
            print(f"  - {ep_dir}")

        if config_path.endswith('.py'):
            config = load_config_py(config_path)
            features = config.features
            robot_type = getattr(config, "robot_type", "crane_plus")
            repo_id = getattr(config, "repo_id", "teleop_rosbag_dataset")
            task_name = getattr(config, "task_name", "teleop")
            camera_topic = getattr(config, "camera_topic", None)
            image_key = getattr(config, "image_key", "observation.images.camera")
            image_size = getattr(config, "image_size", None)
        else:
            # YAML fallback
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            features = config['features']
            robot_type = config.get('robot_type', 'crane_plus')
            repo_id = config.get('repo_id', 'teleop_rosbag_dataset')
            task_name = config.get('task_name', 'teleop')
            camera_topic = config.get('camera_topic')
            image_key = config.get('image_key', 'observation.images.camera')
            image_size = config.get('image_size')

        self.task_name = task_name

        # Camera is optional: without camera_topic in the config, only joint data is converted
        self.camera_topic = camera_topic
        self.image_key = image_key
        self.image_size = tuple(image_size) if image_size else None  # (height, width); None keeps the original size
        if self.camera_topic:
            if self.image_key not in features:
                raise ValueError(f"'{self.image_key}' must be defined in features when camera_topic is set")
            print(f"Camera: {self.camera_topic} -> {self.image_key}, image_size={self.image_size or 'original'}")

        # --- Dataset creation with user prompt if dir exists ---
        while True:
            try:
                self.dataset = LeRobotDataset.create(
                    repo_id=repo_id,
                    fps=target_freq or 20,
                    root=self.output_dir,
                    robot_type=robot_type,
                    features=features,
                    use_videos=True,
                )
                break
            except FileExistsError:
                print(f"[Warning] Directory {self.output_dir} already exists.")
                ans = input("Overwrite? (y: overwrite, n: specify new directory name): ").strip().lower()
                if ans == 'y':
                    # Overwrite: Delete the existing directory and recreate it
                    import shutil
                    shutil.rmtree(self.output_dir)
                    print(f"{self.output_dir} deleted. Recreating...")
                    continue
                else:
                    new_dir = input("Enter new directory name: ").strip()
                    self.output_dir = self.output_dir.parent / new_dir
                    print(f"New output directory: {self.output_dir}")
                    continue

        if "teleop" not in self.dataset.meta.task_to_task_index:
            self.dataset.meta.add_task("teleop")

    def _get_messages(self, episode_dir):
        """Get messages from all required topics in the rosbag.

        Camera messages are not deserialized here to save memory; only their timestamps are kept
        (as (None, timestamp)). The images themselves are read later by _iter_frame_images.
        """
        messages = {
            '/joint_states': [],
            '/crane_plus_arm_controller/joint_trajectory': [],
            '/crane_plus_gripper_controller/joint_trajectory': [],
        }
        if self.camera_topic:
            messages[self.camera_topic] = []
        with Reader(episode_dir) as reader:
            connections = [x for x in reader.connections if x.topic in messages.keys()]
            for connection, timestamp, rawdata in reader.messages(connections=connections):
                if connection.topic == self.camera_topic:
                    messages[connection.topic].append((None, timestamp))
                    continue
                msg = deserialize_cdr(rawdata, connection.msgtype)
                messages[connection.topic].append((msg, timestamp))
        return messages

    def _iter_frame_images(self, episode_dir, frame_image_idx):
        """Yield one image per frame, where frame_image_idx[i] is the index of the camera message for frame i.

        Reads the camera messages in order and decodes only the ones that are used,
        so the whole episode's images never have to be held in memory.
        """
        f = 0
        with Reader(episode_dir) as reader:
            connections = [x for x in reader.connections if x.topic == self.camera_topic]
            for i, (connection, _, rawdata) in enumerate(reader.messages(connections=connections)):
                if f >= len(frame_image_idx):
                    break
                if frame_image_idx[f] != i:
                    continue
                img = decode_image(deserialize_cdr(rawdata, connection.msgtype), self.image_size)
                while f < len(frame_image_idx) and frame_image_idx[f] == i:
                    yield img
                    f += 1

    def _sample_and_hold(self, times, values, target_times):
        values = np.asarray(values)
        if len(values.shape) == 1:
            values = values.reshape(-1, 1)
            new_values = np.zeros((len(target_times), values.shape[1]))
        elif len(values.shape) >= 3:
            new_values = np.zeros((len(target_times), *values.shape[1:]), dtype=values.dtype)
        else:
            new_values = np.zeros((len(target_times), values.shape[1]))
        for i, target_time in enumerate(target_times):
            idx = np.searchsorted(times, target_time, side='right') - 1
            if idx < 0:
                idx = 0
            new_values[i] = values[idx]
        if len(values.shape) == 1:
            new_values = new_values.flatten()
        return new_values

    def convert(self):
        max_frames = 0
        episode_lengths = []
        print("\nFirst pass: calculating episode lengths...")
        for episode_dir in self.episode_dirs:
            try:
                messages = self._get_messages(episode_dir)
                j_times = np.array([ts for _, ts in messages['/joint_states']])
                a_times = np.array([ts for _, ts in messages['/crane_plus_arm_controller/joint_trajectory']])
                g_times = np.array([ts for _, ts in messages['/crane_plus_gripper_controller/joint_trajectory']])
                # Use the intersection of available times
                start_time = max(j_times[0], a_times[0], g_times[0])
                end_time = min(j_times[-1], a_times[-1], g_times[-1])
                if self.camera_topic:
                    i_times = np.array([ts for _, ts in messages[self.camera_topic]])
                    start_time = max(start_time, i_times[0])
                    end_time = min(end_time, i_times[-1])
                target_freq = self.target_freq or 20
                num_frames = int((end_time - start_time) * target_freq / 1e9)
                max_frames = max(max_frames, num_frames)
                episode_lengths.append(num_frames)
                print(f"Episode {episode_dir}: {num_frames} frames")
            except Exception as e:
                print(f"Error processing episode {episode_dir}: {e}")
                episode_lengths.append(0)
                continue

        print(f"\nLongest episode: {max_frames} frames")

        for idx, episode_dir in enumerate(self.episode_dirs):
            print(f"\nProcessing episode {idx} from {episode_dir}...")
            try:
                messages = self._get_messages(episode_dir)
                joint_states = messages['/joint_states']
                arm_cmds = messages['/crane_plus_arm_controller/joint_trajectory']
                gripper_cmds = messages['/crane_plus_gripper_controller/joint_trajectory']
                j_times = np.array([ts for _, ts in joint_states])
                j_pos = np.array([msg.position for msg, _ in joint_states])
                a_times = np.array([ts for _, ts in arm_cmds])
                a_pos = np.array([msg.points[0].positions if msg.points else [0,0,0,0] for msg, _ in arm_cmds])
                g_times = np.array([ts for _, ts in gripper_cmds])
                g_pos = np.array([msg.points[0].positions[0] if (msg.points and len(msg.points[0].positions)>0) else 0.0 for msg, _ in gripper_cmds])
                start_time = max(j_times[0], a_times[0], g_times[0])
                end_time = min(j_times[-1], a_times[-1], g_times[-1])
                if self.camera_topic:
                    i_times = np.array([ts for _, ts in messages[self.camera_topic]])
                    start_time = max(start_time, i_times[0])
                    end_time = min(end_time, i_times[-1])
                target_freq = self.target_freq or 20
                num_frames = episode_lengths[idx]
                target_times = np.linspace(start_time, end_time, num_frames)
                # Sample and hold for each
                j_pos_sampled = self._sample_and_hold(j_times, j_pos, target_times)  # (num_frames, 5)
                a_pos_sampled = self._sample_and_hold(a_times, a_pos, target_times)  # (num_frames, 4)
                g_pos_sampled = self._sample_and_hold(g_times, g_pos, target_times)  # (num_frames,)
                # Concatenate after sampling
                a_pos_full = np.hstack([a_pos_sampled, g_pos_sampled.reshape(-1,1)])  # (num_frames, 5)
                # Sample and hold for the camera too: each frame gets the latest image at that time
                images = None
                if self.camera_topic:
                    frame_image_idx = np.clip(np.searchsorted(i_times, target_times, side='right') - 1, 0, None)
                    images = self._iter_frame_images(episode_dir, frame_image_idx)
                # Add only this episode's own frames; LeRobot handles variable episode lengths
                # (out-of-episode action chunk steps are masked via action_is_pad during training)
                for i in range(num_frames):
                    frame = {
                        "observation.state": torch.tensor(j_pos_sampled[i], dtype=torch.float32),
                        "observation.environment_state": torch.tensor(j_pos_sampled[i], dtype=torch.float32),
                        "action": torch.tensor(a_pos_full[i], dtype=torch.float32),
                        "task": self.task_name,
                    }
                    if images is not None:
                        frame[self.image_key] = next(images)
                    self.dataset.add_frame(frame)
                print(f"Saving episode {idx}...")
                self.dataset.save_episode()
                print(f"Episode {idx} saved successfully")
            except Exception as e:
                print(f"Error processing episode {episode_dir}: {e}")
                import traceback
                traceback.print_exc()
                continue
        return str(self.dataset.root)

def load_config_py(config_path):
    spec = importlib.util.spec_from_file_location("crane_features", config_path)
    config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config)
    return config

def main():
    # Load parameters from conversion_config.yaml
    with open("configs/conversion_config.yaml", "r") as f:
        conv_cfg = yaml.safe_load(f)
    bag_dir = conv_cfg["bag_dir"]
    output_dir = conv_cfg["output_dir"]
    config = conv_cfg["config"]
    target_freq = conv_cfg.get("target_freq", 20)

    converter = RosbagToLeRobot(bag_dir, output_dir, config, target_freq)
    output_path = converter.convert()
    print(f"Dataset saved to: {output_path}")

if __name__ == '__main__':
    main() 