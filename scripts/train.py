# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
# Copyright 2024 Tatsuya Kamijo. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Author: Tatsuya Kamijo (based on HuggingFace LeRobot lerobot/examples/3_train_policy.py)

"""
Train ACT (Action Chunking with Transformer) policy on the collected dataset.

This script trains an ACT policy using configuration parameters specified in:
    - configs/train_config.yaml: Training hyperparameters like batch size, learning rate, etc.
    - configs/crane_features.py: Dataset and feature configuration

Usage:
    pixi run train

"""


from pathlib import Path
import csv
import importlib.util
import shutil
import time
import matplotlib
import matplotlib.pyplot as plt
import wandb
import torch
import yaml
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.common.datasets.utils import dataset_to_policy_features
from lerobot.common.policies.act.configuration_act import ACTConfig
from lerobot.common.policies.act.modeling_act import ACTPolicy
from lerobot.configs.types import FeatureType

matplotlib.use("Agg")

def load_config_py(config_path: Path):
    spec = importlib.util.spec_from_file_location("crane_features", config_path)
    config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config)
    return config

def load_train_config(yaml_path: Path):
    with open(yaml_path, 'r') as f:
        return yaml.safe_load(f)

def build_delta_timestamps(base: dict, chunk_size: int, fps: int) -> dict:
    """Add the action timestamps to the observation ones from the YAML.

    ACT predicts chunk_size future actions starting at the current frame, spaced 1/fps:
    [0, 1/fps, ..., (chunk_size - 1)/fps].
    """
    delta_timestamps = dict(base)
    delta_timestamps["action"] = [i / fps for i in range(chunk_size)]
    print(f"action delta_timestamps: {chunk_size} steps "
          f"(0.0 .. {delta_timestamps['action'][-1]:.2f} s at {fps} Hz)")
    return delta_timestamps

def open_train_log(log_path: Path):
    """Open the loss log CSV (readable without wandb) and write the header. Returns (file, writer)."""
    log_file = open(log_path, "w", newline="")
    log_writer = csv.writer(log_file)
    log_writer.writerow(["step", "loss", "elapsed_s"])
    return log_file, log_writer

def save_loss_plot(steps: list, losses: list, plot_path: Path):
    """Save the loss curve (log scale) as an image."""
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(steps, losses)
    ax.set_xlabel("step")
    ax.set_ylabel("loss")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(plot_path)
    plt.close(fig)

# =====================
# Load parameters from YAML
# =====================
TRAIN_CONFIG_PATH = Path("configs/train_config.yaml")
train_cfg = load_train_config(TRAIN_CONFIG_PATH)
DATASET_ROOT = Path(train_cfg["dataset_root"])
REPO_ID = train_cfg["repo_id"]
OUTPUT_DIRECTORY = Path(train_cfg["output_directory"])
CONFIG_PATH = Path(train_cfg["config_path"])
DEVICE = torch.device(train_cfg["device"])
TRAINING_STEPS = train_cfg["training_steps"]
LOG_FREQ = train_cfg["log_freq"]
NUM_WORKERS = train_cfg["num_workers"]
BATCH_SIZE = train_cfg["batch_size"]
CHUNK_SIZE = train_cfg["chunk_size"]
N_ACTION_STEPS = train_cfg["n_action_steps"]
LEARNING_RATE = train_cfg["learning_rate"]
DELTA_TIMESTAMPS = train_cfg["delta_timestamps"]
# =====================

def main():
    # Create a directory to store the training checkpoint.
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)

    config = load_config_py(CONFIG_PATH)
    features = config.features

    # dir: wandb's local files (e.g. offline runs with WANDB_MODE=offline) go under the output directory
    wandb.init(project="pai_crane_gazebo", dir=OUTPUT_DIRECTORY, config={
        "repo_id": REPO_ID,
        "dataset_root": DATASET_ROOT,
        "training_steps": TRAINING_STEPS,
        "batch_size": BATCH_SIZE,
        "chunk_size": CHUNK_SIZE,
        "n_action_steps": N_ACTION_STEPS,
        "device": str(DEVICE),
    })

    dataset_metadata = LeRobotDatasetMetadata(REPO_ID, root=DATASET_ROOT)
    features = dataset_to_policy_features(features)
    output_features = {key: ft for key, ft in features.items() if ft.type is FeatureType.ACTION}
    input_features = {key: ft for key, ft in features.items() if key not in output_features}
    print(f"input_features: {input_features}")
    print(f"output_features: {output_features}")

    cfg = ACTConfig(
        input_features=input_features,
        output_features=output_features,
        chunk_size=CHUNK_SIZE,
        n_action_steps=N_ACTION_STEPS
    )
    policy = ACTPolicy(cfg, dataset_stats=dataset_metadata.stats)
    policy.train()
    policy.to(DEVICE)

    delta_timestamps = build_delta_timestamps(
        DELTA_TIMESTAMPS,
        CHUNK_SIZE,
        dataset_metadata.fps,
    )

    # Instantiate the dataset
    dataset = LeRobotDataset(REPO_ID, root=DATASET_ROOT, delta_timestamps=delta_timestamps)

    optimizer = torch.optim.Adam(policy.parameters(), lr=LEARNING_RATE)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        num_workers=NUM_WORKERS,
        batch_size=BATCH_SIZE,
        shuffle=True,
        pin_memory=False,
        drop_last=True,
    )

    # Loss log saved next to the checkpoint (readable without wandb)
    log_path = OUTPUT_DIRECTORY / "train_log.csv"
    log_file, log_writer = open_train_log(log_path)
    log_steps, log_losses = [], []
    start_time = time.time()

    step = 0
    done = False
    while not done:
        for batch in dataloader:
            batch = {k: (v.to(DEVICE) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
            # Remove the sequence_length dimension to adapt to the ACT model implementation
            # (state: (B, 1, D) -> (B, D)). Video frames with a single timestamp already come
            # without it ((B, C, H, W)), so only inputs with one extra dimension are squeezed.
            for key, ft in input_features.items():
                if key in DELTA_TIMESTAMPS and batch[key].ndim == len(ft.shape) + 2:
                    batch[key] = batch[key][:, -1]
            # action is left as is
            loss, _ = policy.forward(batch)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if step % LOG_FREQ == 0:
                elapsed = time.time() - start_time
                print(f"step: {step} loss: {loss.item():.3f} elapsed: {elapsed:.1f}s")
                wandb.log({"loss": loss.item(), "step": step})
                log_writer.writerow([step, loss.item(), round(elapsed, 2)])
                log_file.flush()
                log_steps.append(step)
                log_losses.append(loss.item())
            step += 1
            if step >= TRAINING_STEPS:
                done = True
                break

    log_file.close()
    save_loss_plot(log_steps, log_losses, OUTPUT_DIRECTORY / "loss.png")
    print(f"loss log saved to {log_path}")

    policy.save_pretrained(OUTPUT_DIRECTORY)
    # Keep the training hyperparameters (lr, delta_timestamps, etc.) alongside the checkpoint
    shutil.copy(TRAIN_CONFIG_PATH, OUTPUT_DIRECTORY / "train_config.yaml")
    print(f"policy saved to {OUTPUT_DIRECTORY}")
    wandb.finish()

if __name__ == "__main__":
    main()
