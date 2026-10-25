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
Train a policy (ACT or Diffusion Policy) on the collected dataset.

This script trains the policy selected by `policy` using configuration parameters specified in:
    - configs/train_config.yaml: Policy type and its parameters, batch size, learning rate, etc.
    - configs/crane_features*.py: Dataset and feature configuration

Usage:
    pixi run train

"""


from pathlib import Path
import csv
import importlib.util
import shutil
import subprocess
import sys
import time
from datetime import datetime
import matplotlib
import matplotlib.pyplot as plt
import wandb
import torch
import yaml
from lerobot.common.datasets.factory import resolve_delta_timestamps
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.common.datasets.utils import dataset_to_policy_features
from lerobot.common.policies.factory import get_policy_class, make_policy_config
from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import FeatureType, PolicyFeature

matplotlib.use("Agg")

# Policies this script has been checked with (any LeRobot policy type could be added here)
SUPPORTED_POLICIES = ("act", "diffusion")

def run_git(args: list[str], repo_dir: Path) -> str | None:
    """Run a git command in repo_dir and return its output, or None if git or the repository is unavailable."""
    try:
        result = subprocess.run(["git", *args], cwd=repo_dir, capture_output=True, text=True, check=True)
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip()

def save_run_info(output_dir: Path, files_to_copy: list[Path]):
    """Save what is needed to reproduce this run next to the checkpoint.

    - run_info.yaml: git commit of pai_training and lerobot, whether there were uncommitted changes, command
    - git_diff.patch: uncommitted changes to tracked files (only if any)
    - code/: copies of files_to_copy (also covers untracked files, which are not in the diff)
    """
    repo_dir = Path(__file__).resolve().parent.parent
    status = run_git(["status", "--porcelain"], repo_dir)
    run_info = {
        "date": datetime.now().isoformat(timespec="seconds"),
        "command": " ".join(sys.argv),
        "git_commit": run_git(["rev-parse", "HEAD"], repo_dir),
        "git_dirty": bool(status) if status is not None else None,
        "lerobot_commit": run_git(["rev-parse", "HEAD"], repo_dir / "lerobot"),
    }
    with open(output_dir / "run_info.yaml", "w") as f:
        yaml.safe_dump(run_info, f, sort_keys=False)

    diff = run_git(["diff", "HEAD"], repo_dir)
    if diff:
        (output_dir / "git_diff.patch").write_text(diff + "\n")

    code_dir = output_dir / "code"
    code_dir.mkdir(exist_ok=True)
    for path in files_to_copy:
        shutil.copy(path, code_dir / path.name)

def load_config_py(config_path: Path):
    spec = importlib.util.spec_from_file_location("crane_features", config_path)
    config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config)
    return config

def load_train_config(yaml_path: Path):
    with open(yaml_path, 'r') as f:
        return yaml.safe_load(f)

def build_policy_config(
    policy_type: str,
    params: dict,
    input_features: dict[str, PolicyFeature],
    output_features: dict[str, PolicyFeature],
    device: torch.device,
) -> PreTrainedConfig:
    """Create the LeRobot config of the selected policy from its section in train_config.yaml."""
    if policy_type not in SUPPORTED_POLICIES:
        raise ValueError(f"Unsupported policy '{policy_type}'. Choose one of {SUPPORTED_POLICIES}.")
    # YAML has no tuples, but LeRobot configs use them (e.g. crop_shape)
    params = {key: tuple(value) if isinstance(value, list) else value for key, value in params.items()}
    return make_policy_config(
        policy_type,
        input_features=input_features,
        output_features=output_features,
        device=device.type,
        **params,
    )

def print_delta_timestamps(delta_timestamps: dict[str, list[float]]):
    """Show which frames (relative to the current one) are loaded for each feature."""
    for key, timestamps in delta_timestamps.items():
        print(f"delta_timestamps {key}: {len(timestamps)} steps ({timestamps[0]:.2f} .. {timestamps[-1]:.2f} s)")

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
POLICY_TYPE = train_cfg["policy"]
# Policy-specific parameters, e.g. the `act:` or `diffusion:` section
POLICY_PARAMS = train_cfg.get(POLICY_TYPE) or {}
DATASET_ROOT = Path(train_cfg["dataset_root"])
REPO_ID = train_cfg["repo_id"]
OUTPUT_DIRECTORY = Path(train_cfg["output_directory"].format(policy=POLICY_TYPE))
CONFIG_PATH = Path(train_cfg["config_path"])
DEVICE = torch.device(train_cfg["device"])
TRAINING_STEPS = train_cfg["training_steps"]
LOG_FREQ = train_cfg["log_freq"]
NUM_WORKERS = train_cfg["num_workers"]
BATCH_SIZE = train_cfg["batch_size"]
LEARNING_RATE = train_cfg["learning_rate"]
# =====================

def main():
    # Create a directory to store the training checkpoint.
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    # Saved before training so that it is available even if the run fails
    save_run_info(OUTPUT_DIRECTORY, [Path(__file__), TRAIN_CONFIG_PATH, CONFIG_PATH])

    config = load_config_py(CONFIG_PATH)
    features = config.features

    # dir: wandb's local files (e.g. offline runs with WANDB_MODE=offline) go under the output directory
    wandb.init(project="pai_crane_gazebo", dir=OUTPUT_DIRECTORY, config={
        "policy": POLICY_TYPE,
        **POLICY_PARAMS,
        "repo_id": REPO_ID,
        "dataset_root": DATASET_ROOT,
        "training_steps": TRAINING_STEPS,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "device": str(DEVICE),
    })

    dataset_metadata = LeRobotDatasetMetadata(REPO_ID, root=DATASET_ROOT)
    features = dataset_to_policy_features(features)
    output_features = {key: ft for key, ft in features.items() if ft.type is FeatureType.ACTION}
    input_features = {key: ft for key, ft in features.items() if key not in output_features}
    print(f"input_features: {input_features}")
    print(f"output_features: {output_features}")

    cfg = build_policy_config(POLICY_TYPE, POLICY_PARAMS, input_features, output_features, DEVICE)
    policy = get_policy_class(POLICY_TYPE)(cfg, dataset_stats=dataset_metadata.stats)
    policy.train()
    policy.to(DEVICE)
    print(f"policy: {POLICY_TYPE} {POLICY_PARAMS}")

    # Each policy knows which past/future frames it needs (ACT: current observation and chunk_size
    # future actions, Diffusion: n_obs_steps past observations and horizon actions)
    delta_timestamps = resolve_delta_timestamps(cfg, dataset_metadata)
    print_delta_timestamps(delta_timestamps)

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
            # The batch already has the shapes the policy expects, because delta_timestamps comes from
            # its config (ACT: no time dimension on observations, Diffusion: (B, n_obs_steps, ...))
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
