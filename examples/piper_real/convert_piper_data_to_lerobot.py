"""
Script to convert PiPER demonstration data to the LeRobot dataset v2.0 format.

This script processes demonstration data collected with PiPER bimanual setup using GELLO
teleoperation. It converts pickle files (robot_act.pkl, robot_obs.pkl) and video files
to the LeRobot dataset format.

Example usage:
    python convert_piper_data_to_lerobot.py --raw-dir data --repo-id aromanus/openpi_PiPER_demo

Author: Generated for PiPER bimanual setup
"""

import os
import dataclasses
import pickle
from pathlib import Path
import shutil
from typing import Literal
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
import numpy as np
import torch
import tqdm
import tyro
import os
from dotenv import load_dotenv
from src.util.data_util import load_raw_episode_data, load_pickle_file

load_dotenv(dotenv_path=".env")
HF_LEROBOT_HOME = os.getenv("HF_LEROBOT_HOME")

STRICT_ALIGN_FRAMES_OBSV = False
TRANSFORM_PIPER_TO_ALOHA_CONVENTION = False

@dataclasses.dataclass(frozen=True)
class DatasetConfig:
    use_videos: bool = True
    tolerance_s: float = 0.0001
    image_writer_processes: int = 10
    image_writer_threads: int = 5
    video_backend: str | None = None


DEFAULT_DATASET_CONFIG = DatasetConfig()


def create_empty_dataset(
    repo_id: str,
    robot_type: str,
    fps: int,
    mode: Literal["video", "image"] = "video",
    *,
    dataset_config: DatasetConfig = DEFAULT_DATASET_CONFIG,
) -> LeRobotDataset:
    """
    Create an empty LeRobot dataset with PiPER robot configuration.

    Args:
        repo_id: Repository ID for the dataset
        robot_type: Type of robot (e.g., "piper")
        fps: Frames per second for the dataset
        mode: Whether to use video or image format
        dataset_config: Configuration for dataset creation

    Returns:
        Empty LeRobotDataset ready to be populated
    """
    motors = [
        "left_waist",
        "left_shoulder",
        "left_elbow",
        "left_forearm_roll",
        "left_wrist_angle",
        "left_wrist_rotate",
        "left_gripper",
        "right_waist",
        "right_shoulder",
        "right_elbow",
        "right_forearm_roll",
        "right_wrist_angle",
        "right_wrist_rotate",
        "right_gripper"
    ]

    # Camera names matching PiPER setup
    cameras = [
        "top_camera",
        "left_camera",
        "right_camera",
    ]

    # Define features for the dataset
    features = {
        "observation.state": {
            "dtype": "float32",
            "shape": (len(motors),),
            "names": [motors],
        },
        "observation.velocity": {
            "dtype": "float32",
            "shape": (len(motors),),
            "names": [motors],
        },
        "observation.effort": {
            "dtype": "float32",
            "shape": (len(motors),),
            "names": [motors],
        },
        "action": {
            "dtype": "float32",
            "shape": (len(motors),),
            "names": [motors],
        },
    }

    # Add camera features for RGB cameras
    for cam in cameras:
        features[f"observation.images.{cam}"] = {
            "dtype": mode,
            "shape": (3, 480, 640),
            "names": [
                "channels",
                "height",
                "width",
            ],
        }

    # Remove existing dataset if it exists
    lerobot_path = os.path.join(HF_LEROBOT_HOME, repo_id)
    if Path(lerobot_path).exists():
        shutil.rmtree(lerobot_path)

    return LeRobotDataset.create(
        repo_id=repo_id,
        fps=fps,
        robot_type=robot_type,
        features=features,
        use_videos=dataset_config.use_videos,
        tolerance_s=dataset_config.tolerance_s,
        image_writer_processes=dataset_config.image_writer_processes,
        image_writer_threads=dataset_config.image_writer_threads,
        video_backend=dataset_config.video_backend,
    )



def infer_fps_from_timestamps(timestamps: np.ndarray) -> float:
    """
    Infer FPS from timestamp array.

    Args:
        timestamps: Array of timestamps

    Returns:
        Inferred FPS value
    """
    if len(timestamps) < 2:
        return 30.0  # Default fallback

    # Calculate time differences between consecutive frames
    time_diffs = np.diff(timestamps)
    # Remove outliers (anything more than 3 std devs from mean)
    mean_diff = np.mean(time_diffs)
    std_diff = np.std(time_diffs)
    valid_diffs = time_diffs[np.abs(time_diffs - mean_diff) < 3 * std_diff]

    if len(valid_diffs) == 0:
        return 30.0

    # FPS is 1 / average time difference
    avg_time_diff = np.mean(valid_diffs)
    fps = 1.0 / avg_time_diff if avg_time_diff > 0 else 30.0

    return round(fps)


def get_episode_directories(raw_dir: Path) -> list[Path]:
    """
    Get all valid episode directories, excluding .tmp directory.

    Args:
        raw_dir: Root directory containing episode subdirectories

    Returns:
        Sorted list of episode directory paths
    """
    project_dir = os.getcwd()
    episode_dirs = []
    for item in raw_dir.iterdir():
        if item.is_dir() and item.name != ".tmp":
            episode_dirs.append(os.path.join(project_dir, item))

    return sorted(episode_dirs)


def validate_episode_files(episode_dir: Path) -> tuple[bool, list[str]]:
    """
    Validate that an episode directory contains all required files.

    Args:
        episode_dir: Path to episode directory

    Returns:
        Tuple of (is_valid, missing_files) where is_valid is True if all files are present
        and missing_files is a list of missing file names
    """
    required_files = [
        "left_camera_color_timestamps.txt",
        "left_camera-depth.mp4",
        "left_camera_depth_timestamps.txt",
        "left_camera.mp4",
        "machine.jsonc",
        "rdc-meta.json",
        "right_camera_color_timestamps.txt",
        "right_camera-depth.mp4",
        "right_camera_depth_timestamps.txt",
        "right_camera.mp4",
        "robot_act.pkl",
        "robot_obs.pkl",
        "top_camera_color_timestamps.txt",
        "top_camera-depth.mp4",
        "top_camera_depth_timestamps.txt",
        "top_camera.mp4",
    ]

    missing_files = []
    for filename in required_files:
        file_path = episode_dir / filename
        if not file_path.exists():
            missing_files.append(filename)

    return len(missing_files) == 0, missing_files


def validate_all_episodes(episode_dirs: list[Path]) -> None:
    """
    Validate that all episode directories contain required files.

    Args:
        episode_dirs: List of episode directory paths

    Raises:
        ValueError: If any episode directory is missing required files
    """
    invalid_episodes = []

    for ep_dir in episode_dirs:
        is_valid, missing_files = validate_episode_files(Path(ep_dir))
        if not is_valid:
            invalid_episodes.append({
                "directory": ep_dir,
                "missing_files": missing_files
            })

    if invalid_episodes:
        error_msg = "Found episodes with missing required files:\n\n"
        for ep_info in invalid_episodes:
            error_msg += f"Episode: {ep_info['directory']}\n"
            error_msg += f"Missing files:\n"
            for missing_file in ep_info['missing_files']:
                error_msg += f"  - {missing_file}\n"
            error_msg += "\n"

        raise ValueError(error_msg)


# def align_observations_and_actions(
#     observations: list[dict],
#     actions: list[dict],
# ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
#     """
#     Align observations and actions by their timestamps and create aligned tensors.

#     Args:
#         observations: List of observation dictionaries
#         actions: List of action dictionaries

#     Returns:
#         Tuple of (aligned_states, aligned_actions, aligned_velocities, aligned_efforts) as torch tensors
#     """
#     # Extract timestamps and data
#     obs_timestamps = np.array([obs["robot_timestamp"][0] for obs in observations])
#     act_timestamps = np.array([act["action_timestamp"][0] for act in actions])

#     # Get data from observations
#     states = np.array([obs["joint_positions"] for obs in observations])
#     velocities = np.array([obs["joint_velocities"] for obs in observations])
#     efforts = np.array([obs["torques"] for obs in observations])
#     actions_data = np.array([act["action"] for act in actions])

#     # Find the overlapping time range
#     start_time = max(obs_timestamps[0], act_timestamps[0])
#     end_time = min(obs_timestamps[-1], act_timestamps[-1])

#     # Filter to overlapping range
#     obs_mask = (obs_timestamps >= start_time) & (obs_timestamps <= end_time)
#     act_mask = (act_timestamps >= start_time) & (act_timestamps <= end_time)

#     obs_timestamps = obs_timestamps[obs_mask]
#     act_timestamps = act_timestamps[act_mask]
#     states = states[obs_mask]
#     velocities = velocities[obs_mask]
#     efforts = efforts[obs_mask]
#     actions_data = actions_data[act_mask]

#     # Align by taking the minimum length (they should be close in length)
#     min_len = min(len(states), len(actions_data))

#     aligned_states = torch.from_numpy(states[:min_len]).float()
#     aligned_actions = torch.from_numpy(actions_data[:min_len]).float()
#     aligned_velocities = torch.from_numpy(velocities[:min_len]).float()
#     aligned_efforts = torch.from_numpy(efforts[:min_len]).float()

#     return aligned_states, aligned_actions, aligned_velocities, aligned_efforts





def populate_dataset(
    dataset: LeRobotDataset,
    episode_dirs: list[Path],
    task: str,
    episodes: list[int] | None = None,
) -> LeRobotDataset:
    """
    Populate the LeRobot dataset with data from PiPER demonstrations.

    Args:
        dataset: Empty LeRobotDataset to populate
        episode_dirs: List of episode directory paths
        task: Task description
        episodes: Optional list of specific episode indices to process

    Returns:
        Populated LeRobotDataset
    """
    if episodes is None:
        episodes = range(len(episode_dirs))

    for ep_idx in tqdm.tqdm(episodes, desc="Processing episodes"):
        ep_dir = episode_dirs[ep_idx]

        # Load episode data
        imgs_per_cam, states, actions, velocities, efforts = load_raw_episode_data(ep_dir)

        num_states = states.shape[0]

        # Add frames to dataset
        for i in range(num_states):
            frame = {
                "observation.state": states[i].astype(np.float32),
                "observation.velocity": velocities[i].astype(np.float32),
                "observation.effort": efforts[i].astype(np.float32),
                "action": actions[i].astype(np.float32),
                "task": task
            }

            try:
                for camera, img_array in imgs_per_cam.items():
                    frame[f"observation.images.{camera}"] = img_array[i]

            except Exception as e:
                print(e)
                print(f'Not enough images in frame to create full frame dictionary:\n\n Number of states: {num_states} \n Number of frames with images: {len(img_array)}')
        
            # Save the episode
            dataset.add_frame(frame)
        
        dataset.save_episode()

    return dataset


def infer_fps_from_episode(episode_dir: Path) -> float:
    """
    Infer FPS from a single episode's timestamp data.

    Args:
        episode_dir: Path to episode directory

    Returns:
        Inferred FPS
    """
    robot_obs = load_pickle_file(os.path.join(episode_dir, "robot_obs.pkl"))
    timestamps = np.array([obs["robot_timestamp"][0] for obs in robot_obs])
    return infer_fps_from_timestamps(timestamps)


def port_piper(
    raw_dir: Path,
    repo_id: str,
    task: str = "object_transportation",
    *,
    episodes: list[int] | None = None,
    push_to_hub: bool = False,
    token: str | None = None,
    mode: Literal["video", "image"] = "video",
    dataset_config: DatasetConfig = DEFAULT_DATASET_CONFIG,
):
    """
    Convert PiPER demonstration data to LeRobot dataset format.

    Args:
        raw_dir: Directory containing episode subdirectories
        repo_id: Repository ID for the dataset
        task: Task description
        episodes: Optional list of specific episode indices to process
        push_to_hub: Whether to push the dataset to Hugging Face Hub
        token: Hugging Face access token for pushing to hub
        mode: Whether to use video or image format
        dataset_config: Configuration for dataset creation
    """
    # Remove existing dataset if it exists
    lerobot_path = os.path.join(HF_LEROBOT_HOME, repo_id)
    if Path(lerobot_path).exists():
        shutil.rmtree(lerobot_path)

    if not raw_dir.exists():
        raise ValueError(f"Raw data directory does not exist: {raw_dir}")

    # Validate token is provided if push_to_hub is enabled
    if push_to_hub and not token:
        raise ValueError(
            "Hugging Face token is required when --push-to-hub is enabled. "
            "Please provide a token using the --token argument."
        )

    # Get all episode directories
    episode_dirs = get_episode_directories(raw_dir)

    if len(episode_dirs) == 0:
        raise ValueError(f"No episode directories found in {raw_dir}")

    print(f"Found {len(episode_dirs)} episodes to process")

    # Validate all episodes have required files before processing
    print("Validating episode files...")
    validate_all_episodes(episode_dirs)
    print("All episodes validated successfully")

    # Infer FPS from first episode
    fps = 30
    print(f"Inferred FPS: {fps}")

    # Create empty dataset
    dataset = create_empty_dataset(
        repo_id,
        robot_type="piper",
        fps=fps,
        mode=mode,
        dataset_config=dataset_config,
    )

    # Populate dataset
    dataset = populate_dataset(
        dataset,
        episode_dirs,
        task=task,
        episodes=episodes
    )

    # Consolidate the dataset
    # print("Consolidating dataset...")
    # dataset.consolidate()

    # Optionally push to hub
    if push_to_hub:
        print("Pushing to Hugging Face Hub...")
        dataset.push_to_hub(token=token)

    print(f"Dataset created successfully at {lerobot_path}")


if __name__ == "__main__":
    tyro.cli(port_piper)
