"""
Minimal example script for converting a dataset to LeRobot format.

Usage:
python convert_franka_data_to_lerobot.py --raw_dir /path/to/your/data --repo-id username/dataset_name --task "task description"

If you want to push your dataset to the Hugging Face Hub:
python convert_franka_data_to_lerobot.py --raw_dir /path/to/your/data --repo-id username/dataset_name --task "task description" --push-to-hub --token $HF_ACCESS_TOKEN
"""

import argparse
import shutil
import os
import json
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from pathlib import Path
from PIL import Image
from tqdm import tqdm


load_dotenv(dotenv_path=".env")
HF_LEROBOT_HOME = os.getenv("HF_LEROBOT_HOME")
if HF_LEROBOT_HOME:
    HF_LEROBOT_HOME = Path(HF_LEROBOT_HOME)

PI05_PROMPT_PREFIX = '<control mode> joint <control mode> '
TASK = PI05_PROMPT_PREFIX + 'Pick up the ball and place it in bowl.'
DOWNSAMPLE_IMAGES_FOR_LEROBOT = True


def downsample_image(
    img: Image.Image,
    target_size: tuple[int, int] = (224, 224),
) -> Image.Image:
    """
    Downsample a single PIL Image to the target size.

    Args:
        img: PIL Image to downsample
        target_size: Target dimensions (width, height)

    Returns:
        Downsampled PIL Image
    """
    return img.resize(target_size, Image.LANCZOS)

def list_subfolders(data_dir: str) -> list[str]:
    folder_path = Path(data_dir)
    return [f.name for f in folder_path.iterdir() if f.is_dir()]


def count_jpg(folder):
    return len(list(Path(folder).rglob("*.jpg")))


def main():
    parser = argparse.ArgumentParser(description="Convert Franka data to LeRobot format")
    parser.add_argument(
        "--raw_dir",
        type=str,
        required=True,
        help="Path to the raw data directory containing episode folders (0, 1, 2, ...)",
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        required=True,
        help="Repository ID for the dataset (e.g., username/dataset_name)",
    )
    parser.add_argument(
        "--push-to-hub",
        action="store_true",
        help="Push the dataset to Hugging Face Hub",
    )
    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help="Hugging Face access token for pushing to hub",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=30,
        help="Frames per second for the dataset",
    )

    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    repo_id = args.repo_id
    task = TASK
    push_to_hub = args.push_to_hub
    token = args.token
    fps = args.fps

    # Clean up any existing dataset in the output directory
    if HF_LEROBOT_HOME:
        output_path = HF_LEROBOT_HOME / repo_id
        if output_path.exists():
            shutil.rmtree(output_path)

    # Create LeRobot dataset, define features to store
    # OpenPi assumes that proprio is stored in `state` and actions in `action`
    # LeRobot assumes that dtype of image data is `image`
    # robot1_camera1 = front facing camera (image)
    # robot1_camera2 = wrist camera (wrist_image)
    image_shape = (224, 224, 3) if DOWNSAMPLE_IMAGES_FOR_LEROBOT else (256, 256, 3)
    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        robot_type="panda",
        fps=fps,
        features={
            "image": {
                "dtype": "image",
                "shape": image_shape,
                "names": ["height", "width", "channel"],
            },
            "wrist_image": {
                "dtype": "image",
                "shape": image_shape,
                "names": ["height", "width", "channel"],
            },
            "state": {
                "dtype": "float32",
                "shape": (8,),
                "names": ["state"],
            },
            "actions": {
                "dtype": "float32",
                "shape": (8,),
                "names": ["actions"],
            },
        },
        image_writer_threads=10,
        image_writer_processes=5,
    )

    # Get all episode folders (numbered directories)
    episode_folders = [
        f for f in raw_dir.iterdir()
        if f.is_dir() and f.name.isdigit()
    ]
    # Sort by episode number
    episode_folders = sorted(episode_folders, key=lambda x: int(x.name))

    print(f"Found {len(episode_folders)} episodes in {raw_dir}")

    for episode_folder in episode_folders:
        episode_num = episode_folder.name
        print(f"Processing episode {episode_num}...")

        robot1_camera1_folder = episode_folder / "images" / "robot1_camera1"
        robot1_camera2_folder = episode_folder / "images" / "robot1_camera2"
        gello_joint_folder = episode_folder / "gello_joints"
        gello_gripper_folder = episode_folder / "gello_gripper"
        joint_states_folder = episode_folder / "joint_states"

        # Check if required folders exist
        if not robot1_camera1_folder.exists():
            print(f"  Skipping episode {episode_num}: robot1_camera1 folder not found")
            continue
        if not robot1_camera2_folder.exists():
            print(f"  Skipping episode {episode_num}: robot1_camera2 folder not found")
            continue

        # Load JSON data
        with open(gello_joint_folder / "data.json") as f:
            gello_joints_json = json.load(f)
        with open(gello_gripper_folder / "data.json") as f:
            gello_gripper_json = json.load(f)
        with open(joint_states_folder / "data.json") as f:
            joint_states_json = json.load(f)

        num_jpg = count_jpg(robot1_camera1_folder)
        print(f"  Found {num_jpg} frames")

        for i in range(num_jpg):
            # Load images
            # robot1_camera1 = front facing camera = "image"
            # robot1_camera2 = wrist camera = "wrist_image"
            robot1_camera1_image_path = robot1_camera1_folder / f"{i}.jpg"
            robot1_camera2_image_path = robot1_camera2_folder / f"{i}.jpg"

            if not robot1_camera1_image_path.exists() or not robot1_camera2_image_path.exists():
                print(f"  Warning: Missing image at frame {i}, skipping")
                continue

            robot1_camera1_image = Image.open(robot1_camera1_image_path)
            robot1_camera2_image = Image.open(robot1_camera2_image_path)

            if DOWNSAMPLE_IMAGES_FOR_LEROBOT:
                robot1_camera1_image = downsample_image(robot1_camera1_image)
                robot1_camera2_image = downsample_image(robot1_camera2_image)

            # Get action from gello (7 joints + 1 gripper = 8 dims)
            current_joint = np.array(gello_joints_json[str(i)][0], dtype=np.float32)
            current_gripper = np.array([gello_gripper_json[str(i)][0]], dtype=np.float32)
            current_action = np.concatenate([current_joint, current_gripper])

            # Get state from joint_states (7 joints + gripper binary = 8 dims)
            joint_state = joint_states_json[str(i)][0]["position"]
            joint_state = np.array(joint_state)
            # Convert gripper: fingers at indices 7 and 8, sum < 0.04 means closed (0), otherwise open (1)
            joint_state[7] = 0 if joint_state[7] + joint_state[8] < 0.5 * 0.08 else 1
            joint_state = joint_state[:8]
            joint_state = joint_state.astype(np.float32)

            dataset.add_frame(
                {
                    "image": robot1_camera1_image,  # front facing camera
                    "wrist_image": robot1_camera2_image,  # wrist camera
                    "state": joint_state,
                    "actions": current_action,
                    "task": task,
                }
            )

        dataset.save_episode()
        print(f"  Episode {episode_num} saved")

    print(f"\nDataset creation complete. Total episodes: {len(episode_folders)}")

    # Optionally push to the Hugging Face Hub
    if push_to_hub:
        print("Pushing to Hugging Face Hub...")
        dataset.push_to_hub(
            tags=["franka", "panda", "gello"],
            private=False,
            push_videos=True,
            license="apache-2.0",
            token=token,
        )
        print("Push complete!")


if __name__ == "__main__":
    main()
