import dataclasses
from typing import ClassVar
import einops
import numpy as np
from openpi import transforms
from openpi.models import model as _model


def make_piper_example() -> dict:
    """Creates a random input example for the PiPER policy."""
    return {
        "observation/state": np.ones((14,)),
        "observation/imgs/top_camera": np.random.randint(256, size=(3, 224, 224), dtype=np.uint8),
        "observation/imgs/right_camera": np.random.randint(256, size=(3, 224, 224), dtype=np.uint8),
        "observation/imgs/left_camera": np.random.randint(256, size=(3, 224, 224), dtype=np.uint8),
        "prompt": 'do something'
    }

def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


def transform_piper_observation_to_aloha(
    imgs_per_cam: dict[str, np.ndarray],
    state: np.ndarray
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """
    Transform PiPER observation to match ALOHA bimanual conventions.

    PiPER convention: [left_arm (0-6), right_arm (7-13)]
    ALOHA convention: [right_arm (0-6), left_arm (7-13)]

    This function performs:
    1. Swaps joint observations between left and right arms
    2. Swaps left and right camera feeds
    3. Horizontally flips all camera frames to mirror the spatial layout

    This ensures visual-proprioceptive consistency with ALOHA pretraining.

    Args:
        imgs_per_cam: Dictionary mapping camera names to image arrays (H, W, C)
        state: Joint position observations (14,)

    Returns:
        Tuple of transformed (imgs_per_cam, state)
    """
    # Step 1: Swap left and right arm proprioceptive data (dims 0-6 ↔ dims 7-13)
    transformed_state = state.copy()
    transformed_state[:7] = state[7:14]
    transformed_state[7:14] = state[:7]

    # Step 2: Swap left and right camera feeds
    transformed_imgs = {}

    # Swap left ↔ right camera arrays
    if "left_camera" in imgs_per_cam and "right_camera" in imgs_per_cam:
        transformed_imgs["left_camera"] = imgs_per_cam["right_camera"]
        transformed_imgs["right_camera"] = imgs_per_cam["left_camera"]
    elif "left_camera" in imgs_per_cam:
        transformed_imgs["right_camera"] = imgs_per_cam["left_camera"]
    elif "right_camera" in imgs_per_cam:
        transformed_imgs["left_camera"] = imgs_per_cam["right_camera"]

    # Keep top camera (will be flipped below)
    if "top_camera" in imgs_per_cam:
        transformed_imgs["top_camera"] = imgs_per_cam["top_camera"]

    # Step 3: Horizontally flip all camera frames (left-right mirror)
    # This ensures that spatial layout matches the swapped proprioception
    for cam_name in transformed_imgs:
        # Flip along width axis (axis=1 for shape (H, W, C))
        transformed_imgs[cam_name] = np.flip(transformed_imgs[cam_name], axis=1).copy()

    return transformed_imgs, transformed_state


def transform_aloha_actions_to_piper(
    actions: np.ndarray
) -> np.ndarray:
    """
    Transform actions from ALOHA convention back to PiPER convention.

    This reverses the transformation done in transform_piper_observation_to_aloha.

    ALOHA convention (model output): [right_arm (0-6), left_arm (7-13)]
    PiPER convention (robot expects): [left_arm (0-6), right_arm (7-13)]

    Args:
        actions: Action vectors in ALOHA format (action_horizon, 14) or (14,)

    Returns:
        Transformed actions in PiPER format
    """
    transformed_actions = actions.copy()

    # Handle both single action (14,) and action sequence (action_horizon, 14)
    if actions.ndim == 1:
        # Single action: swap dims 0-6 ↔ dims 7-13
        transformed_actions[:7] = actions[7:14]
        transformed_actions[7:14] = actions[:7]
    else:
        # Action sequence: swap dims 0-6 ↔ dims 7-13 for all timesteps
        transformed_actions[:, :7] = actions[:, 7:14]
        transformed_actions[:, 7:14] = actions[:, :7]

    return transformed_actions


@dataclasses.dataclass(frozen=True)
class PiPERInputs(transforms.DataTransformFn):
    """Inputs for the PiPER policy.

    Expected inputs:
    - images: dict[name, img] where img is [channel, height, width]. name must be in EXPECTED_CAMERAS.
    - state: [14]
    - actions: [action_horizon, 14]
    """

    action_dim: int

    # If true, this will convert the joint and gripper values from the PiPER space to
    # the space used by the pi internal runtime which was used to train the base model.
    adapt_to_pi: bool = False

    # The expected cameras names. All input cameras must be in this set. Missing cameras will be
    # replaced with black images and the corresponding `image_mask` will be set to False.
    EXPECTED_CAMERAS: ClassVar[tuple[str, ...]] = ("top_camera", "left_camera", "right_camera")

    model_type: _model.ModelType = _model.ModelType.PI05
    
    # piper_to_aloha = PiperToAlohaAdapter()


    def __call__(self, data: dict) -> dict:
        # Transform state to match Pi 0's expected format (apply joint flipping and gripper conversion)
        state = data["observation/state"]

        # Parse images first
        imgs_per_cam = {
            "top_camera": _parse_image(data["observation/imgs/top_camera"]),
            "right_camera": _parse_image(data["observation/imgs/right_camera"]),
            "left_camera": _parse_image(data["observation/imgs/left_camera"])
        }

        # Apply PiPER to ALOHA transformation if adapt_to_pi is enabled
        if self.adapt_to_pi:
            imgs_per_cam, state = transform_piper_observation_to_aloha(imgs_per_cam, state)

        # Extract transformed images
        base_image = imgs_per_cam["top_camera"]
        right_image = imgs_per_cam["right_camera"]
        left_image = imgs_per_cam["left_camera"]

        state = transforms.pad_to_dim(state, self.action_dim)
        # Create inputs dict. Do not change the keys in the dict below.
        inputs = {
            "state": state,
            "image": {
                "base_0_rgb": base_image,
                "left_wrist_0_rgb": left_image,
                "right_wrist_0_rgb": right_image,
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                "right_wrist_0_rgb": np.True_,
            },
        }

        # Pad actions to the model action dimension. Keep this for your own dataset.
        # Actions are only available during training.
        if "action" in data:
            # We are padding to the model action dim.
            actions = transforms.pad_to_dim(data["action"], self.action_dim)
            # Transform actions using the inverse encoding (for training data)
            # actions = _encode_actions_inv(actions, adapt_to_pi=self.adapt_to_pi)
            inputs["actions"] = actions

        # Pass the prompt (aka language instruction) to the model.
        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class PiPEROutputs(transforms.DataTransformFn):
    """Outputs for the PiPER policy."""

    # If true, this will convert the joint and gripper values from the pi internal runtime space
    # back to the PiPER space for execution.
    adapt_to_pi: bool = False

    # piper_to_aloha = PiperToAlohaAdapter()


    def __call__(self, data: dict) -> dict:
        # Only return the first 14 dims.
        actions = np.asarray(data["actions"][:, :14])

        # Apply ALOHA to PiPER transformation if adapt_to_pi is enabled
        if self.adapt_to_pi:
            actions = transform_aloha_actions_to_piper(actions)

        return {"actions": actions}
