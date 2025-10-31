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


@dataclasses.dataclass(frozen=True)
class PiPERInputs(transforms.DataTransformFn):
    """Inputs for the PiPER policy.

    Expected inputs:
    - images: dict[name, img] where img is [channel, height, width]. name must be in EXPECTED_CAMERAS.
    - state: [14]
    - actions: [action_horizon, 14]
    """

    action_dim: int

    # The expected cameras names. All input cameras must be in this set. Missing cameras will be
    # replaced with black images and the corresponding `image_mask` will be set to False.
    EXPECTED_CAMERAS: ClassVar[tuple[str, ...]] = ("top_camera", "left_camera", "right_camera")

    model_type: _model.ModelType = _model.ModelType.PI05

    def __call__(self, data: dict) -> dict:
        # Transform state to match Pi 0's expected format (apply joint flipping and gripper conversion)
        state = data["observation/state"]

        base_image = _parse_image(data["observation/imgs/top_camera"])
        right_image = _parse_image(data["observation/imgs/right_camera"])
        left_image = _parse_image(data["observation/imgs/left_camera"])

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

    def __call__(self, data: dict) -> dict:
        # Only return the first 14 dims.
        actions = np.asarray(data["actions"][:, :14])
        # actions_copy = actions.copy()

        # actions[:, :7] = actions_copy[:, 7:]
        # actions[:, 7:] = actions_copy[:, :7]
        return {"actions": actions}
