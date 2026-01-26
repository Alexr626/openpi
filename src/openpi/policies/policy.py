from collections.abc import Sequence
from datetime import datetime
import logging
import pathlib
import time
from typing import Any, TypeAlias

import flax
import flax.traverse_util
import jax
import jax.numpy as jnp
import numpy as np
from openpi_client import base_policy as _base_policy
import torch
from typing_extensions import override
from transformers import AutoTokenizer, AutoProcessor

from openpi import transforms as _transforms
from openpi.models import model as _model
from openpi.shared import array_typing as at
from openpi.shared import nnx_utils
from openpi.models_pytorch.Activation_Engineering_helpers import pad_tokens, get_text_based_hidden_states, get_text_based_hidden_states_from_vla, SteeringHook, MultiLayerSteeringHook
from openpi.models_pytorch.CAST_helpers import (
    CASTMultiLayerHook,
    load_layer_cast_config,
    load_and_combine_precomputed_vectors,
    vectors_exist,
    # VTI imports
    VTIHook,
    load_vti_vectors,
    get_vti_metadata
)
from constants import (
    CAST,
    PHASE_CAST,
    VTI,
    VTI_LANGUAGE_LAYER_INDICES,
    VTI_IMAGE_LAYER_INDICES,
    ALPHA,
    CAST_THRESHOLD,
    CAST_LOG_PATH,
    DEPLOYMENT_SEED,
    # Layer-based CAST constants
    CAST_LAYER_CONFIG_PATH,
    CAST_LAYER_CONFIG_NAME,
    NORMALIZE_CONDITION_VECTORS,
    NORMALIZE_BEHAVIOR_VECTORS,
    RUN_VECTORS_DIR,
    VECTORS_DIR
)

BasePolicy: TypeAlias = _base_policy.BasePolicy


class Policy(BasePolicy):
    def __init__(
        self,
        model: _model.BaseModel,
        *,
        rng: at.KeyArrayLike | None = None,
        transforms: Sequence[_transforms.DataTransformFn] = (),
        output_transforms: Sequence[_transforms.DataTransformFn] = (),
        sample_kwargs: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        pytorch_device: str = "cpu",
        is_pytorch: bool = False,
    ):
        """Initialize the Policy.

        Args:
            model: The model to use for action sampling.
            rng: Random number generator key for JAX models. Ignored for PyTorch models.
            transforms: Input data transformations to apply before inference.
            output_transforms: Output data transformations to apply after inference.
            sample_kwargs: Additional keyword arguments to pass to model.sample_actions.
            metadata: Additional metadata to store with the policy.
            pytorch_device: Device to use for PyTorch models (e.g., "cpu", "cuda:0").
                          Only relevant when is_pytorch=True.
            is_pytorch: Whether the model is a PyTorch model. If False, assumes JAX model.
        """
        self._model = model
        self._input_transform = _transforms.compose(transforms)
        self._output_transform = _transforms.compose(output_transforms)
        self._sample_kwargs = sample_kwargs or {}
        self._metadata = metadata or {}
        self._is_pytorch_model = is_pytorch
        self._pytorch_device = pytorch_device
        torch.backends.cudnn.enabled = False

        if self._is_pytorch_model:
            self._model = self._model.to(pytorch_device)
            self._model.eval()
            self._sample_actions = model.sample_actions

            # Set seeds for reproducibility in flow-matching action expert
            if DEPLOYMENT_SEED is not None:
                torch.manual_seed(DEPLOYMENT_SEED)
                torch.cuda.manual_seed_all(DEPLOYMENT_SEED)
                np.random.seed(DEPLOYMENT_SEED)
                # Enable deterministic algorithms where possible
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False
                print(f"Deployment seed set to {DEPLOYMENT_SEED} for reproducibility")
        else:
            # JAX model setup
            self._sample_actions = nnx_utils.module_jit(model.sample_actions)
            self._rng = rng or jax.random.key(0)

        # if ACTIVATION_ENGINEERING:
        #     self.tokenizer = AutoTokenizer.from_pretrained("google/paligemma-3b-pt-224")
        #     positive_ex_padded, negative_ex_padded = pad_tokens(
        #         tokenizer=self.tokenizer,
        #         prompt_add_raw=POSITIVE_EXAMPLE, 
        #         prompt_sub_raw=NEGATIVE_EXAMPLE
        #     )
            
        #     # Compute steering vectors for each layer
        #     steering_vecs = {}
        #     for layer_idx in LAYER_INDICES:
        #         if USE_GEMMA_BACKBONE:
        #             positive_vec = get_text_based_hidden_states_from_vla(
        #                 model=self._model,
        #                 text=positive_ex_padded,
        #                 layer_idx=layer_idx,
        #                 tokenizer=self.tokenizer
        #             )
        #             negative_vec = get_text_based_hidden_states_from_vla(
        #                 model=self._model,
        #                 text=negative_ex_padded,
        #                 layer_idx=layer_idx,
        #                 tokenizer=self.tokenizer
        #             )
        #         else:
        #             positive_vec = get_text_based_hidden_states(
        #                 model=self._model,
        #                 text=positive_ex_padded,
        #                 layer_idx=layer_idx,
        #                 tokenizer=self.tokenizer
        #             )
        #             negative_vec = get_text_based_hidden_states(
        #                 model=self._model,
        #                 text=negative_ex_padded,
        #                 layer_idx=layer_idx,
        #                 tokenizer=self.tokenizer
        #             )
        #         steering_vecs[layer_idx] = positive_vec - negative_vec
            
        #     self.steering_hook = MultiLayerSteeringHook(
        #         model=self._model,
        #         steering_vecs=steering_vecs,
        #         alpha=ALPHA  # Same alpha for all, or pass a dict for per-layer alphas
        #     )

        if CAST:
            # CAST: Conditional Activation Steering (config-based multi-layer version)
            # Loads condition and behavior concept coefficients from YAML config
            # Requires pre-computed vectors from scripts/precompute_vectors.py
            self.tokenizer = AutoTokenizer.from_pretrained("google/paligemma-3b-pt-224")

            # Load the layer CAST config
            layer_config = load_layer_cast_config(
                config_path=str(CAST_LAYER_CONFIG_PATH),
                config_name=CAST_LAYER_CONFIG_NAME
            )

            condition_combination = layer_config.get('condition', {})
            behavior_combination = layer_config.get('behaviors', {})

            print(f"CAST config '{CAST_LAYER_CONFIG_NAME}':")
            print(f"  Condition: {condition_combination}")
            print(f"  Behavior: {behavior_combination}")

            # Load pre-computed vectors (required)
            if not vectors_exist(RUN_VECTORS_DIR):
                raise FileNotFoundError(
                    f"Pre-computed vectors not found at {RUN_VECTORS_DIR}. "
                    f"Run scripts/precompute_vectors.py first to compute vectors."
                )

            print("Loading pre-computed concept vectors from disk...")
            condition_vecs, behavior_vecs = load_and_combine_precomputed_vectors(
                condition_combination=condition_combination,
                behavior_combination=behavior_combination,
                layer_indices=VTI_LANGUAGE_LAYER_INDICES,
                normalize_condition=NORMALIZE_CONDITION_VECTORS,
                normalize_behavior=NORMALIZE_BEHAVIOR_VECTORS,
                device=self._pytorch_device,
                vectors_dir=RUN_VECTORS_DIR
            )

            # Check if we have any vectors to work with
            if not condition_vecs and not behavior_vecs:
                print("CAST: No vectors computed (all concepts are zero in config)")
                self.cast_hook = None
            else:
                # Create timestamped log directory for this run under logs/cast/
                if CAST_LOG_PATH:
                    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                    log_dir = pathlib.Path("logs") / "cast" / timestamp
                    log_dir.mkdir(parents=True, exist_ok=True)
                    print(f"CAST: Logging to {log_dir}")
                else:
                    log_dir = None

                # Create CAST hook with per-layer condition and behavior vectors
                self.cast_hook = CASTMultiLayerHook(
                    model=self._model,
                    behavior_vecs=behavior_vecs,
                    condition_vecs=condition_vecs,
                    alpha=ALPHA,
                    threshold=CAST_THRESHOLD,
                    use_tanh=True,
                    apply_steering=True,  # Set to False for detection-only mode
                    log_dir=str(log_dir) if log_dir else None,
                    config_name=CAST_LAYER_CONFIG_NAME
                )

        elif PHASE_CAST:
            # Multi-phase CAST is not yet supported with precomputed vectors
            # Use CAST=True with a single-phase config instead
            raise NotImplementedError(
                "PHASE_CAST is not yet supported with precomputed vectors. "
                "Use CAST=True with a single-phase config instead, or set PHASE_CAST=False."
            )

        elif VTI:
            # VTI: Visual and Textual Intervention for hallucination reduction
            # Based on: "Reducing Hallucinations in Large Vision-Language Models via Latent Space Steering"
            # Applies unconditional steering to both language model and vision encoder residual streams

            # Determine VTI vectors directory (uses same structure as CAST but under VTI/)
            # Default to 70 demos if not specified otherwise
            vti_num_demos = 70  # Default number of demos used for VTI vector computation
            vti_vectors_dir = VECTORS_DIR / "VTI" / str(vti_num_demos)

            # Check if VTI vectors exist
            vti_metadata = get_vti_metadata(vti_vectors_dir)
            if vti_metadata is None:
                raise FileNotFoundError(
                    f"VTI vectors not found at {vti_vectors_dir}. "
                    f"Run scripts/precompute_vectors_VTI.py first to compute vectors."
                )

            print(f"VTI: Loading pre-computed vectors from {vti_vectors_dir}")
            print(f"  Computed at: {vti_metadata.get('computed_at', 'unknown')}")
            print(f"  Num demos: {vti_metadata.get('num_demos', 'unknown')}")

            # Load VTI vectors for specified layers
            textual_vecs, visual_vecs = load_vti_vectors(
                layer_indices_text=VTI_LANGUAGE_LAYER_INDICES,
                layer_indices_vision=VTI_IMAGE_LAYER_INDICES,
                vectors_dir=vti_vectors_dir,
                device=self._pytorch_device
            )

            # Get alpha values from metadata (or use defaults)
            alpha_text = vti_metadata.get('alpha_text', 1.0)
            alpha_vision = vti_metadata.get('alpha_image', 1.0)

            print(f"VTI: Creating hook with alpha_text={alpha_text}, alpha_vision={alpha_vision}")

            # Create VTI hook
            self.vti_hook = VTIHook(
                model=self._model,
                textual_vecs=textual_vecs,
                visual_vecs=visual_vecs,
                alpha_text=alpha_text,
                alpha_vision=alpha_vision,
                normalize_text=False,
                normalize_vision=False
            )

    @override
    def infer(self, obs: dict, *, noise: np.ndarray | None = None) -> dict:  # type: ignore[misc]
        # Make a copy since transformations may modify the inputs in place.
        inputs = jax.tree.map(lambda x: x, obs)
        inputs = self._input_transform(inputs)
        if not self._is_pytorch_model:
            # Make a batch and convert to jax.Array.
            inputs = jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], inputs)
            self._rng, sample_rng_or_pytorch_device = jax.random.split(self._rng)
        else:
            # Convert inputs to PyTorch tensors and move to correct device
            inputs = jax.tree.map(lambda x: torch.from_numpy(np.array(x)).to(self._pytorch_device)[None, ...], inputs)
            sample_rng_or_pytorch_device = self._pytorch_device

        # Prepare kwargs for sample_actions
        sample_kwargs = dict(self._sample_kwargs)
        if noise is not None:
            noise = torch.from_numpy(noise).to(self._pytorch_device) if self._is_pytorch_model else jnp.asarray(noise)

            if noise.ndim == 2:  # If noise is (action_horizon, action_dim), add batch dimension
                noise = noise[None, ...]  # Make it (1, action_horizon, action_dim)
            sample_kwargs["noise"] = noise

        observation = _model.Observation.from_dict(inputs)
        start_time = time.monotonic()

        outputs = {
            "state": inputs["state"],
            "actions": self._sample_actions(sample_rng_or_pytorch_device, observation, **sample_kwargs),
        }
        model_time = time.monotonic() - start_time
        if self._is_pytorch_model:
            outputs = jax.tree.map(lambda x: np.asarray(x[0, ...].detach().cpu()), outputs)
        else:
            outputs = jax.tree.map(lambda x: np.asarray(x[0, ...]), outputs)

        outputs = self._output_transform(outputs)
        outputs["policy_timing"] = {
            "infer_ms": model_time * 1000,
        }
        return outputs

    @property
    def metadata(self) -> dict[str, Any]:
        return self._metadata


class PolicyRecorder(_base_policy.BasePolicy):
    """Records the policy's behavior to disk."""

    def __init__(self, policy: _base_policy.BasePolicy, record_dir: str):
        self._policy = policy

        logging.info(f"Dumping policy records to: {record_dir}")
        self._record_dir = pathlib.Path(record_dir)
        self._record_dir.mkdir(parents=True, exist_ok=True)
        self._record_step = 0

    @override
    def infer(self, obs: dict) -> dict:  # type: ignore[misc]
        results = self._policy.infer(obs)

        data = {"inputs": obs, "outputs": results}
        data = flax.traverse_util.flatten_dict(data, sep="/")

        output_path = self._record_dir / f"step_{self._record_step}"
        self._record_step += 1

        np.save(output_path, np.asarray(data))
        return results
