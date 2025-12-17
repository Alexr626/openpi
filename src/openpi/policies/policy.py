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
    MultiPhaseCASTHook,
    compute_behavior_vector_cast,
    compute_condition_vector_cast,
    compute_phase_vectors,
    load_layer_cast_config,
    compute_layer_cast_vectors,
)
from constants import (
    ACTIVATION_ENGINEERING,
    CAST,
    PHASE_CAST,
    USE_GEMMA_BACKBONE,
    LAYER_INDICES,
    ALPHA,
    CAST_THRESHOLD,
    CAST_LOG_PATH,
    # Layer-based CAST constants
    CAST_LAYER_CONFIG_PATH,
    CAST_LAYER_CONFIG_NAME,
    CAST_EXAMPLES_PATH,
    # Multi-phase CAST constants
    CAST_PHASE_CONFIG_NAME,
    CONDITION_CONCEPTS,
    BEHAVIOR_CONCEPTS,
    MULTI_PHASE_CAST_APPLY_STEERING,
    PHASE_CONDITIONS,
    PHASE_BEHAVIORS,
    PHASE_THRESHOLDS,
    PHASE_ALPHAS,
    NORMALIZE_CONDITION_VECTORS,
    NORMALIZE_BEHAVIOR_VECTORS
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

        if self._is_pytorch_model:
            self._model = self._model.to(pytorch_device)
            self._model.eval()
            self._sample_actions = model.sample_actions
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

        # elif CAST:
        if CAST:
            # CAST: Conditional Activation Steering (config-based multi-layer version)
            # Loads condition and behavior concept coefficients from YAML config
            # Only computes vectors for concepts with non-zero coefficients
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

            # Compute combined vectors for each layer based on config
            condition_vecs, behavior_vecs = compute_layer_cast_vectors(
                model=self._model,
                yaml_examples_path=CAST_EXAMPLES_PATH,
                layer_indices=LAYER_INDICES,
                tokenizer=self.tokenizer,
                condition_combination=condition_combination,
                behavior_combination=behavior_combination,
                example_type='robotic_specific',
                max_examples=100,
                normalize_condition=NORMALIZE_CONDITION_VECTORS,
                normalize_behavior=NORMALIZE_BEHAVIOR_VECTORS
            )

            # Check if we have any vectors to work with
            if not condition_vecs or not behavior_vecs:
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
            # Determine enabled phases from config: a phase is enabled if ANY concept
            # in its conditions OR behaviors is non-zero
            def _is_phase_enabled(phase_name: str) -> bool:
                """Check if a phase has any non-zero concepts in conditions or behaviors."""
                cond_values = PHASE_CONDITIONS.get(phase_name, {}).values()
                behav_values = PHASE_BEHAVIORS.get(phase_name, {}).values()
                return any(v != 0 for v in cond_values) or any(v != 0 for v in behav_values)

            all_phases = ["approach", "pregrasp_hover", "transport", "preplace_hover", "release", "retract"]
            enabled_phases = [phase for phase in all_phases if _is_phase_enabled(phase)]

            if enabled_phases:
                # Multi-phase CAST mode
                print(f"Multi-Phase CAST: Enabled phases: {enabled_phases}")
                self.tokenizer = AutoTokenizer.from_pretrained("google/paligemma-3b-pt-224")

                # Compute phase vectors from YAML examples
                phase_condition_vecs = {}
                phase_behavior_vecs = {}
                for layer_idx in LAYER_INDICES:
                    phase_condition_vecs[layer_idx], phase_behavior_vecs[layer_idx] = compute_phase_vectors(
                        model=self._model,
                        yaml_path=CAST_EXAMPLES_PATH,
                        layer_idx=layer_idx,
                        tokenizer=self.tokenizer,
                        phase_conditions=PHASE_CONDITIONS,
                        phase_behaviors=PHASE_BEHAVIORS,
                        condition_concepts=CONDITION_CONCEPTS,
                        behavior_concepts=BEHAVIOR_CONCEPTS,
                        enabled_phases=enabled_phases,
                        example_type='robotic_specific',
                        max_examples=100,
                        normalize_condition_vectors=NORMALIZE_CONDITION_VECTORS,
                        normalize_behavior_vectors=NORMALIZE_BEHAVIOR_VECTORS
                    )

                # Create timestamped log directory for this run under logs/phase/
                if CAST_LOG_PATH:
                    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                    log_dir = pathlib.Path("logs") / "phase" / timestamp
                    log_dir.mkdir(parents=True, exist_ok=True)
                    print(f"Multi-Phase CAST: Logging to {log_dir}")
                else:
                    log_dir = None

                # Create multi-phase hooks (one per layer)
                self.multi_phase_hooks = []
                for layer_idx in LAYER_INDICES:
                    # Generate layer-specific log path
                    if log_dir:
                        log_path = str(log_dir / f"cast_similarity_layer{layer_idx}.json")
                    else:
                        log_path = None

                    hook = MultiPhaseCASTHook(
                        model=self._model,
                        layer_idx=layer_idx,
                        phase_condition_vecs=phase_condition_vecs[layer_idx],
                        phase_behavior_vecs=phase_behavior_vecs[layer_idx],
                        phase_thresholds=PHASE_THRESHOLDS,
                        phase_alphas=PHASE_ALPHAS,
                        use_tanh=True,
                        apply_steering=MULTI_PHASE_CAST_APPLY_STEERING,
                        log_path=log_path,
                        config_name=CAST_PHASE_CONFIG_NAME,
                        config_conditions=PHASE_CONDITIONS,
                        config_behaviors=PHASE_BEHAVIORS
                    )
                    self.multi_phase_hooks.append(hook)
            else:
                self.steering_hook = None
                self.cast_hook = None
                self.multi_phase_hooks = []

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
