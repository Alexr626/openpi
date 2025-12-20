"""
Conditional Activation Steering (CAST) Implementation for Pi 0.5 VLA

Based on: "PROGRAMMING REFUSAL WITH CONDITIONAL ACTIVATION STEERING" by Lee et al. (2025)

CAST extends Activation Engineering by adding a condition vector that determines
when to apply the behavior steering. The modified hidden state is computed as:
    h' ← h + f(sim(h, proj_c h)) · α · v

Where:
    - h: original hidden state
    - c: condition vector
    - v: behavior vector
    - α: scaling factor
    - f: thresholding function (outputs 1 if similarity > θ, else 0)
    - proj_c h = (c⊗c / c·c) h: projection of h onto c
    - sim(h, g) = h·g / (|h||g|): cosine similarity

This module contains:
- Runtime functions for applying CAST during inference (hooks, condition checking, steering)
- Functions for loading pre-computed vectors from disk
- YAML/config loading utilities

Vector computation functions are in scripts/precompute_vectors.py
"""

import os
import torch
import torch.nn.functional as F
from torch import Tensor
import json
import time
import yaml
from pathlib import Path
from typing import Optional, Union, List, Tuple, Dict
import numpy as np
from openpi.models_pytorch.pi0_pytorch import PI0Pytorch

# Shared file for cross-process coordination of deploy session path
# This file is written by deploy.py and read by the policy server
DEPLOY_SESSION_FILE = '/tmp/deploy_session_path.txt'


# =============================================================================
# YAML EXAMPLES LOADING
# =============================================================================

def load_contrasting_examples(yaml_path: str) -> Dict:
    """
    Load contrasting examples from a YAML file.

    Args:
        yaml_path: Path to the YAML file containing positive/negative examples

    Returns:
        Dictionary with structure:
        {
            'positive_examples': {category: {concept: {'examples': [...], 'category': ...}}},
            'negative_examples': {category: {concept: {'examples': [...], 'category': ...}}},
            'metadata': {...}
        }
    """
    yaml_path = Path(yaml_path)
    if not yaml_path.exists():
        raise FileNotFoundError(f"Examples file not found: {yaml_path}")

    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)

    return data


def get_examples_for_concept(
    yaml_data: Dict,
    concept: str,
    example_type: str = 'general_physical',
    max_examples: Optional[int] = None
) -> Tuple[List[str], List[str]]:
    """
    Get positive and negative examples for a specific concept.

    Args:
        yaml_data: Loaded YAML data
        concept: Concept name (e.g., 'Holding', 'Target', 'Ascent')
        example_type: 'robotic_specific' or 'general_physical'
        max_examples: Maximum number of examples to return (None for all)

    Returns:
        Tuple of (positive_examples, negative_examples)
    """
    positive_examples = yaml_data.get('positive_examples', {})
    positive_examples = positive_examples.get(example_type, {})
    positive_examples = positive_examples.get(concept, {})
    positive_examples = positive_examples.get('examples', [])

    negative_examples = yaml_data.get('negative_examples', {})
    negative_examples = negative_examples.get(example_type, {})
    negative_examples = negative_examples.get(concept, {})
    negative_examples = negative_examples.get('examples', [])

    # Try to get concept-specific negative examples
    neg_concept_data = yaml_data.get('negative_examples', {}).get(example_type, {}).get(concept, {})
    if isinstance(neg_concept_data, dict) and 'examples' in neg_concept_data:
        negative_examples = neg_concept_data['examples']

    if max_examples:
        positive_examples = positive_examples[:max_examples]
        negative_examples = negative_examples[:max_examples]

    return positive_examples, negative_examples


# =============================================================================
# VECTOR COMBINATION
# =============================================================================

def combine_concept_vectors(
    concept_vectors: Dict[str, Tensor],
    combination: Dict[str, int],
    normalize: bool = False
) -> Tensor:
    """
    Combine multiple concept vectors according to a combination specification.

    Args:
        concept_vectors: Dictionary mapping concept name -> vector
        combination: Dictionary mapping concept name -> sign (1, -1, or 0)
                    1 = add vector, -1 = subtract vector, 0 = don't include
        normalize: Whether to normalize the combined vector

    Returns:
        Combined vector [hidden_dim]
    """
    combined = None
    device = None
    dtype = None

    for concept, sign in combination.items():
        if sign == 0:
            continue
        if concept not in concept_vectors:
            print(f"Warning: Concept '{concept}' not found in computed vectors, skipping")
            continue

        vec = concept_vectors[concept]
        if combined is None:
            device = vec.device
            dtype = vec.dtype
            combined = torch.zeros_like(vec)

        combined = combined + sign * vec

    if combined is None:
        # All concepts were 0 or missing - return None to indicate no vector
        return None

    if normalize:
        norm = torch.norm(combined)
        if norm > 1e-8:
            combined = combined / norm

    return combined


# =============================================================================
# LAYER-BASED CAST CONFIG LOADING
# =============================================================================

def load_layer_cast_config(config_path: str, config_name: str) -> Dict:
    """
    Load a layer-based CAST configuration from YAML file.

    Args:
        config_path: Path to the YAML config file
        config_name: Name of the configuration to load

    Returns:
        Dictionary with 'condition' and 'behaviors' keys mapping concept -> coefficient
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"CAST layer config file not found: {config_path}")

    with open(config_path, 'r') as f:
        all_configs = yaml.safe_load(f)

    if config_name not in all_configs:
        available = list(all_configs.keys())
        raise ValueError(f"Config '{config_name}' not found. Available: {available}")

    config = all_configs[config_name]
    print(f"Loaded CAST layer config: '{config_name}'")

    return config


# =============================================================================
# PRE-COMPUTED VECTOR LOADING
# =============================================================================

def load_precomputed_concept_vectors(
    concepts: List[str],
    layer_indices: List[int],
    vectors_dir: Path
) -> Dict[int, Dict[str, Tensor]]:
    """
    Load pre-computed concept vectors from disk.

    Args:
        concepts: List of concept names to load
        layer_indices: List of layer indices to load vectors for
        vectors_dir: Directory containing saved vectors

    Returns:
        Dict mapping layer_idx -> {concept_name -> vector}
    """
    vectors_dir = Path(vectors_dir)

    if not vectors_dir.exists():
        raise FileNotFoundError(f"Vectors directory not found: {vectors_dir}")

    result = {}
    missing = []

    for layer_idx in layer_indices:
        result[layer_idx] = {}

        for concept in concepts:
            filename = f"{concept}_layer{layer_idx}.pt"
            filepath = vectors_dir / filename

            if filepath.exists():
                result[layer_idx][concept] = torch.load(filepath, weights_only=True)
            else:
                missing.append(filename)

    if missing:
        raise FileNotFoundError(
            f"Missing pre-computed vectors: {missing}. "
            f"Run scripts/precompute_vectors.py first."
        )

    print(f"Loaded {len(concepts) * len(layer_indices)} pre-computed vectors from {vectors_dir}")
    return result


def load_and_combine_precomputed_vectors(
    condition_combination: Dict[str, int],
    behavior_combination: Dict[str, int],
    layer_indices: List[int],
    vectors_dir: Optional[Path] = None,
    normalize_condition: bool = False,
    normalize_behavior: bool = False,
    device: Optional[str] = None
) -> Tuple[Dict[int, Tensor], Dict[int, Tensor]]:
    """
    Load pre-computed concept vectors and combine them based on config coefficients.

    This is the main function to use in deployment - it loads pre-computed vectors
    and combines them according to the config's coefficients.

    Args:
        condition_combination: Dict mapping condition concept -> coefficient (-1, 0, 1)
        behavior_combination: Dict mapping behavior concept -> coefficient (-1, 0, 1)
        layer_indices: List of layer indices
        vectors_dir: Directory containing saved vectors
        normalize_condition: Whether to normalize combined condition vectors
        normalize_behavior: Whether to normalize combined behavior vectors
        device: Device to move vectors to (optional)

    Returns:
        Tuple of (condition_vecs, behavior_vecs)
        Each is a dict mapping layer_idx -> combined vector
    """
    # Determine which concepts need to be loaded (non-zero coefficients)
    used_condition_concepts = [c for c, coef in condition_combination.items() if coef != 0]
    used_behavior_concepts = [c for c, coef in behavior_combination.items() if coef != 0]
    all_used_concepts = list(set(used_condition_concepts + used_behavior_concepts))

    if not all_used_concepts:
        print("Warning: No concepts have non-zero coefficients, returning empty vectors")
        return {}, {}

    print(f"Loading pre-computed vectors for {len(all_used_concepts)} concepts")
    print(f"  Condition concepts: {used_condition_concepts}")
    print(f"  Behavior concepts: {used_behavior_concepts}")

    # Load the needed vectors
    concept_vectors_by_layer = load_precomputed_concept_vectors(
        concepts=all_used_concepts,
        layer_indices=layer_indices,
        vectors_dir=vectors_dir
    )

    # Combine vectors for each layer
    condition_vecs = {}
    behavior_vecs = {}

    for layer_idx in layer_indices:
        concept_vectors = concept_vectors_by_layer[layer_idx]

        # Combine condition vectors
        if used_condition_concepts:
            cond_vec = combine_concept_vectors(
                concept_vectors,
                condition_combination,
                normalize=normalize_condition
            )
            if cond_vec is not None:
                if device:
                    cond_vec = cond_vec.to(device)
                condition_vecs[layer_idx] = cond_vec

        # Combine behavior vectors
        if used_behavior_concepts:
            behav_vec = combine_concept_vectors(
                concept_vectors,
                behavior_combination,
                normalize=normalize_behavior
            )
            if behav_vec is not None:
                if device:
                    behav_vec = behav_vec.to(device)
                behavior_vecs[layer_idx] = behav_vec

    print(f"Combined vectors for {len(condition_vecs)} layers")
    return condition_vecs, behavior_vecs


def get_vectors_metadata(vectors_dir: Optional[Path] = None) -> Optional[Dict]:
    """
    Load metadata about pre-computed vectors.

    Args:
        vectors_dir: Directory containing saved vectors

    Returns:
        Metadata dict or None if not found
    """
    metadata_path = vectors_dir / "metadata.json"

    if not metadata_path.exists():
        return None

    with open(metadata_path, 'r') as f:
        return json.load(f)


def vectors_exist(vectors_dir) -> bool:
    """Check if pre-computed vectors exist."""
    metadata_path = Path(vectors_dir) / "metadata.json"
    return metadata_path.exists()


# =============================================================================
# CORE CAST OPERATIONS
# =============================================================================

def project_onto_condition_vector(
    hidden_state: Tensor,
    condition_vec: Tensor,
    use_tanh: bool = True
) -> Tensor:
    """
    Project hidden state onto condition vector: proj_c(h) = (c⊗c / c·c) h

    This computes how aligned the hidden state is with the condition direction.

    Args:
        hidden_state: Hidden state [batch, seq_len, hidden_dim] or [hidden_dim]
        condition_vec: Condition vector [hidden_dim]
        use_tanh: Apply tanh non-linearity for stability (as in paper)

    Returns:
        Projection [same shape as hidden_state]
    """
    # Ensure condition_vec is 1D
    c = condition_vec.view(-1)

    # Compute c·c (squared norm of condition vector)
    c_dot_c = torch.dot(c, c)

    if c_dot_c < 1e-8:
        raise ValueError("Condition vector has near-zero norm")

    original_shape = hidden_state.shape
    h_flat = hidden_state.view(-1, hidden_state.shape[-1])  # [N, hidden_dim]

    # Compute h·c for each position
    h_dot_c = h_flat @ c  # [N]

    # Compute projection coefficient: (h·c) / (c·c)
    proj_coeff = h_dot_c / c_dot_c  # [N]

    # Compute projection: coeff * c
    projection = proj_coeff.unsqueeze(-1) * c.unsqueeze(0)  # [N, hidden_dim]

    if use_tanh:
        projection = torch.tanh(projection)

    return projection.view(original_shape)


def compute_cosine_similarity(
    hidden_state: Tensor,
    projection: Tensor
) -> Tensor:
    """
    Compute cosine similarity: sim(h, proj_c(h)) = h·g / (|h||g|)

    Args:
        hidden_state: Hidden state [batch, seq_len, hidden_dim] or [hidden_dim]
        projection: Projection [same shape as hidden_state]

    Returns:
        Similarity scalar or tensor [batch, seq_len] if input was batched
    """
    original_shape = hidden_state.shape[:-1]

    h_flat = hidden_state.view(-1, hidden_state.shape[-1])
    p_flat = projection.view(-1, projection.shape[-1])

    similarity = F.cosine_similarity(h_flat, p_flat, dim=-1)

    if len(original_shape) > 0:
        return similarity.view(original_shape)
    return similarity


def check_condition(
    hidden_state: Tensor,
    condition_vec: Tensor,
    threshold: float = 0.5,
    use_tanh: bool = True,
    average_over_tokens: bool = True
) -> Tuple[bool, float]:
    """
    Check if the condition is met for applying behavior steering.

    This is done once during the prompt processing phase.

    Args:
        hidden_state: Hidden state [batch, seq_len, hidden_dim]
        condition_vec: Condition vector [hidden_dim]
        threshold: Similarity threshold θ
        use_tanh: Apply tanh to projection
        average_over_tokens: If True, average hidden states before checking

    Returns:
        Tuple of (condition_triggered: bool, similarity_value: float)
    """
    if average_over_tokens:
        # Average across tokens to get [batch, hidden_dim] or [hidden_dim]
        h_avg = hidden_state.mean(dim=-2)
    else:
        h_avg = hidden_state

    # Project onto condition vector
    projection = project_onto_condition_vector(h_avg, condition_vec, use_tanh)

    # Compute similarity
    similarity = compute_cosine_similarity(h_avg, projection)

    # Average similarity if batched
    if similarity.dim() > 0:
        similarity_scalar = similarity.mean().item()
    else:
        similarity_scalar = similarity.item()

    # Apply threshold
    condition_triggered = similarity_scalar > threshold

    return condition_triggered, similarity_scalar


def apply_behavior_steering(
    hidden_state: Tensor,
    behavior_vec: Tensor,
    alpha: float
) -> Tensor:
    """
    Apply behavior vector to hidden state: h' = h + α · v

    Called when condition is triggered.

    Args:
        hidden_state: Hidden state [batch, seq_len, hidden_dim]
        behavior_vec: Behavior vector [hidden_dim]
        alpha: Scaling factor

    Returns:
        Modified hidden state [batch, seq_len, hidden_dim]
    """
    # Broadcast behavior vector to match hidden state shape
    v = behavior_vec.view(1, 1, -1)  # [1, 1, hidden_dim]

    return hidden_state + alpha * v


def apply_cast_steering(
    hidden_state: Tensor,
    condition_vec: Tensor,
    behavior_vec: Tensor,
    alpha: float,
    threshold: float = 0.5,
    use_tanh: bool = True
) -> Tuple[Tensor, bool, float]:
    """
    Full CAST steering: h' ← h + f(sim(h, proj_c h)) · α · v

    This combines condition checking and behavior application.

    Args:
        hidden_state: Hidden state [batch, seq_len, hidden_dim]
        condition_vec: Condition vector [hidden_dim]
        behavior_vec: Behavior vector [hidden_dim]
        alpha: Scaling factor
        threshold: Similarity threshold θ
        use_tanh: Apply tanh to projection

    Returns:
        Tuple of (modified_hidden_state, condition_triggered, similarity_value)
    """
    # Check condition
    triggered, similarity = check_condition(
        hidden_state, condition_vec, threshold, use_tanh
    )

    if triggered:
        modified = apply_behavior_steering(hidden_state, behavior_vec, alpha)
    else:
        modified = hidden_state

    return modified, triggered, similarity


# =============================================================================
# CAST MULTI-LAYER HOOK
# =============================================================================

class CASTMultiLayerHook:
    """
    Multi-layer CAST that applies condition checking and/or behavior steering
    at multiple layers. The mode is automatically inferred from which vectors
    are provided:

    - Both behavior_vecs and condition_vecs provided: Full CAST with conditional steering
    - Only condition_vecs provided: Condition-only mode (compute/log similarity, no steering)
    - Only behavior_vecs provided: Steering-only mode (always apply steering unconditionally)

    Both condition and behavior vectors are computed per-layer from contrasting examples.
    """

    def __init__(
        self,
        model: PI0Pytorch,
        behavior_vecs: Optional[Dict[int, Tensor]] = None,
        condition_vecs: Optional[Dict[int, Tensor]] = None,
        alpha: Union[float, Dict[int, float]] = 1.0,
        threshold: Union[float, Dict[int, float]] = 0.5,
        use_tanh: bool = True,
        apply_steering: bool = True,
        log_dir: Optional[str] = None,
        config_name: Optional[str] = None
    ):
        """
        Args:
            model: PI0Pytorch model
            behavior_vecs: Dict mapping layer_idx -> behavior vector.
                          If provided alone, enables steering-only mode.
                          If provided with condition_vecs, enables full CAST.
            condition_vecs: Dict mapping layer_idx -> condition vector.
                           If provided alone, enables condition-only mode.
                           If provided with behavior_vecs, enables full CAST.
            alpha: Scaling factor (single value or per-layer dict)
            threshold: Similarity threshold θ (single value or per-layer dict)
            use_tanh: Apply tanh to projection
            apply_steering: If True, apply behavior steering when condition triggered (for full CAST mode).
                           If False, only detect and log (useful for tuning thresholds).
                           Ignored in condition-only mode (never steers) and steering-only mode (always steers).
            log_dir: Optional directory to save per-layer similarity logs.
            config_name: Name of the configuration being used (for logging).
        """
        self.model = model
        self.device = model.paligemma_with_expert.paligemma.device

        # Infer mode from which vectors are provided
        has_behavior = behavior_vecs is not None and len(behavior_vecs) > 0
        has_condition = condition_vecs is not None and len(condition_vecs) > 0

        if has_behavior and has_condition:
            self.mode = 'both'
            # Validate that both have same layer indices
            if set(behavior_vecs.keys()) != set(condition_vecs.keys()):
                raise ValueError(
                    f"When both behavior_vecs and condition_vecs are provided, they must have same layer indices. "
                    f"Got behavior: {set(behavior_vecs.keys())}, condition: {set(condition_vecs.keys())}"
                )
            self.layer_indices = sorted(behavior_vecs.keys())
        elif has_condition:
            self.mode = 'condition_only'
            self.layer_indices = sorted(condition_vecs.keys())
        elif has_behavior:
            self.mode = 'steering_only'
            self.layer_indices = sorted(behavior_vecs.keys())
        else:
            raise ValueError("At least one of behavior_vecs or condition_vecs must be provided")

        # Store vectors (may be empty for some modes)
        self.behavior_vecs = {}
        self.condition_vecs = {}

        if has_behavior:
            self.behavior_vecs = {idx: vec.to(self.device) for idx, vec in behavior_vecs.items()}
        if has_condition:
            self.condition_vecs = {idx: vec.to(self.device) for idx, vec in condition_vecs.items()}

        self.use_tanh = use_tanh
        self.apply_steering = apply_steering
        self.log_dir = log_dir
        self.config_name = config_name

        # Handle single value or per-layer alpha
        if isinstance(alpha, (int, float)):
            self.alphas = {idx: float(alpha) for idx in self.layer_indices}
        else:
            self.alphas = alpha

        # Handle single value or per-layer threshold
        if isinstance(threshold, (int, float)):
            self.thresholds = {idx: float(threshold) for idx in self.layer_indices}
        else:
            self.thresholds = threshold

        self.handles: Dict[int, any] = {}

        # Per-layer state tracking
        self.condition_triggered: Dict[int, bool] = {idx: False for idx in self.layer_indices}
        self.similarity_values: Dict[int, float] = {idx: 0.0 for idx in self.layer_indices}

        # Per-layer similarity logging
        self.similarity_history: Dict[int, List[Dict]] = {idx: [] for idx in self.layer_indices}
        self.timestep = 0

    def _make_hook_fn(self, layer_idx: int):
        """Create hook function for a specific layer based on current mode"""
        alpha = self.alphas.get(layer_idx, 1.0)
        threshold = self.thresholds.get(layer_idx, 0.5)

        # Get vectors if available
        condition_vec = self.condition_vecs.get(layer_idx)
        behavior_vec = self.behavior_vecs.get(layer_idx)

        def hook_fn(module, input, output):
            hidden = output[0] if isinstance(output, tuple) else output

            triggered = False
            similarity = 0.0

            # Condition checking (for 'both' and 'condition_only' modes)
            if self.mode in ('both', 'condition_only') and condition_vec is not None:
                triggered, similarity = check_condition(
                    hidden, condition_vec, threshold, self.use_tanh
                )

                self.condition_triggered[layer_idx] = triggered
                self.similarity_values[layer_idx] = similarity

                # Log similarity score
                log_entry = {
                    'timestep': self.timestep,
                    'layer_idx': layer_idx,
                    'similarity': similarity,
                    'triggered': triggered,
                    'threshold': threshold,
                    'timestamp': time.time()
                }
                self.similarity_history[layer_idx].append(log_entry)

                status = "TRIGGERED" if triggered else "not triggered"
                print(f"CAST [L{layer_idx}]: {status} (sim={similarity:.4f}, θ={threshold})")

            # Determine if we should apply steering
            should_steer = False
            if self.mode == 'steering_only':
                # Always steer in steering_only mode
                should_steer = True
            elif self.mode == 'both':
                # Steer if condition triggered and apply_steering is enabled
                should_steer = triggered and self.apply_steering
            # In 'condition_only' mode, never steer

            # Apply steering if appropriate
            if should_steer and behavior_vec is not None:
                modified = apply_behavior_steering(hidden, behavior_vec, alpha)

                if self.mode == 'steering_only':
                    print(f"CAST [L{layer_idx}]: Steering applied (α={alpha}) [unconditional]")
                else:
                    print(f"CAST [L{layer_idx}]: Steering applied (α={alpha})")

                if isinstance(output, tuple):
                    return (modified.to(hidden.dtype),) + output[1:]
                return modified.to(hidden.dtype)

            return output

        return hook_fn

    def register(self):
        """Register hooks at all layers"""
        for layer_idx in self.layer_indices:
            layer = self.model.paligemma_with_expert.paligemma.language_model.layers[layer_idx]
            self.handles[layer_idx] = layer.mlp.down_proj.register_forward_hook(
                self._make_hook_fn(layer_idx)
            )

        # Describe the mode
        if self.mode == 'condition_only':
            mode_desc = "CONDITION ONLY (no steering)"
        elif self.mode == 'steering_only':
            mode_desc = "STEERING ONLY (unconditional)"
        elif self.apply_steering:
            mode_desc = "FULL CAST (condition + steering)"
        else:
            mode_desc = "DETECTION ONLY (condition check, steering disabled)"

        print(f"CAST: Registered hooks at layers {self.layer_indices} - {mode_desc}")

    def remove(self):
        """Remove all hooks and save logs if log_dir was provided"""
        for handle in self.handles.values():
            handle.remove()
        self.handles.clear()

        # Auto-save similarity logs when removing hooks
        if self.log_dir:
            self.save_similarity_logs()

    def step(self):
        """Increment timestep counter (call after each forward pass)"""
        self.timestep += 1

    def reset(self):
        """Reset state for new sequence"""
        self.condition_triggered = {idx: False for idx in self.layer_indices}
        self.similarity_values = {idx: 0.0 for idx in self.layer_indices}

    def save_similarity_logs(self, log_dir: Optional[str] = None, cleanup: bool = True):
        """
        Save per-layer similarity history to JSON files.

        Args:
            log_dir: Directory to save logs. If None, reads from shared session file,
                     then falls back to self.log_dir.
            cleanup: If True, clear similarity history after saving (default: True).
                     This prevents duplicate saves if called multiple times.
        """
        # Priority: explicit arg > shared session file > self.log_dir
        save_dir = log_dir
        config_name_override = None  # Config name from deploy.py (if available)

        if not save_dir:
            # Check for shared session file written by deploy.py (JSON format)
            if os.path.exists(DEPLOY_SESSION_FILE):
                try:
                    with open(DEPLOY_SESSION_FILE, 'r') as f:
                        session_info = json.load(f)
                    session_dir = session_info.get('session_dir')
                    config_name_override = session_info.get('cast_config')
                    if session_dir:
                        # Use cast/ subdirectory within the deploy session directory
                        save_dir = os.path.join(session_dir, 'cast')
                        print(f"CAST: Using deploy session for logging: {save_dir}")
                    if config_name_override:
                        print(f"CAST: Using config name from deploy.py: {config_name_override}")
                except json.JSONDecodeError:
                    # Fallback: try reading as plain text (old format)
                    try:
                        with open(DEPLOY_SESSION_FILE, 'r') as f:
                            session_dir = f.read().strip()
                        if session_dir:
                            save_dir = os.path.join(session_dir, 'cast')
                            print(f"CAST: Using deploy session for logging (legacy format): {save_dir}")
                    except Exception as e:
                        print(f"CAST: Warning - could not read session file: {e}")
                except Exception as e:
                    print(f"CAST: Warning - could not read session file: {e}")

            if not save_dir:
                save_dir = self.log_dir

        if not save_dir:
            print("Warning: No log directory specified, cannot save similarity logs")
            return

        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        # Use config name from deploy.py if available, otherwise use self.config_name
        effective_config_name = config_name_override if config_name_override else self.config_name

        saved_count = 0
        for layer_idx in self.layer_indices:
            history = self.similarity_history[layer_idx]
            if not history:
                continue

            log_data = {
                'metadata': {
                    'config_name': effective_config_name,
                    'layer_idx': layer_idx,
                    'threshold': self.thresholds[layer_idx],
                    'alpha': self.alphas[layer_idx],
                    'apply_steering': self.apply_steering,
                    'total_timesteps': len(history)
                },
                'history': history
            }

            log_path = save_dir / f"cast_similarity_layer{layer_idx}.json"
            with open(log_path, 'w') as f:
                json.dump(log_data, f, indent=2)

            print(f"CAST [L{layer_idx}]: Saved log ({len(history)} timesteps) to {log_path}")
            saved_count += 1

        # Cleanup: clear history after saving to prevent duplicate saves
        if cleanup and saved_count > 0:
            self.clear_history()
            print(f"CAST: Cleared similarity history after saving {saved_count} layer logs")

    def clear_history(self):
        """Clear all similarity history and reset timestep counter"""
        for layer_idx in self.layer_indices:
            self.similarity_history[layer_idx].clear()
        self.timestep = 0

    def get_similarity_history(self, layer_idx: Optional[int] = None) -> Union[Dict[int, List[Dict]], List[Dict]]:
        """
        Return similarity history.

        Args:
            layer_idx: If provided, return history for that layer only.
                      If None, return dict of all layers.
        """
        if layer_idx is not None:
            return self.similarity_history[layer_idx].copy()
        return {idx: hist.copy() for idx, hist in self.similarity_history.items()}

    def get_trigger_summary(self) -> Dict[int, Dict]:
        """Get summary of trigger statistics per layer"""
        summary = {}
        for layer_idx in self.layer_indices:
            history = self.similarity_history[layer_idx]
            if not history:
                continue

            triggered_count = sum(1 for entry in history if entry['triggered'])
            total = len(history)
            similarities = [entry['similarity'] for entry in history]

            summary[layer_idx] = {
                'triggered_count': triggered_count,
                'total_timesteps': total,
                'trigger_rate': triggered_count / total if total > 0 else 0,
                'similarity_mean': np.mean(similarities) if similarities else 0,
                'similarity_std': np.std(similarities) if similarities else 0,
                'similarity_min': np.min(similarities) if similarities else 0,
                'similarity_max': np.max(similarities) if similarities else 0,
            }

        return summary


# =============================================================================
# MULTI-PHASE CAST HOOK
# =============================================================================

class MultiPhaseCASTHook:
    """
    CAST hook for a single layer that supports multiple phases.

    At each timestep, checks similarity against all phase conditions and
    applies the behavior steering for the phase with highest similarity
    (if above threshold).

    Create one instance per layer.
    """

    def __init__(
        self,
        model: PI0Pytorch,
        layer_idx: int,
        phase_condition_vecs: Dict[str, Tensor],
        phase_behavior_vecs: Dict[str, Tensor],
        phase_thresholds: Dict[str, float],
        phase_alphas: Dict[str, float],
        use_tanh: bool = True,
        apply_steering: bool = True,
        log_path: Optional[str] = None,
        config_name: Optional[str] = None,
        config_conditions: Optional[Dict[str, Dict[str, int]]] = None,
        config_behaviors: Optional[Dict[str, Dict[str, int]]] = None
    ):
        """
        Args:
            model: PI0Pytorch model
            layer_idx: The layer index to attach this hook to
            phase_condition_vecs: Dict mapping phase_name -> condition vector
            phase_behavior_vecs: Dict mapping phase_name -> behavior vector
            phase_thresholds: Dict mapping phase_name -> threshold
            phase_alphas: Dict mapping phase_name -> alpha
            use_tanh: Apply tanh to projections
            apply_steering: If True, apply behavior steering when phases detected.
                           If False, only detect and log phases (useful for tuning).
            log_path: Optional path to save similarity logs
            config_name: Name of the phase configuration being used
            config_conditions: Full phase conditions dict from config
            config_behaviors: Full phase behaviors dict from config
        """
        self.model = model
        self.layer_idx = layer_idx
        self.device = model.paligemma_with_expert.paligemma.device

        self.phase_condition_vecs = {
            name: vec.to(self.device) for name, vec in phase_condition_vecs.items()
        }
        self.phase_behavior_vecs = {
            name: vec.to(self.device) for name, vec in phase_behavior_vecs.items()
        }
        self.phases = list(phase_condition_vecs.keys())

        self.phase_thresholds = phase_thresholds
        self.phase_alphas = phase_alphas
        self.use_tanh = use_tanh
        self.apply_steering = apply_steering
        self.log_path = log_path

        # Config metadata for logging
        self.config_name = config_name
        self.config_conditions = config_conditions
        self.config_behaviors = config_behaviors

        self.handle = None

        # State
        self.current_phase: Optional[str] = None
        self.phase_similarities: Dict[str, float] = {}
        self.timestep = 0

        # Logging
        self.similarity_history: List[Dict] = []

    def _hook_fn(self, module, input, output):
        """Check conditions and apply steering in a single hook"""
        hidden = output[0] if isinstance(output, tuple) else output

        # Compute similarity for each phase
        self.phase_similarities = {}
        max_sim = -float('inf')
        best_phase = None

        for phase in self.phases:
            cond_vec = self.phase_condition_vecs[phase]
            threshold = self.phase_thresholds.get(phase, 0.5)

            triggered, similarity = check_condition(
                hidden, cond_vec, threshold, self.use_tanh
            )
            self.phase_similarities[phase] = similarity

            if triggered and similarity > max_sim:
                max_sim = similarity
                best_phase = phase

        self.current_phase = best_phase

        # Log
        log_entry = {
            'timestep': self.timestep,
            'layer_idx': self.layer_idx,
            'phase_similarities': dict(self.phase_similarities),
            'detected_phase': self.current_phase,
            'timestamp': time.time()
        }
        self.similarity_history.append(log_entry)
        self.timestep += 1

        # Print status
        sims_str = ", ".join([f"{p}={s:.4f}" for p, s in self.phase_similarities.items()])
        if self.current_phase:
            print(f"MultiPhase CAST [L{self.layer_idx}]: Detected '{self.current_phase}' [{sims_str}]")
        else:
            print(f"MultiPhase CAST [L{self.layer_idx}]: No phase detected [{sims_str}]")

        # Apply steering if enabled and a phase was detected
        if self.current_phase is None:
            return output

        if not self.apply_steering:
            return output

        behavior_vec = self.phase_behavior_vecs.get(self.current_phase)
        if behavior_vec is None:
            return output

        alpha = self.phase_alphas.get(self.current_phase, 1.0)

        # Apply steering
        modified = apply_behavior_steering(hidden, behavior_vec, alpha)

        print(f"MultiPhase CAST [L{self.layer_idx}]: Steering '{self.current_phase}' (alpha={alpha})")

        if isinstance(output, tuple):
            return (modified.to(hidden.dtype),) + output[1:]
        return modified.to(hidden.dtype)

    def register(self):
        """Register the hook"""
        layer = self.model.paligemma_with_expert.paligemma.language_model.layers[self.layer_idx]
        self.handle = layer.mlp.down_proj.register_forward_hook(self._hook_fn)

        mode = "STEERING ENABLED" if self.apply_steering else "DETECTION ONLY (steering disabled)"
        print(f"MultiPhase CAST: Registered hook at layer {self.layer_idx} - {mode}")
        print(f"  Phases: {self.phases}")

    def remove(self):
        """Remove the hook and save log"""
        if self.handle:
            self.handle.remove()
            self.handle = None

        if self.log_path and self.similarity_history:
            self.save_similarity_log()

    def save_similarity_log(self, path: Optional[str] = None):
        """Save similarity history to JSON"""
        save_path = path or self.log_path
        if not save_path:
            return

        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        log_data = {
            'metadata': {
                'config_name': self.config_name,
                'config_conditions': self.config_conditions,
                'config_behaviors': self.config_behaviors,
                'phases': self.phases,
                'layer_idx': self.layer_idx,
                'thresholds': self.phase_thresholds,
                'alphas': self.phase_alphas,
                'apply_steering': self.apply_steering,
                'total_timesteps': len(self.similarity_history)
            },
            'history': self.similarity_history
        }

        with open(save_path, 'w') as f:
            json.dump(log_data, f, indent=2)

        print(f"MultiPhase CAST [L{self.layer_idx}]: Saved log ({len(self.similarity_history)} timesteps) to {save_path}")

    def get_similarity_history(self) -> List[Dict]:
        """Return similarity history"""
        return self.similarity_history.copy()

    def clear_history(self):
        """Clear history and reset timestep"""
        self.similarity_history.clear()
        self.timestep = 0


# =============================================================================
# SIMILARITY LOGGING AND VISUALIZATION
# =============================================================================

def plot_similarity_scores(
    log_path: str,
    output_path: Optional[str] = None,
    show_threshold: bool = True,
    show_triggers: bool = True,
    figsize: Tuple[int, int] = (12, 6),
    title: Optional[str] = None
) -> None:
    """
    Plot similarity scores from a saved CAST log file.

    Args:
        log_path: Path to the JSON log file saved by CASTMultiLayerHook
        output_path: Optional path to save the plot image. If None, displays interactively.
        show_threshold: If True, draws a horizontal line at the threshold value
        show_triggers: If True, highlights timesteps where condition was triggered
        figsize: Figure size as (width, height) in inches
        title: Optional title for the plot. If None, uses default title.
    """
    import matplotlib.pyplot as plt

    # Load the log file
    log_path = Path(log_path)
    if not log_path.exists():
        raise FileNotFoundError(f"Log file not found: {log_path}")

    with open(log_path, 'r') as f:
        log_data = json.load(f)

    metadata = log_data.get('metadata', {})
    history = log_data.get('history', [])

    if not history:
        print("Warning: No similarity history found in log file")
        return

    # Extract data
    timesteps = [entry['timestep'] for entry in history]
    similarities = [entry['similarity'] for entry in history]
    triggered = [entry['triggered'] for entry in history]
    threshold = metadata.get('threshold', history[0].get('threshold', 0.5))

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Plot similarity scores
    ax.plot(timesteps, similarities, 'b-', linewidth=1.5, label='Not triggered', zorder=2)
    ax.scatter(timesteps, similarities, c='blue', s=20, zorder=3)

    # Show threshold line
    if show_threshold:
        ax.axhline(y=threshold, color='r', linestyle='--', linewidth=1.5,
                   label=f'Threshold (θ={threshold})', zorder=1)

    # Highlight triggered timesteps
    if show_triggers:
        triggered_timesteps = [t for t, trig in zip(timesteps, triggered) if trig]
        triggered_sims = [s for s, trig in zip(similarities, triggered) if trig]
        if triggered_timesteps:
            ax.scatter(triggered_timesteps, triggered_sims, c='green', s=80,
                      marker='o', edgecolors='darkgreen', linewidths=2,
                      label='Triggered', zorder=4)

    # Labels and title
    ax.set_xlabel('Timestep', fontsize=12)
    ax.set_ylabel('Cosine Similarity', fontsize=12)

    if title:
        ax.set_title(title, fontsize=14)
    else:
        total_triggered = sum(triggered)
        ax.set_title(f'CAST Condition Similarity Over Time\n'
                    f'(Triggered: {total_triggered}/{len(timesteps)} timesteps)', fontsize=14)

    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)

    # Set y-axis limits with some padding
    y_min = min(similarities) - 0.05
    y_max = max(max(similarities), threshold) + 0.05
    ax.set_ylim(y_min, y_max)

    plt.tight_layout()

    # Save or show
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved to {output_path}")
    else:
        plt.show()

    plt.close()


def load_similarity_log(log_path: str) -> Tuple[Dict, List[Dict]]:
    """
    Load a similarity log file and return metadata and history.

    Args:
        log_path: Path to the JSON log file

    Returns:
        Tuple of (metadata dict, history list)
    """
    with open(log_path, 'r') as f:
        log_data = json.load(f)

    return log_data.get('metadata', {}), log_data.get('history', [])


def summarize_similarity_log(log_path: str) -> Dict:
    """
    Generate summary statistics from a similarity log file.

    Args:
        log_path: Path to the JSON log file

    Returns:
        Dictionary with summary statistics
    """
    metadata, history = load_similarity_log(log_path)

    if not history:
        return {'error': 'No history found'}

    similarities = [entry['similarity'] for entry in history]
    triggered = [entry['triggered'] for entry in history]

    return {
        'total_timesteps': len(history),
        'triggered_count': sum(triggered),
        'trigger_rate': sum(triggered) / len(triggered) if triggered else 0,
        'similarity_mean': np.mean(similarities),
        'similarity_std': np.std(similarities),
        'similarity_min': np.min(similarities),
        'similarity_max': np.max(similarities),
        'threshold': metadata.get('threshold'),
        'condition_layer': metadata.get('condition_layer_idx'),
        'behavior_layers': metadata.get('behavior_layer_indices')
    }


if __name__ == '__main__':
    print("CAST Helpers module loaded successfully")
    print("")
    print("Available classes:")
    print("  - CASTMultiLayerHook: Multi-layer CAST with per-layer condition+behavior vectors")
    print("  - MultiPhaseCASTHook: Multi-phase CAST for task phase detection")
    print("")
    print("Vector loading functions:")
    print("  - load_precomputed_concept_vectors: Load individual concept vectors")
    print("  - load_and_combine_precomputed_vectors: Load and combine for deployment")
    print("  - vectors_exist: Check if precomputed vectors exist")
    print("")
    print("To compute vectors, use: scripts/precompute_vectors.py")
