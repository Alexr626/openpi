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
"""

import torch
import torch.nn.functional as F
from torch import Tensor
import math
import json
import time
import yaml
from pathlib import Path
from typing import Optional, Union, List, Tuple, Dict
from sklearn.decomposition import PCA
import numpy as np
from transformers import AutoTokenizer
from openpi.models_pytorch.pi0_pytorch import PI0Pytorch, make_att_2d_masks


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
    example_type: str = 'xH',
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
# MULTI-CONCEPT VECTOR COMPUTATION
# =============================================================================

def compute_all_concept_vectors(
    model: PI0Pytorch,
    yaml_path: str,
    layer_idx: int,
    tokenizer: AutoTokenizer,
    concepts: List[str],
    example_type: str = 'robotic_specific',
    max_examples_per_concept: int = 100
) -> Dict[str, Tensor]:
    """
    Compute vectors for all specified concepts from the YAML examples.

    Args:
        model: PI0Pytorch model
        yaml_path: Path to YAML file with contrasting examples
        layer_idx: Layer to extract vectors from
        tokenizer: Tokenizer
        concepts: List of concept names to compute vectors for
        example_type: 'robotic_specific' or 'general_physical'
        max_examples_per_concept: Max examples to use per concept

    Returns:
        Dictionary mapping concept name -> vector [hidden_dim]
    """
    yaml_data = load_contrasting_examples(yaml_path)
    concept_vectors = {}

    for concept in concepts:
        print(f"Computing vector for concept: {concept}")
        positive_examples, negative_examples = get_examples_for_concept(
            yaml_data, concept, example_type, max_examples_per_concept
        )

        if not positive_examples or not negative_examples:
            print(f"  Warning: No examples found for {concept}, skipping")
            continue

        # Compute vector using PCA-based extraction
        vector = compute_behavior_vector_cast(
            model=model,
            positive_prompts=positive_examples,
            negative_prompts=negative_examples,
            layer_idx=layer_idx,
            tokenizer=tokenizer,
            suffix_start_idx=None
        )
        concept_vectors[concept] = vector
        print(f"  Computed vector with shape {vector.shape}")

    return concept_vectors


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


def compute_phase_vectors(
    model: PI0Pytorch,
    yaml_path: str,
    layer_idx: int,
    tokenizer: AutoTokenizer,
    phase_conditions: Dict[str, Dict[str, int]],
    phase_behaviors: Dict[str, Dict[str, int]],
    condition_concepts: List[str],
    behavior_concepts: List[str],
    enabled_phases: List[str],
    example_type: str = 'robotic_specific',
    max_examples: int = 100,
    normalize_condition_vectors: bool = False,
    normalize_behavior_vectors: bool = False
) -> Tuple[Dict[str, Tensor], Dict[str, Tensor]]:
    """
    Compute condition and behavior vectors for all enabled phases at the given layer index.

    Args:
        model: PI0Pytorch model
        yaml_path: Path to YAML examples
        layer_idx: Layer for vector extraction
        tokenizer: Tokenizer
        phase_conditions: Phase condition combinations from constants
        phase_behaviors: Phase behavior combinations from constants
        condition_concepts: List of condition concept names
        behavior_concepts: List of behavior concept names
        enabled_phases: List of phase names that are enabled
        example_type: 'robotic_specific' or 'general_physical'
        max_examples: Max examples per concept

    Returns:
        Tuple of (phase_condition_vectors, phase_behavior_vectors)
        Each is a dict mapping phase_name -> combined vector
    """
    # Determine which concepts are actually used (have non-zero values in any enabled phase)
    used_concepts = set()
    for phase in enabled_phases:
        # Check condition concepts
        if phase in phase_conditions:
            for concept, sign in phase_conditions[phase].items():
                if sign != 0 and concept in condition_concepts:
                    used_concepts.add(concept)
        # Check behavior concepts
        if phase in phase_behaviors:
            for concept, sign in phase_behaviors[phase].items():
                if sign != 0 and concept in behavior_concepts:
                    used_concepts.add(concept)

    if not used_concepts:
        print(f"Warning: No concepts are used across enabled phases, returning empty vectors")
        return {}, {}

    used_concepts = list(used_concepts)
    print(f"Computing vectors for {len(used_concepts)} used concepts: {used_concepts}")
    concept_vectors = compute_all_concept_vectors(
        model, yaml_path, layer_idx, tokenizer,
        used_concepts, example_type, max_examples
    )

    # Compute combined vectors for each enabled phase
    phase_condition_vecs = {}
    phase_behavior_vecs = {}

    for phase in enabled_phases:
        print(f"Combining vectors for phase: {phase}")

        # Condition vector
        if phase in phase_conditions:
            cond_combination = phase_conditions[phase]
            cond_vec = combine_concept_vectors(
                concept_vectors, cond_combination, normalize=normalize_condition_vectors
            )
            if cond_vec is not None:
                phase_condition_vecs[phase] = cond_vec
                print(f"  Condition vector computed")
            else:
                print(f"  Condition vector skipped (all concepts zero)")

        # Behavior vector
        if phase in phase_behaviors:
            behav_combination = phase_behaviors[phase]
            behav_vec = combine_concept_vectors(
                concept_vectors, behav_combination, normalize=normalize_behavior_vectors
            )
            if behav_vec is not None:
                phase_behavior_vecs[phase] = behav_vec
                print(f"  Behavior vector computed")
            else:
                print(f"  Behavior vector skipped (all concepts zero)")

    return phase_condition_vecs, phase_behavior_vecs


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
# VECTOR EXTRACTION PHASE
# =============================================================================

def collect_hidden_states_for_cast(
    model: PI0Pytorch,
    prompts: List[str],
    layer_idx: int,
    tokenizer: AutoTokenizer,
    average_over_suffix_only: bool = False,
    suffix_start_idx: Optional[int] = None
) -> Tensor:
    """
    Collect hidden states from multiple prompts at a specific layer.

    Args:
        model: PI0Pytorch model instance
        prompts: List of prompt strings
        layer_idx: Which Gemma decoder layer to extract from
        tokenizer: Tokenizer for PaliGemma
        average_over_suffix_only: If True, average only over suffix tokens (for behavior vectors)
                                  If False, average over all tokens (for condition vectors)
        suffix_start_idx: If provided, this is the token index where the suffix starts.
                         Only used when average_over_suffix_only=True

    Returns:
        Hidden states tensor [num_prompts, hidden_dim] after token averaging
    """
    device = model.paligemma_with_expert.paligemma.device
    all_hidden_states = []

    for prompt in prompts:
        # Tokenize
        inputs = tokenizer(text=prompt, return_tensors="pt").to(device)
        input_ids = inputs["input_ids"]
        attention_mask = inputs.get("attention_mask", torch.ones_like(input_ids))

        # Get sequence length for suffix calculation
        seq_len = input_ids.shape[1]

        # Embed using the VLA's embedding layer
        token_embeds = model.paligemma_with_expert.embed_language_tokens(input_ids)
        embed_dim = token_embeds.shape[-1]
        token_embeds = token_embeds * math.sqrt(embed_dim)

        # Convert to model dtype
        if model.paligemma_with_expert.paligemma.language_model.layers[0].self_attn.q_proj.weight.dtype == torch.bfloat16:
            token_embeds = token_embeds.to(dtype=torch.bfloat16)

        activations = {}

        def hook(module, input, output):
            activations['hidden'] = (output[0] if isinstance(output, tuple) else output).detach()

        handle = model.paligemma_with_expert.paligemma.language_model.layers[layer_idx].mlp.down_proj.register_forward_hook(hook)

        try:
            with torch.no_grad():
                position_ids = torch.cumsum(attention_mask, dim=1) - 1
                att_masks = torch.zeros(1, seq_len, dtype=torch.bool, device=device)
                pad_masks = attention_mask.bool()
                att_2d_masks = make_att_2d_masks(pad_masks, att_masks)
                att_2d_masks_4d = att_2d_masks[:, None, :, :]
                att_2d_masks_4d = torch.where(att_2d_masks_4d, 0.0, -2.3819763e38).to(torch.bfloat16)

                _ = model.paligemma_with_expert.paligemma.language_model(
                    inputs_embeds=token_embeds,
                    attention_mask=att_2d_masks_4d,
                    position_ids=position_ids,
                )
        finally:
            handle.remove()

        hidden = activations['hidden']  # [1, seq_len, hidden_dim]

        # Apply token averaging
        if average_over_suffix_only and suffix_start_idx is not None:
            # Average only over suffix tokens (for behavior vectors)
            suffix_hidden = hidden[:, suffix_start_idx:, :]
            averaged = suffix_hidden.mean(dim=1)  # [1, hidden_dim]
        else:
            # Average over all tokens (for condition vectors)
            averaged = hidden.mean(dim=1)  # [1, hidden_dim]

        all_hidden_states.append(averaged)

    # Stack all prompts: [num_prompts, hidden_dim]
    return torch.cat(all_hidden_states, dim=0)


def extract_vector_with_pca(
    positive_hidden_states: Tensor,
    negative_hidden_states: Tensor
) -> Tensor:
    """
    Extract a vector using PCA on mean-centered contrasting examples.

    Following the CAST paper:
    1. Calculate μ = (H+ + H-) / 2
    2. Subtract μ from all examples
    3. Stack examples alternating positive/negative
    4. Apply PCA and extract first principal component

    Args:
        positive_hidden_states: Hidden states from positive prompts [n_examples, hidden_dim]
        negative_hidden_states: Hidden states from negative prompts [n_examples, hidden_dim]

    Returns:
        First principal component vector [hidden_dim]
    """
    device = positive_hidden_states.device
    dtype = positive_hidden_states.dtype

    # Step 1: Calculate mean (center point between positive and negative)
    # μ = mean of all examples
    all_examples = torch.cat([positive_hidden_states, negative_hidden_states], dim=0)
    mu = all_examples.mean(dim=0, keepdim=True)  # [1, hidden_dim]

    # Step 2: Mean center all examples
    positive_centered = positive_hidden_states - mu
    negative_centered = negative_hidden_states - mu

    # Step 3: Stack alternating positive/negative for PCA
    n_examples = positive_hidden_states.shape[0]
    stacked = torch.zeros(2 * n_examples, positive_hidden_states.shape[1],
                          device=device, dtype=dtype)
    for i in range(n_examples):
        stacked[2*i] = positive_centered[i]
        stacked[2*i + 1] = negative_centered[i]

    # Step 4: Apply PCA to extract first principal component
    # Move to CPU for sklearn PCA, then back to device
    stacked_np = stacked.float().cpu().numpy()
    pca = PCA(n_components=1)
    pca.fit(stacked_np)

    # First principal component
    principal_component = torch.from_numpy(pca.components_[0]).to(device=device, dtype=dtype)

    # Ensure the direction points from negative to positive
    # Check by computing mean projections
    pos_proj = (positive_centered @ principal_component.unsqueeze(-1)).mean()
    neg_proj = (negative_centered @ principal_component.unsqueeze(-1)).mean()

    if neg_proj > pos_proj:
        principal_component = -principal_component

    return principal_component


def compute_behavior_vector_cast(
    model: PI0Pytorch,
    positive_prompts: List[str],
    negative_prompts: List[str],
    layer_idx: int,
    tokenizer: AutoTokenizer,
    suffix_start_idx: Optional[int] = None
) -> Tensor:
    """
    Compute behavior vector for CAST using PCA on suffix-averaged hidden states.

    For behavior vectors, we average only over suffix tokens (the part of the prompt
    that specifies the desired behavior).

    Args:
        model: PI0Pytorch model instance
        positive_prompts: List of positive example prompts
        negative_prompts: List of negative example prompts
        layer_idx: Which layer to extract from
        tokenizer: Tokenizer
        suffix_start_idx: Token index where the behavior-relevant suffix starts
                         If None, uses all tokens

    Returns:
        Behavior vector [hidden_dim]
    """
    # Collect hidden states with suffix averaging
    positive_hidden = collect_hidden_states_for_cast(
        model, positive_prompts, layer_idx, tokenizer,
        average_over_suffix_only=(suffix_start_idx is not None),
        suffix_start_idx=suffix_start_idx
    )
    negative_hidden = collect_hidden_states_for_cast(
        model, negative_prompts, layer_idx, tokenizer,
        average_over_suffix_only=(suffix_start_idx is not None),
        suffix_start_idx=suffix_start_idx
    )

    # Extract vector using PCA
    return extract_vector_with_pca(positive_hidden, negative_hidden)


def compute_condition_vector_cast(
    model: PI0Pytorch,
    positive_prompts: List[str],
    negative_prompts: List[str],
    layer_idx: int,
    tokenizer: AutoTokenizer
) -> Tensor:
    """
    Compute condition vector for CAST using PCA on full-sequence-averaged hidden states.

    For condition vectors, we average over all tokens (to capture the full context
    that triggers the condition).

    Args:
        model: PI0Pytorch model instance
        positive_prompts: List of positive condition prompts (when condition should trigger)
        negative_prompts: List of negative condition prompts (when condition should NOT trigger)
        layer_idx: Which layer to extract from
        tokenizer: Tokenizer

    Returns:
        Condition vector [hidden_dim]
    """
    # Collect hidden states with full token averaging
    positive_hidden = collect_hidden_states_for_cast(
        model, positive_prompts, layer_idx, tokenizer,
        average_over_suffix_only=False
    )
    negative_hidden = collect_hidden_states_for_cast(
        model, negative_prompts, layer_idx, tokenizer,
        average_over_suffix_only=False
    )

    # Extract vector using PCA
    return extract_vector_with_pca(positive_hidden, negative_hidden)


def compute_cast_vectors_multi_layer(
    model: PI0Pytorch,
    behavior_positive_prompts: List[str],
    behavior_negative_prompts: List[str],
    condition_positive_prompts: List[str],
    condition_negative_prompts: List[str],
    behavior_layer_indices: List[int],
    condition_layer_idx: int,
    tokenizer: AutoTokenizer,
    behavior_suffix_start_idx: Optional[int] = None
) -> Tuple[Dict[int, Tensor], Tensor]:
    """
    Compute all CAST vectors for multi-layer application.

    Args:
        model: PI0Pytorch model
        behavior_positive_prompts: Positive prompts for behavior vector
        behavior_negative_prompts: Negative prompts for behavior vector
        condition_positive_prompts: Positive prompts for condition (when to trigger)
        condition_negative_prompts: Negative prompts for condition (when NOT to trigger)
        behavior_layer_indices: List of layers to apply behavior steering
        condition_layer_idx: Layer to check condition
        tokenizer: Tokenizer
        behavior_suffix_start_idx: Optional suffix start for behavior extraction

    Returns:
        Tuple of (behavior_vectors_dict, condition_vector)
        - behavior_vectors_dict: {layer_idx: behavior_vector}
        - condition_vector: [hidden_dim]
    """
    # Compute behavior vectors for each layer
    behavior_vectors = {}
    for layer_idx in behavior_layer_indices:
        behavior_vectors[layer_idx] = compute_behavior_vector_cast(
            model, behavior_positive_prompts, behavior_negative_prompts,
            layer_idx, tokenizer, behavior_suffix_start_idx
        )

    # Compute single condition vector
    condition_vector = compute_condition_vector_cast(
        model, condition_positive_prompts, condition_negative_prompts,
        condition_layer_idx, tokenizer
    )

    return behavior_vectors, condition_vector


# =============================================================================
# INFERENCE/APPLICATION PHASE
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
# HOOK CLASSES FOR INFERENCE
# =============================================================================

class CASTConditionChecker:
    """
    Hook to check condition during first forward pass.

    This captures the hidden state at the condition layer and checks
    whether the condition is triggered.
    """

    def __init__(
        self,
        model: PI0Pytorch,
        condition_vec: Tensor,
        condition_layer_idx: int,
        threshold: float = 0.5,
        use_tanh: bool = True
    ):
        self.model = model
        self.device = model.paligemma_with_expert.paligemma.device
        self.condition_vec = condition_vec.to(self.device)
        self.condition_layer_idx = condition_layer_idx
        self.threshold = threshold
        self.use_tanh = use_tanh
        self.handle = None
        self.condition_triggered = False
        self.similarity_value = 0.0
        self.checked = False

    def hook_fn(self, module, input, output):
        """Hook to check condition"""
        if self.checked:
            return  # Only check once

        hidden = output[0] if isinstance(output, tuple) else output

        self.condition_triggered, self.similarity_value = check_condition(
            hidden, self.condition_vec, self.threshold, self.use_tanh
        )
        self.checked = True

        status = "TRIGGERED" if self.condition_triggered else "NOT triggered"
        print(f"CAST Condition Check: {status} (similarity={self.similarity_value:.4f}, threshold={self.threshold})")

    def register(self):
        """Register hook"""
        layer = self.model.paligemma_with_expert.paligemma.language_model.layers[self.condition_layer_idx]
        self.handle = layer.mlp.down_proj.register_forward_hook(self.hook_fn)

    def remove(self):
        """Remove hook"""
        if self.handle:
            self.handle.remove()
            self.handle = None

    def reset(self):
        """Reset for new sequence"""
        self.checked = False
        self.condition_triggered = False
        self.similarity_value = 0.0


class CASTBehaviorApplier:
    """
    Hook to apply behavior steering when condition is triggered.

    This applies the behavior vector at specified layers only when
    the condition checker has determined the condition is met.
    """

    def __init__(
        self,
        model: PI0Pytorch,
        behavior_vecs: Dict[int, Tensor],
        condition_checker: CASTConditionChecker,
        alpha: Union[float, Dict[int, float]]
    ):
        """
        Args:
            model: PI0Pytorch model
            behavior_vecs: Dict mapping layer_idx -> behavior vector
            condition_checker: CASTConditionChecker instance to query
            alpha: Scaling factor (single value or per-layer dict)
        """
        self.model = model
        self.device = model.paligemma_with_expert.paligemma.device
        self.behavior_vecs = {idx: vec.to(self.device) for idx, vec in behavior_vecs.items()}
        self.condition_checker = condition_checker

        if isinstance(alpha, (int, float)):
            self.alphas = {idx: float(alpha) for idx in behavior_vecs.keys()}
        else:
            self.alphas = alpha

        self.handles: Dict[int, any] = {}

    def _make_hook_fn(self, layer_idx: int):
        """Create hook function for a specific layer"""
        behavior_vec = self.behavior_vecs[layer_idx]
        alpha = self.alphas[layer_idx]

        def hook_fn(module, input, output):
            # Only apply if condition was triggered
            if not self.condition_checker.condition_triggered:
                return output

            hidden = output[0] if isinstance(output, tuple) else output

            # Apply behavior steering
            modified = apply_behavior_steering(hidden, behavior_vec, alpha)

            print(f"CAST: Applying behavior steering at layer {layer_idx} (alpha={alpha})")

            if isinstance(output, tuple):
                return (modified.to(hidden.dtype),) + output[1:]
            return modified.to(hidden.dtype)

        return hook_fn

    def register(self):
        """Register hooks to all behavior layers"""
        for layer_idx in self.behavior_vecs.keys():
            layer = self.model.paligemma_with_expert.paligemma.language_model.layers[layer_idx]
            self.handles[layer_idx] = layer.mlp.down_proj.register_forward_hook(
                self._make_hook_fn(layer_idx)
            )

    def remove(self):
        """Remove all hooks"""
        for handle in self.handles.values():
            handle.remove()
        self.handles.clear()


class CASTSteeringManager:
    """
    High-level manager for CAST steering during inference.

    Coordinates condition checking and behavior application across
    multiple forward passes.

    Usage:
        manager = CASTSteeringManager(
            model, behavior_vecs, condition_vec,
            behavior_layer_indices, condition_layer_idx,
            alpha, threshold
        )
        manager.register()

        # Run inference...

        manager.remove()
    """

    def __init__(
        self,
        model: PI0Pytorch,
        behavior_vecs: Dict[int, Tensor],
        condition_vec: Tensor,
        behavior_layer_indices: List[int],
        condition_layer_idx: int,
        alpha: Union[float, Dict[int, float]],
        threshold: float = 0.5,
        use_tanh: bool = True
    ):
        """
        Args:
            model: PI0Pytorch model
            behavior_vecs: Dict mapping layer_idx -> behavior vector
            condition_vec: Condition vector [hidden_dim]
            behavior_layer_indices: Layers where behavior is applied
            condition_layer_idx: Layer where condition is checked
            alpha: Scaling factor
            threshold: Similarity threshold θ
            use_tanh: Apply tanh to projection
        """
        self.model = model
        self.condition_layer_idx = condition_layer_idx

        # Create condition checker
        self.condition_checker = CASTConditionChecker(
            model, condition_vec, condition_layer_idx, threshold, use_tanh
        )

        # Create behavior applier
        self.behavior_applier = CASTBehaviorApplier(
            model, behavior_vecs, self.condition_checker, alpha
        )

        self._registered = False

    def register(self):
        """Register all hooks"""
        if self._registered:
            return
        self.condition_checker.register()
        self.behavior_applier.register()
        self._registered = True

    def remove(self):
        """Remove all hooks"""
        if not self._registered:
            return
        self.behavior_applier.remove()
        self.condition_checker.remove()
        self._registered = False

    def reset(self):
        """Reset condition checker for new sequence"""
        self.condition_checker.reset()

    @property
    def condition_triggered(self) -> bool:
        """Whether condition was triggered in last check"""
        return self.condition_checker.condition_triggered

    @property
    def similarity_value(self) -> float:
        """Similarity value from last check"""
        return self.condition_checker.similarity_value


class CASTSingleLayerHook:
    """
    Simplified CAST hook that applies both condition check and behavior
    at the same layer. Useful when condition_layer == behavior_layer.

    Performs the full CAST operation: h' ← h + f(sim(h, proj_c h)) · α · v
    """

    def __init__(
        self,
        model: PI0Pytorch,
        condition_vec: Tensor,
        behavior_vec: Tensor,
        alpha: float,
        layer_idx: int,
        threshold: float = 0.5,
        use_tanh: bool = True
    ):
        self.model = model
        self.device = model.paligemma_with_expert.paligemma.device
        self.condition_vec = condition_vec.to(self.device)
        self.behavior_vec = behavior_vec.to(self.device)
        self.alpha = alpha
        self.layer_idx = layer_idx
        self.threshold = threshold
        self.use_tanh = use_tanh
        self.handle = None
        self.last_triggered = False
        self.last_similarity = 0.0

    def hook_fn(self, module, input, output):
        """Hook function that applies full CAST steering"""
        hidden = output[0] if isinstance(output, tuple) else output

        modified, triggered, similarity = apply_cast_steering(
            hidden, self.condition_vec, self.behavior_vec,
            self.alpha, self.threshold, self.use_tanh
        )

        self.last_triggered = triggered
        self.last_similarity = similarity

        status = "APPLIED" if triggered else "SKIPPED"
        print(f"CAST at layer {self.layer_idx}: {status} (sim={similarity:.4f})")

        if isinstance(output, tuple):
            return (modified.to(hidden.dtype),) + output[1:]
        return modified.to(hidden.dtype)

    def register(self):
        """Register hook"""
        layer = self.model.paligemma_with_expert.paligemma.language_model.layers[self.layer_idx]
        self.handle = layer.mlp.down_proj.register_forward_hook(self.hook_fn)

    def remove(self):
        """Remove hook"""
        if self.handle:
            self.handle.remove()
            self.handle = None


class CASTMultiLayerHook:
    """
    Multi-layer CAST that applies condition check at one layer and
    behavior steering at multiple layers.

    Similar to CASTSteeringManager but with a simpler interface matching
    the existing MultiLayerSteeringHook pattern.
    """

    def __init__(
        self,
        model: PI0Pytorch,
        behavior_vecs: Dict[int, Tensor],
        condition_vec: Tensor,
        condition_layer_idx: int,
        alpha: Union[float, Dict[int, float]],
        threshold: float = 0.5,
        use_tanh: bool = True,
        check_every_forward: bool = True,
        log_path: Optional[str] = None
    ):
        """
        Args:
            model: PI0Pytorch model
            behavior_vecs: Dict mapping layer_idx -> behavior vector
            condition_vec: Condition vector [hidden_dim]
            condition_layer_idx: Layer where condition is checked
            alpha: Scaling factor (single or per-layer)
            threshold: Similarity threshold θ
            use_tanh: Apply tanh to projection
            check_every_forward: If True, re-evaluate condition on every forward pass
                                 (for VLA deployment). If False, only check once per
                                 sequence (for autoregressive text generation).
            log_path: Optional path to save similarity scores. If provided, scores
                      are logged at each timestep and saved when remove() is called.
        """
        self.model = model
        self.device = model.paligemma_with_expert.paligemma.device
        self.behavior_vecs = {idx: vec.to(self.device) for idx, vec in behavior_vecs.items()}
        self.condition_vec = condition_vec.to(self.device)
        self.condition_layer_idx = condition_layer_idx
        self.threshold = threshold
        self.use_tanh = use_tanh
        self.check_every_forward = check_every_forward
        self.log_path = log_path

        if isinstance(alpha, (int, float)):
            self.alphas = {idx: float(alpha) for idx in behavior_vecs.keys()}
        else:
            self.alphas = alpha

        self.handles: Dict[int, any] = {}
        self.condition_triggered = False
        self.similarity_value = 0.0
        self._condition_checked = False

        # Similarity logging
        self.similarity_history: List[Dict] = []
        self.timestep = 0

    def _condition_hook_fn(self, module, input, output):
        """Hook for condition checking layer"""
        # For VLA deployment, check condition on every forward pass
        # For autoregressive generation, only check once per sequence
        if not self.check_every_forward and self._condition_checked:
            return

        hidden = output[0] if isinstance(output, tuple) else output

        self.condition_triggered, self.similarity_value = check_condition(
            hidden, self.condition_vec, self.threshold, self.use_tanh
        )
        self._condition_checked = True

        # Log similarity score
        log_entry = {
            'timestep': self.timestep,
            'similarity': self.similarity_value,
            'triggered': self.condition_triggered,
            'threshold': self.threshold,
            'timestamp': time.time()
        }
        self.similarity_history.append(log_entry)
        self.timestep += 1

        status = "TRIGGERED" if self.condition_triggered else "NOT triggered"
        print(f"CAST Condition: {status} (similarity={self.similarity_value:.4f})")

    def _make_behavior_hook_fn(self, layer_idx: int):
        """Create behavior hook for a specific layer"""
        behavior_vec = self.behavior_vecs[layer_idx]
        alpha = self.alphas[layer_idx]

        def hook_fn(module, input, output):
            if not self.condition_triggered:
                return output

            hidden = output[0] if isinstance(output, tuple) else output
            modified = apply_behavior_steering(hidden, behavior_vec, alpha)

            print(f"CAST: Steering at layer {layer_idx}")

            if isinstance(output, tuple):
                return (modified.to(hidden.dtype),) + output[1:]
            return modified.to(hidden.dtype)

        return hook_fn

    def register(self):
        """Register all hooks"""
        # Register condition hook
        cond_layer = self.model.paligemma_with_expert.paligemma.language_model.layers[self.condition_layer_idx]
        self.handles['condition'] = cond_layer.mlp.down_proj.register_forward_hook(self._condition_hook_fn)

        # Register behavior hooks
        for layer_idx in self.behavior_vecs.keys():
            layer = self.model.paligemma_with_expert.paligemma.language_model.layers[layer_idx]
            self.handles[layer_idx] = layer.mlp.down_proj.register_forward_hook(
                self._make_behavior_hook_fn(layer_idx)
            )

    def remove(self):
        """Remove all hooks and save similarity log if path was provided"""
        for handle in self.handles.values():
            handle.remove()
        self.handles.clear()

        # Auto-save similarity log when removing hooks
        if self.log_path and self.similarity_history:
            self.save_similarity_log()

    def reset(self):
        """Reset for new sequence"""
        self._condition_checked = False
        self.condition_triggered = False
        self.similarity_value = 0.0

    def save_similarity_log(self, path: Optional[str] = None):
        """
        Save similarity history to a JSON file.

        Args:
            path: Path to save the log. If None, uses self.log_path.
                  If both are None, does nothing.
        """
        save_path = path or self.log_path
        if not save_path:
            print("Warning: No log path specified, cannot save similarity log")
            return

        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        log_data = {
            'metadata': {
                'threshold': self.threshold,
                'condition_layer_idx': self.condition_layer_idx,
                'behavior_layer_indices': list(self.behavior_vecs.keys()),
                'alpha': self.alphas,
                'total_timesteps': len(self.similarity_history)
            },
            'history': self.similarity_history
        }

        with open(save_path, 'w') as f:
            json.dump(log_data, f, indent=2)

        print(f"CAST: Saved similarity log ({len(self.similarity_history)} timesteps) to {save_path}")

    def clear_history(self):
        """Clear the similarity history and reset timestep counter"""
        self.similarity_history.clear()
        self.timestep = 0

    def get_similarity_history(self) -> List[Dict]:
        """Return the similarity history"""
        return self.similarity_history.copy()


# =============================================================================
# CONVENIENCE FUNCTIONS FOR SIMPLE USAGE
# =============================================================================

def compute_simple_behavior_vector(
    model: PI0Pytorch,
    positive_prompt: str,
    negative_prompt: str,
    layer_idx: int,
    tokenizer: AutoTokenizer
) -> Tensor:
    """
    Simple behavior vector computation from a single pair of prompts.

    For more robust vectors, use compute_behavior_vector_cast with multiple prompts.

    Args:
        model: PI0Pytorch model
        positive_prompt: Positive example prompt
        negative_prompt: Negative example prompt
        layer_idx: Layer to extract from
        tokenizer: Tokenizer

    Returns:
        Behavior vector [hidden_dim]
    """
    return compute_behavior_vector_cast(
        model, [positive_prompt], [negative_prompt],
        layer_idx, tokenizer, suffix_start_idx=None
    )


def compute_simple_condition_vector(
    model: PI0Pytorch,
    condition_trigger_prompt: str,
    condition_no_trigger_prompt: str,
    layer_idx: int,
    tokenizer: AutoTokenizer
) -> Tensor:
    """
    Simple condition vector computation from a single pair of prompts.

    Args:
        model: PI0Pytorch model
        condition_trigger_prompt: Prompt where condition should trigger
        condition_no_trigger_prompt: Prompt where condition should NOT trigger
        layer_idx: Layer to extract from
        tokenizer: Tokenizer

    Returns:
        Condition vector [hidden_dim]
    """
    return compute_condition_vector_cast(
        model, [condition_trigger_prompt], [condition_no_trigger_prompt],
        layer_idx, tokenizer
    )


def setup_cast_steering(
    model: PI0Pytorch,
    behavior_positive_prompt: str,
    behavior_negative_prompt: str,
    condition_trigger_prompt: str,
    condition_no_trigger_prompt: str,
    behavior_layer_indices: List[int],
    condition_layer_idx: int,
    tokenizer: AutoTokenizer,
    alpha: float = 1.0,
    threshold: float = 0.5
) -> CASTMultiLayerHook:
    """
    Convenience function to set up CAST steering from simple prompts.

    Args:
        model: PI0Pytorch model
        behavior_positive_prompt: Positive behavior prompt
        behavior_negative_prompt: Negative behavior prompt
        condition_trigger_prompt: Prompt where condition should trigger
        condition_no_trigger_prompt: Prompt where condition should NOT trigger
        behavior_layer_indices: Layers to apply behavior steering
        condition_layer_idx: Layer to check condition
        tokenizer: Tokenizer
        alpha: Scaling factor
        threshold: Similarity threshold

    Returns:
        CASTMultiLayerHook ready to be registered
    """
    # Compute behavior vectors for each layer
    behavior_vecs = {}
    for layer_idx in behavior_layer_indices:
        behavior_vecs[layer_idx] = compute_simple_behavior_vector(
            model, behavior_positive_prompt, behavior_negative_prompt,
            layer_idx, tokenizer
        )

    # Compute condition vector
    condition_vec = compute_simple_condition_vector(
        model, condition_trigger_prompt, condition_no_trigger_prompt,
        condition_layer_idx, tokenizer
    )

    # Create and return hook
    return CASTMultiLayerHook(
        model, behavior_vecs, condition_vec,
        condition_layer_idx, alpha, threshold
    )


# =============================================================================
# MULTI-CONDITION SUPPORT
# =============================================================================

class MultiConditionCASTHook:
    """
    CAST with multiple conditions combined with logical operations.

    Supports OR (any condition triggers) and AND (all conditions must trigger)
    combinations of multiple condition vectors.
    """

    def __init__(
        self,
        model: PI0Pytorch,
        behavior_vecs: Dict[int, Tensor],
        condition_vecs: List[Tensor],
        condition_layer_indices: List[int],
        alpha: Union[float, Dict[int, float]],
        threshold: float = 0.5,
        combination_mode: str = 'or',  # 'or' or 'and'
        use_tanh: bool = True
    ):
        """
        Args:
            model: PI0Pytorch model
            behavior_vecs: Dict mapping layer_idx -> behavior vector
            condition_vecs: List of condition vectors
            condition_layer_indices: Layer index for each condition vector
            alpha: Scaling factor
            threshold: Similarity threshold for each condition
            combination_mode: 'or' (any triggers) or 'and' (all must trigger)
            use_tanh: Apply tanh to projection
        """
        self.model = model
        self.device = model.paligemma_with_expert.paligemma.device
        self.behavior_vecs = {idx: vec.to(self.device) for idx, vec in behavior_vecs.items()}
        self.condition_vecs = [vec.to(self.device) for vec in condition_vecs]
        self.condition_layer_indices = condition_layer_indices
        self.threshold = threshold
        self.combination_mode = combination_mode
        self.use_tanh = use_tanh

        if isinstance(alpha, (int, float)):
            self.alphas = {idx: float(alpha) for idx in behavior_vecs.keys()}
        else:
            self.alphas = alpha

        self.handles: Dict[str, any] = {}
        self.condition_results: List[bool] = [False] * len(condition_vecs)
        self.similarity_values: List[float] = [0.0] * len(condition_vecs)
        self._conditions_checked: List[bool] = [False] * len(condition_vecs)

    @property
    def condition_triggered(self) -> bool:
        """Combined condition result based on combination mode"""
        if self.combination_mode == 'or':
            return any(self.condition_results)
        elif self.combination_mode == 'and':
            return all(self.condition_results)
        else:
            raise ValueError(f"Unknown combination_mode: {self.combination_mode}")

    def _make_condition_hook_fn(self, cond_idx: int):
        """Create condition hook for a specific condition"""
        condition_vec = self.condition_vecs[cond_idx]

        def hook_fn(module, input, output):
            if self._conditions_checked[cond_idx]:
                return

            hidden = output[0] if isinstance(output, tuple) else output
            triggered, similarity = check_condition(
                hidden, condition_vec, self.threshold, self.use_tanh
            )

            self.condition_results[cond_idx] = triggered
            self.similarity_values[cond_idx] = similarity
            self._conditions_checked[cond_idx] = True

            status = "triggered" if triggered else "not triggered"
            print(f"CAST Condition {cond_idx}: {status} (sim={similarity:.4f})")

        return hook_fn

    def _make_behavior_hook_fn(self, layer_idx: int):
        """Create behavior hook"""
        behavior_vec = self.behavior_vecs[layer_idx]
        alpha = self.alphas[layer_idx]

        def hook_fn(module, input, output):
            if not self.condition_triggered:
                return output

            hidden = output[0] if isinstance(output, tuple) else output
            modified = apply_behavior_steering(hidden, behavior_vec, alpha)

            print(f"CAST: Multi-condition steering at layer {layer_idx}")

            if isinstance(output, tuple):
                return (modified.to(hidden.dtype),) + output[1:]
            return modified.to(hidden.dtype)

        return hook_fn

    def register(self):
        """Register all hooks"""
        # Register condition hooks
        for i, layer_idx in enumerate(self.condition_layer_indices):
            layer = self.model.paligemma_with_expert.paligemma.language_model.layers[layer_idx]
            self.handles[f'condition_{i}'] = layer.mlp.down_proj.register_forward_hook(
                self._make_condition_hook_fn(i)
            )

        # Register behavior hooks
        for layer_idx in self.behavior_vecs.keys():
            layer = self.model.paligemma_with_expert.paligemma.language_model.layers[layer_idx]
            self.handles[f'behavior_{layer_idx}'] = layer.mlp.down_proj.register_forward_hook(
                self._make_behavior_hook_fn(layer_idx)
            )

    def remove(self):
        """Remove all hooks"""
        for handle in self.handles.values():
            handle.remove()
        self.handles.clear()

    def reset(self):
        """Reset for new sequence"""
        n_conds = len(self.condition_vecs)
        self.condition_results = [False] * n_conds
        self.similarity_values = [0.0] * n_conds
        self._conditions_checked = [False] * n_conds


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
    print("Available classes:")
    print("  - CASTConditionChecker: Check condition at a single layer")
    print("  - CASTBehaviorApplier: Apply behavior steering at multiple layers")
    print("  - CASTSteeringManager: High-level manager coordinating both")
    print("  - CASTSingleLayerHook: Simple single-layer CAST")
    print("  - CASTMultiLayerHook: Multi-layer CAST with separate condition/behavior layers")
    print("  - MultiConditionCASTHook: Multiple conditions with OR/AND logic")
    print("")
    print("Convenience functions:")
    print("  - compute_behavior_vector_cast: PCA-based behavior vector extraction")
    print("  - compute_condition_vector_cast: PCA-based condition vector extraction")
    print("  - setup_cast_steering: Quick setup from simple prompts")
