"""
Example script demonstrating Conditional Activation Steering (CAST) with pretrained PaliGemma.

This script validates the CAST implementation by:
1. Loading precomputed condition and behavior vectors (same as used for VLA deployment)
2. Applying multi-layer CAST during text generation
3. Comparing baseline vs CAST outputs to verify steering is working

Uses the same vectors and configuration as VLA deployment, but on pretrained PaliGemma.

Prerequisites:
    Run scripts/precompute_vectors.py --vlm to generate the VLM vectors first.

Usage:
    python openpi/src/openpi/models_pytorch/example_CAST_usage.py
    python openpi/src/openpi/models_pytorch/example_CAST_usage.py --mode steering
    python openpi/src/openpi/models_pytorch/example_CAST_usage.py --alpha 6.0 --threshold 0.05
"""

import sys
from pathlib import Path

# Add project paths
project_root = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(project_root / "openpi" / "src"))
sys.path.insert(0, str(project_root))

from transformers import AutoProcessor, AutoTokenizer, PaliGemmaForConditionalGeneration
from PIL import Image
import torch
import torch.nn.functional as F
import json
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Union
import numpy as np

# Import constants and helpers
from constants import (
    LAYER_INDICES,
    ALPHA,
    CAST_THRESHOLD,
    CAST_LAYER_CONFIG_PATH,
    CAST_LAYER_CONFIG_NAME,
    EXAMPLE_TYPE,
    NUM_EXAMPLES,
    VECTORS_DIR,
    NORMALIZE_CONDITION_VECTORS,
    NORMALIZE_BEHAVIOR_VECTORS,
)
from openpi.models_pytorch.CAST_helpers import (
    load_layer_cast_config,
    load_precomputed_concept_vectors,
    combine_concept_vectors,
)


# =============================================================================
# CONFIGURATION
# =============================================================================

# Text generation prompt to test steering on
TASK_PROMPT = "Finish the sentence: it's not about the __, it's about the ____"

# Generation parameters
MAX_NEW_TOKENS = 30
TEMPERATURE = 0.8
USE_IMAGE = False  # Set True to test with a dummy image

# VLM vectors directory
VLM_VECTORS_DIR = VECTORS_DIR / "vlm" / EXAMPLE_TYPE / str(NUM_EXAMPLES)

OUTPUT_DIR = Path("/home/alex/dev/piper_bimanual/CAST_results/vlm_validation")


# =============================================================================
# CAST INFERENCE FUNCTIONS (adapted from CAST_helpers.py for PaliGemma)
# =============================================================================

def project_onto_condition_vector(
    hidden_state: torch.Tensor,
    condition_vec: torch.Tensor,
    use_tanh: bool = True
) -> torch.Tensor:
    """Project hidden state onto condition vector"""
    c = condition_vec.view(-1)
    c_dot_c = torch.dot(c, c)

    if c_dot_c < 1e-8:
        raise ValueError("Condition vector has near-zero norm")

    original_shape = hidden_state.shape
    h_flat = hidden_state.view(-1, hidden_state.shape[-1])

    h_dot_c = h_flat @ c
    proj_coeff = h_dot_c / c_dot_c
    projection = proj_coeff.unsqueeze(-1) * c.unsqueeze(0)

    if use_tanh:
        projection = torch.tanh(projection)

    return projection.view(original_shape)


def check_condition(
    hidden_state: torch.Tensor,
    condition_vec: torch.Tensor,
    threshold: float = 0.5,
    use_tanh: bool = True
) -> Tuple[bool, float]:
    """Check if condition is met for applying steering"""
    h_avg = hidden_state.mean(dim=-2)
    projection = project_onto_condition_vector(h_avg, condition_vec, use_tanh)

    # Compute cosine similarity
    h_flat = h_avg.view(-1, h_avg.shape[-1])
    p_flat = projection.view(-1, projection.shape[-1])
    similarity = F.cosine_similarity(h_flat, p_flat, dim=-1)

    similarity_scalar = similarity.mean().item()
    condition_triggered = similarity_scalar > threshold

    return condition_triggered, similarity_scalar


def apply_behavior_steering(
    hidden_state: torch.Tensor,
    behavior_vec: torch.Tensor,
    alpha: float
) -> torch.Tensor:
    """Apply behavior vector: h' = h + α · v"""
    v = behavior_vec.view(1, 1, -1)
    return hidden_state + alpha * v


class CASTMultiLayerHookForPaliGemma:
    """
    Multi-layer CAST hook for pretrained PaliGemma.

    This is a simplified version of CASTMultiLayerHook from CAST_helpers.py,
    adapted to work directly with PaliGemmaForConditionalGeneration.
    """

    def __init__(
        self,
        model: PaliGemmaForConditionalGeneration,
        behavior_vecs: Dict[int, torch.Tensor],
        condition_vecs: Dict[int, torch.Tensor],
        alpha: Union[float, Dict[int, float]] = 1.0,
        threshold: Union[float, Dict[int, float]] = 0.5,
        use_tanh: bool = True,
        apply_steering: bool = True,
        verbose: bool = True
    ):
        self.model = model
        self.device = model.device

        self.behavior_vecs = {idx: vec.to(self.device) for idx, vec in behavior_vecs.items()}
        self.condition_vecs = {idx: vec.to(self.device) for idx, vec in condition_vecs.items()}
        self.layer_indices = sorted(behavior_vecs.keys())

        self.use_tanh = use_tanh
        self.apply_steering = apply_steering
        self.verbose = verbose

        if isinstance(alpha, (int, float)):
            self.alphas = {idx: float(alpha) for idx in self.layer_indices}
        else:
            self.alphas = alpha

        if isinstance(threshold, (int, float)):
            self.thresholds = {idx: float(threshold) for idx in self.layer_indices}
        else:
            self.thresholds = threshold

        self.handles: Dict[int, any] = {}
        self.similarity_history: Dict[int, List[Dict]] = {idx: [] for idx in self.layer_indices}
        self.timestep = 0

    def _make_hook_fn(self, layer_idx: int):
        alpha = self.alphas.get(layer_idx, 1.0)
        threshold = self.thresholds.get(layer_idx, 0.5)
        condition_vec = self.condition_vecs[layer_idx]
        behavior_vec = self.behavior_vecs[layer_idx]

        def hook_fn(module, input, output):
            hidden = output[0] if isinstance(output, tuple) else output

            triggered, similarity = check_condition(hidden, condition_vec, threshold, self.use_tanh)

            self.similarity_history[layer_idx].append({
                'timestep': self.timestep,
                'similarity': similarity,
                'triggered': triggered,
                'threshold': threshold
            })

            if self.verbose:
                status = "TRIGGERED" if triggered else "not triggered"
                print(f"CAST [L{layer_idx}]: {status} (sim={similarity:.4f}, th={threshold})")

            if triggered and self.apply_steering:
                modified = apply_behavior_steering(hidden, behavior_vec, alpha)
                if self.verbose:
                    print(f"CAST [L{layer_idx}]: Steering applied (alpha={alpha})")

                if isinstance(output, tuple):
                    return (modified.to(hidden.dtype),) + output[1:]
                return modified.to(hidden.dtype)

            return output

        return hook_fn

    def register(self):
        """Register hooks at all layers"""
        for layer_idx in self.layer_indices:
            layer = self.model.language_model.model.layers[layer_idx]
            self.handles[layer_idx] = layer.mlp.down_proj.register_forward_hook(
                self._make_hook_fn(layer_idx)
            )
        print(f"CAST: Registered hooks at layers {self.layer_indices}")

    def remove(self):
        """Remove all hooks"""
        for handle in self.handles.values():
            handle.remove()
        self.handles.clear()

    def step(self):
        self.timestep += 1

    def get_trigger_summary(self) -> Dict[int, Dict]:
        """Get summary statistics per layer"""
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
                'similarity_mean': np.mean(similarities),
                'similarity_std': np.std(similarities),
                'similarity_min': np.min(similarities),
                'similarity_max': np.max(similarities),
            }
        return summary


# =============================================================================
# VECTOR LOADING
# =============================================================================

def load_vlm_vectors(
    config_name: str = CAST_LAYER_CONFIG_NAME,
    layer_indices: List[int] = LAYER_INDICES,
    vectors_dir: Path = VLM_VECTORS_DIR,
    device: str = "cuda"
) -> Tuple[Dict[int, torch.Tensor], Dict[int, torch.Tensor], Dict]:
    """
    Load precomputed VLM vectors and combine them according to the CAST config.

    Returns:
        Tuple of (condition_vecs, behavior_vecs, config)
        Each vec dict maps layer_idx -> combined vector
    """
    # Load the CAST layer config (same as used for VLA)
    layer_config = load_layer_cast_config(
        config_path=str(CAST_LAYER_CONFIG_PATH),
        config_name=config_name
    )

    condition_combination = layer_config.get('condition', {})
    behavior_combination = layer_config.get('behaviors', {})

    print(f"CAST config '{config_name}':")
    print(f"  Condition: {condition_combination}")
    print(f"  Behavior: {behavior_combination}")

    # Determine which concepts we need to load
    used_condition_concepts = [c for c, coef in condition_combination.items() if coef != 0]
    used_behavior_concepts = [c for c, coef in behavior_combination.items() if coef != 0]
    all_used_concepts = list(set(used_condition_concepts + used_behavior_concepts))

    if not all_used_concepts:
        raise ValueError("No concepts have non-zero coefficients in the config")

    print(f"Loading vectors for concepts: {all_used_concepts}")

    # Load precomputed vectors
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
                normalize=NORMALIZE_CONDITION_VECTORS
            )
            if cond_vec is not None:
                condition_vecs[layer_idx] = cond_vec.to(device)

        # Combine behavior vectors
        if used_behavior_concepts:
            behav_vec = combine_concept_vectors(
                concept_vectors,
                behavior_combination,
                normalize=NORMALIZE_BEHAVIOR_VECTORS
            )
            if behav_vec is not None:
                behavior_vecs[layer_idx] = behav_vec.to(device)

    print(f"Loaded and combined vectors for {len(condition_vecs)} layers")

    return condition_vecs, behavior_vecs, layer_config


# =============================================================================
# MAIN EXPERIMENT
# =============================================================================

def run_cast_experiment(
    task_prompt: str = TASK_PROMPT,
    layer_indices: List[int] = LAYER_INDICES,
    alpha: float = ALPHA,
    threshold: float = CAST_THRESHOLD,
    max_new_tokens: int = MAX_NEW_TOKENS,
    temperature: float = TEMPERATURE,
    use_image: bool = USE_IMAGE,
    verbose: bool = True
):
    """
    Run CAST experiment on pretrained PaliGemma using precomputed vectors.

    Loads vectors, runs baseline inference, runs CAST inference,
    and compares outputs.
    """
    print("=" * 80)
    print("CAST VALIDATION EXPERIMENT")
    print("Using precomputed vectors from scripts/precompute_vectors.py --vlm")
    print("=" * 80)

    # Check if VLM vectors exist
    if not VLM_VECTORS_DIR.exists():
        print(f"\nERROR: VLM vectors not found at {VLM_VECTORS_DIR}")
        print("Please run: python scripts/precompute_vectors.py --vlm")
        return None

    # Load model and tokenizer
    print("\n1. Loading pretrained PaliGemma model...")
    tokenizer = AutoTokenizer.from_pretrained("google/paligemma-3b-pt-224")
    processor = AutoProcessor.from_pretrained("google/paligemma-3b-pt-224") if use_image else None

    model = PaliGemmaForConditionalGeneration.from_pretrained(
        "google/paligemma-3b-pt-224",
        torch_dtype=torch.bfloat16
    )
    model.eval()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    print(f"   Model loaded on {device}")

    # Load precomputed vectors
    print(f"\n2. Loading precomputed vectors...")
    print(f"   Vectors dir: {VLM_VECTORS_DIR}")
    print(f"   Config: {CAST_LAYER_CONFIG_NAME}")

    condition_vecs, behavior_vecs, config = load_vlm_vectors(
        config_name=CAST_LAYER_CONFIG_NAME,
        layer_indices=layer_indices,
        vectors_dir=VLM_VECTORS_DIR,
        device=device
    )

    if not condition_vecs or not behavior_vecs:
        print("ERROR: No vectors loaded. Check your config and vectors directory.")
        return None

    print(f"   Loaded vectors for {len(condition_vecs)} layers")
    print(f"   Vector shape: {list(behavior_vecs.values())[0].shape}")

    # Create dummy image if needed
    image = Image.new('RGB', (224, 224), color='gray') if use_image else None

    # Prepare inputs for generation
    print(f"\n3. Preparing inputs...")
    print(f"   Task prompt: '{task_prompt}'")

    if use_image and processor is not None:
        inputs = processor(text=task_prompt, images=image, return_tensors="pt").to(device)
    else:
        inputs = tokenizer(text=task_prompt, return_tensors="pt").to(device)

    # Run baseline inference (NO CAST)
    print(f"\n4. Running BASELINE inference (no CAST)...")
    with torch.no_grad():
        baseline_output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            pad_token_id=tokenizer.eos_token_id
        )

    baseline_text = tokenizer.decode(baseline_output_ids[0], skip_special_tokens=True)
    print(f"   Baseline output: {baseline_text}")

    # Create and register CAST hook
    print(f"\n5. Running CAST inference...")
    print(f"   Alpha: {alpha}, Threshold: {threshold}")

    cast_hook = CASTMultiLayerHookForPaliGemma(
        model=model,
        behavior_vecs=behavior_vecs,
        condition_vecs=condition_vecs,
        alpha=alpha,
        threshold=threshold,
        use_tanh=True,
        apply_steering=True,
        verbose=verbose
    )

    cast_hook.register()

    with torch.no_grad():
        cast_output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            pad_token_id=tokenizer.eos_token_id
        )

    cast_hook.remove()

    cast_text = tokenizer.decode(cast_output_ids[0], skip_special_tokens=True)
    print(f"\n   CAST output: {cast_text}")

    # Print summary
    print("\n" + "=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)
    print(f"\nPrompt: '{task_prompt}'")
    print(f"\nBaseline: {baseline_text}")
    print(f"\nCAST:     {cast_text}")

    if baseline_text != cast_text:
        print(f"\n  CAST modified the output!")
    else:
        print(f"\n  Outputs are identical - try adjusting alpha or threshold")

    # Print trigger summary
    print("\n" + "-" * 40)
    print("Trigger Statistics by Layer:")
    trigger_summary = cast_hook.get_trigger_summary()
    for layer_idx, stats in trigger_summary.items():
        print(f"  Layer {layer_idx}: triggered {stats['triggered_count']}/{stats['total_timesteps']} times "
              f"(rate: {stats['trigger_rate']:.2%}), "
              f"sim: {stats['similarity_mean']:.4f} +/- {stats['similarity_std']:.4f}")

    # Save results
    if OUTPUT_DIR:
        save_results(
            output_dir=OUTPUT_DIR,
            baseline_text=baseline_text,
            cast_text=cast_text,
            task_prompt=task_prompt,
            layer_indices=layer_indices,
            alpha=alpha,
            threshold=threshold,
            config=config,
            trigger_summary=trigger_summary
        )

    return {
        'baseline': baseline_text,
        'cast': cast_text,
        'trigger_summary': trigger_summary
    }


def run_steering_only_experiment(
    task_prompt: str = TASK_PROMPT,
    alpha: float = ALPHA,
):
    """
    Run experiment with UNCONDITIONAL steering (no condition check).

    This validates that the steering vectors themselves work,
    independent of the condition checking logic.
    """
    print("=" * 80)
    print("STEERING-ONLY VALIDATION (No condition check)")
    print("Using precomputed vectors from scripts/precompute_vectors.py --vlm")
    print("=" * 80)

    # Check if VLM vectors exist
    if not VLM_VECTORS_DIR.exists():
        print(f"\nERROR: VLM vectors not found at {VLM_VECTORS_DIR}")
        print("Please run: python scripts/precompute_vectors.py --vlm")
        return None

    print("\n1. Loading pretrained PaliGemma model...")
    tokenizer = AutoTokenizer.from_pretrained("google/paligemma-3b-pt-224")

    model = PaliGemmaForConditionalGeneration.from_pretrained(
        "google/paligemma-3b-pt-224",
        torch_dtype=torch.bfloat16
    )
    model.eval()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    # Load precomputed vectors
    print(f"\n2. Loading precomputed vectors...")
    condition_vecs, behavior_vecs, config = load_vlm_vectors(
        config_name=CAST_LAYER_CONFIG_NAME,
        layer_indices=LAYER_INDICES,
        vectors_dir=VLM_VECTORS_DIR,
        device=device
    )

    # Prepare inputs
    inputs = tokenizer(text=task_prompt, return_tensors="pt").to(device)

    # Baseline
    print(f"\n3. Running BASELINE inference...")
    with torch.no_grad():
        baseline_output_ids = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=True,
            temperature=TEMPERATURE,
            pad_token_id=tokenizer.eos_token_id
        )
    baseline_text = tokenizer.decode(baseline_output_ids[0], skip_special_tokens=True)
    print(f"   Baseline: {baseline_text}")

    # Steering (unconditional)
    print(f"\n4. Running STEERED inference (unconditional, alpha={alpha})...")

    handles = []

    def make_steering_hook(layer_idx):
        vec = behavior_vecs[layer_idx]
        def hook_fn(module, input, output):
            hidden = output[0] if isinstance(output, tuple) else output
            modified = hidden + alpha * vec.view(1, 1, -1)
            if isinstance(output, tuple):
                return (modified.to(hidden.dtype),) + output[1:]
            return modified.to(hidden.dtype)
        return hook_fn

    for layer_idx in behavior_vecs.keys():
        layer = model.language_model.model.layers[layer_idx]
        handle = layer.mlp.down_proj.register_forward_hook(make_steering_hook(layer_idx))
        handles.append(handle)

    with torch.no_grad():
        steered_output_ids = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=True,
            temperature=TEMPERATURE,
            pad_token_id=tokenizer.eos_token_id
        )

    for handle in handles:
        handle.remove()

    steered_text = tokenizer.decode(steered_output_ids[0], skip_special_tokens=True)
    print(f"   Steered: {steered_text}")

    print("\n" + "=" * 80)
    print("COMPARISON")
    print("=" * 80)
    print(f"Baseline: {baseline_text}")
    print(f"Steered:  {steered_text}")

    if baseline_text != steered_text:
        print("\n  Steering modified the output!")
    else:
        print("\n  Outputs are identical")

    return {
        'baseline': baseline_text,
        'steered': steered_text
    }


def save_results(
    output_dir: Path,
    baseline_text: str,
    cast_text: str,
    task_prompt: str,
    layer_indices: List[int],
    alpha: float,
    threshold: float,
    config: Dict,
    trigger_summary: Dict
):
    """Save experiment results to disk"""
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    exp_dir = output_dir / f"cast_validation_{timestamp}"
    exp_dir.mkdir(parents=True, exist_ok=True)

    result_data = {
        "timestamp": datetime.now().isoformat(),
        "task_prompt": task_prompt,
        "cast_config": {
            "config_name": CAST_LAYER_CONFIG_NAME,
            "condition": config.get('condition', {}),
            "behaviors": config.get('behaviors', {}),
        },
        "parameters": {
            "layer_indices": layer_indices,
            "alpha": alpha,
            "threshold": threshold,
            "vectors_dir": str(VLM_VECTORS_DIR),
        },
        "outputs": {
            "baseline": baseline_text,
            "cast": cast_text,
        },
        "trigger_summary": {str(k): v for k, v in trigger_summary.items()}
    }

    with open(exp_dir / "results.json", 'w') as f:
        json.dump(result_data, f, indent=2)

    with open(exp_dir / "baseline.txt", 'w') as f:
        f.write(baseline_text)

    with open(exp_dir / "cast.txt", 'w') as f:
        f.write(cast_text)

    print(f"\n  Results saved to: {exp_dir}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Validate CAST implementation on pretrained PaliGemma")
    parser.add_argument("--mode", choices=["cast", "steering"], default="cast",
                       help="'cast' for full CAST, 'steering' for unconditional steering only")
    parser.add_argument("--alpha", type=float, default=ALPHA, help="Steering strength")
    parser.add_argument("--threshold", type=float, default=CAST_THRESHOLD, help="Condition threshold")
    parser.add_argument("--quiet", action="store_true", help="Reduce verbosity")
    parser.add_argument("--prompt", type=str, default=TASK_PROMPT, help="Task prompt to test")

    args = parser.parse_args()

    if args.mode == "cast":
        run_cast_experiment(
            task_prompt=args.prompt,
            alpha=args.alpha,
            threshold=args.threshold,
            verbose=not args.quiet
        )
    else:
        run_steering_only_experiment(
            task_prompt=args.prompt,
            alpha=args.alpha
        )
