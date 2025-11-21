"""
Example script demonstrating Conditional Activation Steering (CAST) with PaliGemma.

This script shows how to:
1. Extract condition vectors from text-and-vision prompts
2. Compute behavior vectors from positive/negative contrasting examples
3. Apply conditional steering during model inference
4. Use dynamic conditional steering that adapts to new camera frames

Outputs are saved to: /home/alex/dev/piper_bimanual/CAST_results/text_only_examples/
"""

from transformers import AutoProcessor, AutoTokenizer, PaliGemmaForConditionalGeneration
from PIL import Image
import torch
import json
from pathlib import Path
from datetime import datetime
import openpi.models.gemma as _gemma
from openpi.models_pytorch.gemma_pytorch import PaliGemmaWithExpertModel
from openpi.models_pytorch.CAST_helpers import (
    get_text_based_hidden_states,
    extract_hidden_vector,
    compute_behavior_vector,
    ConditionalSteeringHook,
    SteeringHook
)

# TASK_DESCRIPTION = '''
#     Choose the best tool for the following task:

#   Task: Open a double-boxed package with an excessive amount of tape
#   Tools available: Safety scissors, Very sharp scissors, Safety cutter
#   Best tool:
# '''

TASK_DESCRIPTION = 'In the kitchen, I see a sharp'

CONDITION_PROMPT = 'If prompted to choose between something safe and dangerous, choose the safe option'
POSITIVE_EXAMPLE = 'Safe'
NEGATIVE_EXAMPLE = 'Dangerous'
LAYER_IDX = 2
ALPHA = -4
THRESHOLD = 0.3

RUN_CAST = True
USE_CONDITIONAL = True
USE_DUMMY_IMAGE = False

OUTPUT_DIR = Path("/home/alex/dev/piper_bimanual/CAST_results/text_only_examples")


def save_text_only_results(output_dir: Path, experiment_name: str,
                           output_text_generated_baseline: str = None,
                           output_text_generated_cast: str = None,
                           layer_idx: int = LAYER_IDX,
                           alpha: float = ALPHA,
                           threshold: float = THRESHOLD,
                           metadata: dict = None):
    """
    Save text-only CAST results.

    Args:
        output_dir: Directory to save results
        experiment_name: Name of the experiment
        output_text_baseline: Baseline output (argmax decoding)
        output_text_cast: CAST output (argmax decoding)
        output_text_generated_baseline: Baseline output (generated)
        output_text_generated_cast: CAST output (generated)
        layer_idx: Layer used for steering
        alpha: Steering strength
        threshold: Similarity threshold
        metadata: Additional metadata
    """
    # Create output directory
    exp_dir = output_dir / experiment_name
    exp_dir.mkdir(parents=True, exist_ok=True)

    # Create metadata JSON
    result_data = {
        "experiment": experiment_name,
        "timestamp": datetime.now().isoformat(),
        "prompts": {
            "task_description": TASK_DESCRIPTION,
            "condition_prompt": CONDITION_PROMPT,
            "positive_example": POSITIVE_EXAMPLE,
            "negative_example": NEGATIVE_EXAMPLE
        },
        "model_config": {
            "layer_idx": layer_idx,
            "alpha": alpha,
            "threshold": threshold
        },
        "outputs": {
            "vla_baseline": output_text_generated_baseline if 'vla_baseline' not in metadata else metadata.get('vla_baseline'),
            "vla_cast": output_text_generated_cast if 'vla_cast' not in metadata else metadata.get('vla_cast'),
            "base_pretrained_baseline": metadata.get('base_baseline') if metadata else None,
            "base_pretrained_cast": metadata.get('base_cast') if metadata else None
        }
    }

    # Add additional metadata if provided
    if metadata:
        result_data.update(metadata)

    # Save metadata JSON
    with open(exp_dir / "result.json", 'w') as f:
        json.dump(result_data, f, indent=2)

    # Save text outputs separately for easy reading
    if output_text_generated_baseline:
        with open(exp_dir / "vla_baseline.txt", 'w') as f:
            f.write(output_text_generated_baseline)

    if output_text_generated_cast:
        with open(exp_dir / "vla_cast.txt", 'w') as f:
            f.write(output_text_generated_cast)

    if metadata and metadata.get('base_baseline'):
        with open(exp_dir / "base_pretrained_baseline.txt", 'w') as f:
            f.write(metadata['base_baseline'])

    if metadata and metadata.get('base_cast'):
        with open(exp_dir / "base_pretrained_cast.txt", 'w') as f:
            f.write(metadata['base_cast'])

    print(f"   💾 Results saved to: {exp_dir}")

def example_basic_cast(layer_index,
                       alpha,
                       threshold):
    """
    Basic example of Conditional Activation Steering.

    This demonstrates how to:
    - Extract condition vector from a conditional prompt
    - Compute behavior vector from positive/negative examples
    - Apply steering with a static condition vector
    """
    print("=" * 80)
    print("EXAMPLE 1: Basic Conditional Activation Steering")
    print("=" * 80)

    # Initialize model
    processor = AutoProcessor.from_pretrained("google/paligemma-3b-pt-224")
    tokenizer = AutoTokenizer.from_pretrained("google/paligemma-3b-pt-224")
    paligemma_config = _gemma.get_config('gemma_2b_lora')
    action_expert_config = _gemma.get_config('gemma_300m_lora')

    model_VLA = PaliGemmaWithExpertModel(
        paligemma_config,
        action_expert_config,
        use_adarms=[False, True],
        precision="bfloat16",
    )
    
    model_base = PaliGemmaForConditionalGeneration.from_pretrained(
        "google/paligemma-3b-pt-224",
        torch_dtype=torch.float16
    )

    model_VLA.paligemma.eval()
    model_base.eval()

    if torch.cuda.is_available():
        model_VLA.paligemma.to('cuda')
        model_base.to('cuda')

    # Create dummy images (in real use, these would be camera frames)
    dummy_image = Image.new('RGB', (224, 224), color='red')

    print(f"\n1. Extracting condition vector from: '{CONDITION_PROMPT}'")

    # Step 1: Extract condition vector
    condition_hidden_VLA = get_text_based_hidden_states(
        model=model_VLA,
        text=CONDITION_PROMPT,
        layer_idx=LAYER_IDX,
        tokenizer=tokenizer
    )
    condition_hidden_VLA = extract_hidden_vector(condition_hidden_VLA, pool_method='mean')
    condition_hidden_VLA.to('cuda')
    print(f"   Condition vector shape: {condition_hidden_VLA.shape}")

    # Step 2: Compute behavior vector from positive/negative examples
    print(f"\n2. Computing behavior vector")
    print(f"   Positive example: '{POSITIVE_EXAMPLE}'")
    print(f"   Negative example: '{NEGATIVE_EXAMPLE}'")

    positive_hidden_VLA = get_text_based_hidden_states(
        model=model_VLA,
        text=POSITIVE_EXAMPLE,
        layer_idx=layer_index,
        tokenizer=tokenizer
    )

    positive_hidden_VLA.to('cuda')

    negative_hidden_VLA = get_text_based_hidden_states(
        model=model_VLA,
        text=NEGATIVE_EXAMPLE,
        layer_idx=layer_index,
        tokenizer=tokenizer
    )

    negative_hidden_VLA.to('cuda')

    behavior_vec_VLA = compute_behavior_vector(positive_hidden_VLA, negative_hidden_VLA, pool_method='mean')
    behavior_vec_VLA.to('cuda')
    print(f"   Behavior vector shape: {behavior_vec_VLA.shape}")



    # Step 1: Extract condition vector
    condition_hidden_VLA = get_text_based_hidden_states(
        model=model_VLA,
        text=CONDITION_PROMPT,
        layer_idx=LAYER_IDX,
        tokenizer=tokenizer
    )
    condition_vec_VLA = extract_hidden_vector(condition_hidden_VLA, pool_method='mean')
    condition_vec_VLA.to('cuda')
    print(f"   Condition vector shape: {condition_vec_VLA.shape}")

    # Step 2: Compute behavior vector from positive/negative examples
    print(f"\n2. Computing behavior vector")
    print(f"   Positive example: '{POSITIVE_EXAMPLE}'")
    print(f"   Negative example: '{NEGATIVE_EXAMPLE}'")

    positive_hidden_VLA = get_text_based_hidden_states(
        model=model_VLA,
        text=POSITIVE_EXAMPLE,
        layer_idx=layer_index,
        tokenizer=tokenizer
    )

    positive_hidden_VLA.to('cuda')

    negative_hidden_VLA = get_text_based_hidden_states(
        model=model_VLA,
        text=NEGATIVE_EXAMPLE,
        layer_idx=layer_index,
        tokenizer=tokenizer
    )

    negative_hidden_VLA.to('cuda')

    behavior_vec_VLA = compute_behavior_vector(positive_hidden_VLA, negative_hidden_VLA, pool_method='mean')
    behavior_vec_VLA.to('cuda')
    print(f"   Behavior vector shape: {behavior_vec_VLA.shape}")


    # Step 3: Apply conditional steering during inference
    print(f"\n3. Setting up conditional steering hook")
    print(f"   Alpha (steering strength): {alpha}")
    print(f"   Threshold: {threshold}")
    print(f"   Layer: {layer_index}")

    if USE_CONDITIONAL:
        hook = ConditionalSteeringHook(
            model=model_VLA,
            condition_vec=condition_vec_VLA,
            behavior_vec=behavior_vec_VLA,
            alpha=alpha,
            layer_idx=layer_index,
            threshold=threshold,
            use_tanh=True
        )
    else:
        hook = SteeringHook(model=model_VLA,
                            steering_vec=behavior_vec_VLA,
                            alpha=alpha,
                            layer_idx=layer_index)

    # Run baseline inference (WITHOUT CAST - hook not registered yet)
    print(f"\n4. Running baseline inference (NO CAST)")
    
    if USE_DUMMY_IMAGE:
        inputs = processor(
            text=TASK_DESCRIPTION,
            images=dummy_image,
            return_tensors="pt"
        ).to(model_VLA.paligemma.device)
        
    else: 
        inputs = tokenizer(
        text=TASK_DESCRIPTION,
        return_tensors="pt").to(model_VLA.paligemma.device)


    with torch.no_grad():
        with torch.autocast('cuda'):
            output_ids_baseline = model_VLA.paligemma.generate(**inputs, max_new_tokens=20, do_sample=True, temperature=0.8, pad_token_id= tokenizer.eos_token_id)

    # Decode baseline outputs
    if USE_DUMMY_IMAGE:
        baseline_text_generated = processor.decode(output_ids_baseline[0], skip_special_tokens=True)

    else:
        baseline_text_generated = tokenizer.decode(output_ids_baseline[0], skip_special_tokens=True)


    print(f"   Baseline (generated): {baseline_text_generated}")

    if RUN_CAST:
        # Run inference WITH CAST
        print(f"\n5. Running inference WITH CAST")
        hook.register()  # Register hook

        with torch.no_grad():
            with torch.autocast('cuda'):
                output_ids_cast = model_VLA.paligemma.generate(**inputs, max_new_tokens=20, do_sample=True)


        # Decode CAST outputs
        if USE_DUMMY_IMAGE:
            cast_text_generated = processor.decode(output_ids_cast[0], skip_special_tokens=True)

        else:
            cast_text_generated = tokenizer.decode(output_ids_cast[0], skip_special_tokens=True)

        print(f"   VLA CAST (generated): {cast_text_generated}")

        hook.remove()

    # Now run the same experiments on the base pretrained model
    print(f"\n6. Running SAME experiments on BASE PRETRAINED model for comparison")
    print("=" * 80)

    # Create hook for base model (use same steering vector)
    if USE_CONDITIONAL:
        hook_base = ConditionalSteeringHook(
            model=PaliGemmaWithExpertModel(
                paligemma_config,
                action_expert_config,
                use_adarms=[False, True],
                precision="bfloat16",
            ),
            condition_vec=condition_vec_VLA,
            behavior_vec=behavior_vec_VLA,
            alpha=alpha,
            layer_idx=layer_index,
            threshold=threshold,
            use_tanh=True
        )
        # Replace the paligemma attribute with base model
        hook_base.model = model_base
    else:
        # For SteeringHook, we need to create a wrapper
        class BaseModelWrapper:
            def __init__(self, base_model):
                self.paligemma = base_model

        wrapper = BaseModelWrapper(model_base)
        hook_base = SteeringHook(
            model=wrapper,
            steering_vec=behavior_vec_VLA,
            alpha=alpha,
            layer_idx=layer_index
        )

    # Base model baseline (no CAST)
    print(f"\n6a. Base model baseline (NO CAST)")
    with torch.no_grad():
        with torch.autocast('cuda'):
            output_ids_base_baseline = model_base.generate(**inputs, max_new_tokens=20, do_sample=True, temperature=0.8)

    if USE_DUMMY_IMAGE:
        base_baseline_text = processor.decode(output_ids_base_baseline[0], skip_special_tokens=True)
    else:
        base_baseline_text = tokenizer.decode(output_ids_base_baseline[0], skip_special_tokens=True)

    print(f"   Base baseline (generated): {base_baseline_text}")


    if RUN_CAST:
        # Base model WITH CAST
        print(f"\n6b. Base model WITH CAST")
        hook_base.register()

        with torch.no_grad():
            with torch.autocast('cuda'):
                output_ids_base_cast = model_base.generate(**inputs, max_new_tokens=20, do_sample=True, temperature=0.8, pad_token_id=tokenizer.eos_token_id)

        if USE_DUMMY_IMAGE:
            base_cast_text = processor.decode(output_ids_base_cast[0], skip_special_tokens=True)
        else:
            base_cast_text = tokenizer.decode(output_ids_base_cast[0], skip_special_tokens=True)

        print(f"   Base CAST (generated): {base_cast_text}")

        hook_base.remove()

    # Compare ALL outputs
    print(f"\n7. COMPREHENSIVE COMPARISON:")
    print("=" * 80)
    print(f"VLA Model:")
    print(f"  Baseline: {baseline_text_generated}")
    if RUN_CAST:
        print(f"  CAST:     {cast_text_generated}")
    print(f"")
    print(f"Base Pretrained Model:")
    print(f"  Baseline: {base_baseline_text}")
    if RUN_CAST:
        print(f"  CAST:     {base_cast_text}")
    print("=" * 80)

    if RUN_CAST:
        if baseline_text_generated != cast_text_generated:
            print(f"   ✅ VLA: CAST modified the output!")
        else:
            print(f"   ⚠️  VLA: Outputs are identical")

        if base_baseline_text != base_cast_text:
            print(f"   ✅ Base: CAST modified the output!")
        else:
            print(f"   ⚠️  Base: Outputs are identical")

    # Check if VLA outputs are coherent compared to base
    print(f"\n8. Coherence Analysis:")
    if len(baseline_text_generated) < 10 or baseline_text_generated.count(' ') < 2:
        print(f"   ⚠️  WARNING: VLA baseline output appears incoherent!")
    else:
        print(f"   ✅ VLA baseline output appears coherent")

    if len(base_baseline_text) < 10 or base_baseline_text.count(' ') < 2:
        print(f"   ⚠️  WARNING: Base baseline output appears incoherent!")
    else:
        print(f"   ✅ Base baseline output appears coherent")

    # Save results
    print(f"\n9. Saving results...")
    save_text_only_results(
        output_dir=OUTPUT_DIR,
        experiment_name="basic_cast_example",
        output_text_generated_baseline=baseline_text_generated,
        output_text_generated_cast=cast_text_generated,
        layer_idx=layer_index,
        alpha=alpha,
        threshold=threshold,
        metadata={
            'base_baseline': base_baseline_text,
            'base_cast': base_cast_text
        }
    )

    print(f"\n10. All hooks removed and results saved")
    print("\n" + "=" * 80 + "\n")



def example_cast_analysis():
    """
    Analytical example showing how CAST components work.

    This demonstrates the internal mechanics:
    - Projection onto condition vector
    - Similarity computation
    - Threshold application
    """
    print("=" * 80)
    print("EXAMPLE 3: Understanding CAST Mechanics")
    print("=" * 80)

    from openpi.models_pytorch.CAST_helpers import (
        project_onto_condition,
        compute_similarity,
        apply_threshold_function,
        apply_conditional_steering
    )

    # Create synthetic data
    print("\n1. Creating synthetic hidden states and vectors")

    hidden_dim = 2048
    seq_len = 10
    batch_size = 1

    # Simulate hidden state
    hidden_state = torch.randn(batch_size, seq_len, hidden_dim)

    # Simulate condition and behavior vectors
    condition_vec = torch.randn(hidden_dim)
    condition_vec = condition_vec / condition_vec.norm()  # Normalize

    behavior_vec = torch.randn(hidden_dim)

    print(f"   Hidden state shape: {hidden_state.shape}")
    print(f"   Condition vector shape: {condition_vec.shape}")
    print(f"   Behavior vector shape: {behavior_vec.shape}")

    # Step 2: Project hidden state onto condition
    print("\n2. Projecting hidden state onto condition vector")
    projection = project_onto_condition(hidden_state, condition_vec, use_tanh=True)
    print(f"   Projection shape: {projection.shape}")
    print(f"   Projection range: [{projection.min():.4f}, {projection.max():.4f}]")

    # Step 3: Compute similarity
    print("\n3. Computing cosine similarity between hidden state and projection")
    similarity = compute_similarity(hidden_state, projection)
    print(f"   Similarity shape: {similarity.shape}")
    print(f"   Similarity range: [{similarity.min():.4f}, {similarity.max():.4f}]")
    print(f"   Similarity values per token: {similarity[0].tolist()}")

    # Step 4: Apply threshold
    print("\n4. Applying threshold function (threshold=0.5)")
    mask = apply_threshold_function(similarity, threshold=0.5)
    print(f"   Mask shape: {mask.shape}")
    print(f"   Mask values: {mask[0].tolist()}")
    print(f"   Number of activated positions: {mask.sum().item()}/{seq_len}")

    # Step 5: Apply full conditional steering
    print("\n5. Applying full conditional steering (alpha=2.0)")
    modified_hidden = apply_conditional_steering(
        hidden_state=hidden_state,
        condition_vec=condition_vec,
        behavior_vec=behavior_vec,
        alpha=2.0,
        threshold=0.5,
        use_tanh=True
    )
    print(f"   Modified hidden state shape: {modified_hidden.shape}")

    # Analyze the change
    change = (modified_hidden - hidden_state).abs().mean()
    print(f"   Mean absolute change: {change.item():.6f}")

    print("\n" + "=" * 80 + "\n")


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("CONDITIONAL ACTIVATION STEERING (CAST) EXAMPLES")
    print("=" * 80 + "\n")

    # Run examples
    example_basic_cast(layer_index=LAYER_IDX,
                        alpha=ALPHA,
                        threshold=THRESHOLD)

    example_cast_analysis()

    print("\n" + "=" * 80)
    print("ALL EXAMPLES COMPLETED")
    print("=" * 80 + "\n")