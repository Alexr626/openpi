"""
Example script demonstrating CAST with actual robot camera frames.

This script demonstrates CAST using real frames from the PiPER robot's 3-camera setup
(top, left wrist, right wrist) during an object transportation task.

Outputs are saved to: /home/alex/dev/piper_bimanual/CAST_results/
"""

from transformers import AutoProcessor, AutoTokenizer
from PIL import Image
import torch
import numpy as np
from pathlib import Path
import json
from datetime import datetime
import openpi.models.gemma as _gemma
from openpi.models_pytorch.gemma_pytorch import PaliGemmaWithExpertModel
from openpi.models_pytorch.Activation_Engineering_helpers import (
    get_text_based_hidden_states,
    extract_hidden_vector,
    compute_behavior_vector,
    ConditionalSteeringHook
)


# Configuration
FRAMES_DIR = Path("/home/alex/dev/piper_bimanual/frames_20251118_173707_573706")
OUTPUT_DIR = Path("/home/alex/dev/piper_bimanual/CAST_results")
LAYER_IDX = 6


def load_camera_frames(timestep: int, frames_dir: Path):
    """
    Load frames from all 3 cameras for a given timestep.

    Args:
        timestep: Frame number to load
        frames_dir: Path to frames directory

    Returns:
        Dict with 'top', 'left', 'right' PIL Images
    """
    frames = {}

    # Load top camera
    top_path = frames_dir / "top_camera" / f"frame_{timestep:06d}.png"
    if top_path.exists():
        frames['top'] = Image.open(top_path).convert('RGB')

    # Load left wrist camera
    left_path = frames_dir / "left_camera" / f"frame_{timestep:06d}.png"
    if left_path.exists():
        frames['left'] = Image.open(left_path).convert('RGB')

    # Load right wrist camera
    right_path = frames_dir / "right_camera" / f"frame_{timestep:06d}.png"
    if right_path.exists():
        frames['right'] = Image.open(right_path).convert('RGB')

    return frames


def concatenate_camera_frames(frames_dict):
    """
    Concatenate 3 camera frames horizontally to create single image for PaliGemma.

    Pi 0.5 typically processes multiple camera views by concatenating them.
    Order: [top | left_wrist | right_wrist]

    Args:
        frames_dict: Dict with 'top', 'left', 'right' PIL Images

    Returns:
        Single PIL Image with all cameras concatenated horizontally
    """
    # Convert to numpy arrays
    top_np = np.array(frames_dict['top'])
    left_np = np.array(frames_dict['left'])
    right_np = np.array(frames_dict['right'])

    # Concatenate horizontally [top | left | right]
    concatenated = np.concatenate([top_np, left_np, right_np], axis=1)

    # Convert back to PIL
    return Image.fromarray(concatenated)


def create_output_directory(timestep: int, experiment_name: str, use_cast: bool):
    """
    Create structured output directory for saving results.

    Args:
        timestep: Frame timestep number
        experiment_name: Name of the experiment
        use_cast: Whether CAST was used

    Returns:
        Path to output directory
    """
    cast_suffix = "with_CAST" if use_cast else "without_CAST"
    output_path = OUTPUT_DIR / experiment_name / f"timestep_{timestep:06d}_{cast_suffix}"
    output_path.mkdir(parents=True, exist_ok=True)
    return output_path


def save_results(output_dir: Path, timestep: int, frames_dict: dict,
                 concatenated_image: Image.Image, output_text: str,
                 use_cast: bool, metadata: dict = None):
    """
    Save all results to structured output directory.

    Args:
        output_dir: Path to output directory
        timestep: Frame timestep number
        frames_dict: Dict with individual camera frames
        concatenated_image: Concatenated image used for inference
        output_text: Decoded text output from PaliGemma
        use_cast: Whether CAST was used
        metadata: Additional metadata to save
    """
    # Save individual camera frames
    frames_dict['top'].save(output_dir / "camera_top.png")
    frames_dict['left'].save(output_dir / "camera_left.png")
    frames_dict['right'].save(output_dir / "camera_right.png")

    # Save concatenated image
    concatenated_image.save(output_dir / "concatenated_cameras.png")

    # Create metadata JSON
    result_data = {
        "timestep": timestep,
        "use_cast": use_cast,
        "timestamp": datetime.now().isoformat(),
        "prompts": {
            "task_instruction": TASK_DESCRIPTION,
            "condition_prompt": CONDITION_PROMPT,
            "positive_example": POSITIVE_EXAMPLE,
            "negative_example": NEGATIVE_EXAMPLE
        },
        "model_config": {
            "layer_idx": LAYER_IDX,
            "alpha": 3 if use_cast else None,
            "threshold": 0.3 if use_cast else None
        },
        "output": {
            "decoded_text": output_text
        }
    }

    # Add additional metadata if provided
    if metadata:
        result_data.update(metadata)

    # Save metadata JSON
    with open(output_dir / "result.json", 'w') as f:
        json.dump(result_data, f, indent=2)

    # Save output text separately for easy reading
    with open(output_dir / "output.txt", 'w') as f:
        f.write(output_text)


def example_cast_with_single_timestep(timestep=0):
    """
    Demonstrate CAST with frames from a single timestep.

    Shows how to:
    1. Load frames from 3 cameras
    2. Concatenate them for PaliGemma input
    3. Apply CAST
    4. Get textual output
    """
    print("=" * 80)
    print(f"EXAMPLE 1: CAST with Robot Frames (Timestep {timestep})")
    print("=" * 80)

    # Initialize model
    print("\n1. Initializing PaliGemma model...")
    processor = AutoProcessor.from_pretrained("google/paligemma-3b-pt-224")
    tokenizer = AutoTokenizer.from_pretrained("google/paligemma-3b-pt-224")
    paligemma_config = _gemma.get_config('gemma_2b_lora')
    action_expert_config = _gemma.get_config('gemma_300m_lora')

    model = PaliGemmaWithExpertModel(
        paligemma_config,
        action_expert_config,
        use_adarms=[False, True],
        precision="bfloat16"
    )
    print("   ✅ Model initialized")

    # Load camera frames
    print(f"\n2. Loading camera frames from timestep {timestep}...")
    frames = load_camera_frames(timestep, FRAMES_DIR)
    print(f"   Loaded {len(frames)} camera views: {list(frames.keys())}")

    # Concatenate frames for PaliGemma input
    print("\n3. Concatenating camera frames...")
    concatenated_image = concatenate_camera_frames(frames)
    print(f"   Concatenated image size: {concatenated_image.size}")
    print(f"   (Original camera resolution × 3)")

    # Compute CAST vectors
    print("\n4. Computing CAST vectors...")
    print(f"   Condition: '{CONDITION_PROMPT}'")
    print(f"   Positive: '{POSITIVE_EXAMPLE}'")
    print(f"   Negative: '{NEGATIVE_EXAMPLE}'")

    # Compute condition vector (text-only)
    condition_hidden = get_text_based_hidden_states(
        model=model,
        text=CONDITION_PROMPT,
        layer_idx=LAYER_IDX,
        tokenizer=tokenizer
    )
    condition_vec = extract_hidden_vector(condition_hidden, pool_method='mean')
    print(f"   ✅ Condition vector: {condition_vec.shape}")

    # Compute behavior vector (text-only)
    positive_hidden = get_text_based_hidden_states(
        model=model,
        text=POSITIVE_EXAMPLE,
        layer_idx=LAYER_IDX,
        tokenizer=tokenizer
    )
    negative_hidden = get_text_based_hidden_states(
        model=model,
        text=NEGATIVE_EXAMPLE,
        layer_idx=LAYER_IDX,
        tokenizer=tokenizer
    )
    behavior_vec = compute_behavior_vector(positive_hidden, negative_hidden, pool_method='mean')
    print(f"   ✅ Behavior vector: {behavior_vec.shape}")

    # Run inference WITHOUT CAST (baseline)
    print(f"\n5. Running baseline inference (NO CAST)...")
    print(f"   Task: '{TASK_DESCRIPTION}'")

    inputs = processor(
        text=TASK_DESCRIPTION,
        images=concatenated_image,
        return_tensors="pt"
    ).to(model.paligemma.device)

    with torch.no_grad():
        output_baseline = model.paligemma(**inputs)

    baseline_text = processor.decode(
        output_baseline.logits[0].argmax(dim=-1),
        skip_special_tokens=True
    )
    print(f"   Output (baseline): {baseline_text}")

    # Run inference WITH CAST
    print(f"\n6. Running inference WITH CAST...")
    print(f"   Alpha: 2.0, Threshold: 0.5, Layer: {LAYER_IDX}")

    hook = ConditionalSteeringHook(
        model=model,
        condition_vec=condition_vec,
        behavior_vec=behavior_vec,
        alpha=2.0,
        layer_idx=LAYER_IDX,
        threshold=0.5
    )
    hook.register()

    with torch.no_grad():
        output_cast = model.paligemma(**inputs)

    cast_text = processor.decode(
        output_cast.logits[0].argmax(dim=-1),
        skip_special_tokens=True
    )
    print(f"   Output (with CAST): {cast_text}")

    hook.remove()

    # Compare
    print(f"\n7. Comparison:")
    print(f"   Baseline: {baseline_text}")
    print(f"   With CAST: {cast_text}")
    if baseline_text != cast_text:
        print(f"   ✅ CAST modified the output!")
    else:
        print(f"   ⚠️  Outputs are identical (try adjusting alpha/threshold)")

    # Save results
    print(f"\n8. Saving results...")

    # Save baseline results
    output_dir_baseline = create_output_directory(timestep, "single_timestep_demo", use_cast=False)
    save_results(
        output_dir=output_dir_baseline,
        timestep=timestep,
        frames_dict=frames,
        concatenated_image=concatenated_image,
        output_text=baseline_text,
        use_cast=False
    )
    print(f"   ✅ Baseline results saved to: {output_dir_baseline}")

    # Save CAST results
    output_dir_cast = create_output_directory(timestep, "single_timestep_demo", use_cast=True)
    save_results(
        output_dir=output_dir_cast,
        timestep=timestep,
        frames_dict=frames,
        concatenated_image=concatenated_image,
        output_text=cast_text,
        use_cast=True
    )
    print(f"   ✅ CAST results saved to: {output_dir_cast}")

    print("\n" + "=" * 80 + "\n")


def example_cast_over_episode():
    """
    Demonstrate CAST over 10 timesteps covering the entire episode.

    Shows how CAST affects model outputs as the robot progresses through
    the object transportation task.
    """
    print("=" * 80)
    print("EXAMPLE 2: CAST Over Full Episode (10 Timesteps)")
    print("=" * 80)

    # Initialize model
    print("\n1. Initializing model...")
    processor = AutoProcessor.from_pretrained("google/paligemma-3b-pt-224")
    tokenizer = AutoTokenizer.from_pretrained("google/paligemma-3b-pt-224")
    paligemma_config = _gemma.get_config('gemma_2b_lora')
    action_expert_config = _gemma.get_config('gemma_300m_lora')

    model = PaliGemmaWithExpertModel(
        paligemma_config,
        action_expert_config,
        use_adarms=[False, True],
        precision="bfloat16"
    )
    print("   ✅ Model initialized")

    # Compute CAST vectors (once)
    print("\n2. Computing CAST vectors...")
    condition_hidden = get_text_based_hidden_states(
        model, CONDITION_PROMPT, LAYER_IDX, tokenizer
    )
    condition_vec = extract_hidden_vector(condition_hidden, pool_method='mean')

    positive_hidden = get_text_based_hidden_states(
        model, POSITIVE_EXAMPLE, LAYER_IDX, tokenizer
    )
    negative_hidden = get_text_based_hidden_states(
        model, NEGATIVE_EXAMPLE, LAYER_IDX, tokenizer
    )
    behavior_vec = compute_behavior_vector(positive_hidden, negative_hidden, pool_method='mean')
    print(f"   ✅ Vectors computed")

    # Setup CAST hook
    print("\n3. Registering CAST hook...")
    hook = ConditionalSteeringHook(
        model=model,
        condition_vec=condition_vec,
        behavior_vec=behavior_vec,
        alpha=2.0,
        layer_idx=LAYER_IDX,
        threshold=0.5
    )
    hook.register()
    print(f"   ✅ Hook registered (alpha=2.0, threshold=0.5)")

    # Select 10 evenly spaced timesteps from ~517 total frames
    # Timesteps: 0, 50, 100, 150, 200, 250, 300, 350, 400, 450
    timesteps = [0, 50, 100, 150, 200, 250, 300, 350, 400, 450]

    print(f"\n4. Running inference over {len(timesteps)} timesteps...")
    print(f"   Timesteps: {timesteps}")
    print(f"   Task instruction: '{TASK_DESCRIPTION}'")
    print("\n" + "-" * 80)

    results = []

    for i, timestep in enumerate(timesteps):
        print(f"\nTimestep {timestep} (#{i+1}/10):")

        # Load and concatenate frames
        frames = load_camera_frames(timestep, FRAMES_DIR)
        if len(frames) < 3:
            print(f"  ⚠️  Warning: Only {len(frames)} cameras available")
            continue

        concatenated_image = concatenate_camera_frames(frames)

        # Run inference with CAST
        inputs = processor(
            text=TASK_DESCRIPTION,
            images=concatenated_image,
            return_tensors="pt"
        ).to(model.paligemma.device)

        with torch.no_grad():
            output = model.paligemma(**inputs)

        # Decode output
        output_text = processor.decode(
            output.logits[0].argmax(dim=-1),
            skip_special_tokens=True
        )

        print(f"  Output: {output_text}")

        # Save results for this timestep
        output_dir = create_output_directory(timestep, "episode_progression", use_cast=True)
        save_results(
            output_dir=output_dir,
            timestep=timestep,
            frames_dict=frames,
            concatenated_image=concatenated_image,
            output_text=output_text,
            use_cast=True,
            metadata={"sequence_position": i + 1, "total_timesteps": len(timesteps)}
        )

        results.append({
            'timestep': timestep,
            'output': output_text,
            'output_dir': str(output_dir)
        })

    hook.remove()

    # Summary
    print("\n" + "-" * 80)
    print("\n5. Summary of outputs over episode:")
    print("\nTimestep | Output")
    print("-" * 80)
    for r in results:
        # Truncate long outputs for summary
        output_short = r['output'][:60] + "..." if len(r['output']) > 60 else r['output']
        print(f"{r['timestep']:8d} | {output_short}")

    # Save episode summary
    print(f"\n6. Saving episode summary...")
    summary_path = OUTPUT_DIR / "episode_progression" / "summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, 'w') as f:
        json.dump({
            "experiment": "episode_progression",
            "timesteps": timesteps,
            "results": results,
            "prompts": {
                "task_instruction": TASK_DESCRIPTION,
                "condition_prompt": CONDITION_PROMPT,
                "positive_example": POSITIVE_EXAMPLE,
                "negative_example": NEGATIVE_EXAMPLE
            },
            "timestamp": datetime.now().isoformat()
        }, f, indent=2)
    print(f"   ✅ Episode summary saved to: {summary_path}")

    print("\n" + "=" * 80)
    print("ANALYSIS:")
    print("  - Check if outputs change as robot progresses through task")
    print("  - Early frames (0-150): Robot reaching for ball")
    print("  - Mid frames (200-300): Robot grasping/transporting")
    print("  - Late frames (350-450): Robot placing ball in basket")
    print("  - CAST should adapt behavior based on visual state")
    print(f"  - All results saved to: {OUTPUT_DIR / 'episode_progression'}")
    print("=" * 80 + "\n")


def example_comparison_with_without_cast():
    """
    Compare outputs with and without CAST at key timesteps.

    This shows the direct effect of CAST by comparing baseline vs. steered outputs.
    """
    print("=" * 80)
    print("EXAMPLE 3: Baseline vs. CAST Comparison")
    print("=" * 80)

    # Initialize model
    print("\n1. Initializing model...")
    processor = AutoProcessor.from_pretrained("google/paligemma-3b-pt-224")
    tokenizer = AutoTokenizer.from_pretrained("google/paligemma-3b-pt-224")
    paligemma_config = _gemma.get_config('gemma_2b_lora')
    action_expert_config = _gemma.get_config('gemma_300m_lora')

    model = PaliGemmaWithExpertModel(
        paligemma_config,
        action_expert_config,
        use_adarms=[False, True],
        precision="bfloat16"
    )

    # Compute CAST vectors
    print("\n2. Computing CAST vectors...")
    condition_hidden = get_text_based_hidden_states(
        model, CONDITION_PROMPT, LAYER_IDX, tokenizer
    )
    condition_vec = extract_hidden_vector(condition_hidden, pool_method='mean')

    positive_hidden = get_text_based_hidden_states(
        model, POSITIVE_EXAMPLE, LAYER_IDX, tokenizer
    )
    negative_hidden = get_text_based_hidden_states(
        model, NEGATIVE_EXAMPLE, LAYER_IDX, tokenizer
    )
    behavior_vec = compute_behavior_vector(positive_hidden, negative_hidden, pool_method='mean')

    # Setup CAST hook
    hook = ConditionalSteeringHook(
        model=model,
        condition_vec=condition_vec,
        behavior_vec=behavior_vec,
        alpha=2.0,
        layer_idx=LAYER_IDX,
        threshold=0.5
    )

    # Test at 3 key timesteps: beginning, middle, end
    key_timesteps = [0, 200, 450]
    labels = ["Beginning (reaching)", "Middle (grasping/transporting)", "End (placing)"]

    print("\n3. Comparing baseline vs. CAST at key timesteps...")
    print("\n" + "-" * 80)

    for timestep, label in zip(key_timesteps, labels):
        print(f"\n{label} - Timestep {timestep}:")

        # Load frames
        frames = load_camera_frames(timestep, FRAMES_DIR)
        if len(frames) < 3:
            continue
        concatenated_image = concatenate_camera_frames(frames)

        inputs = processor(
            text=TASK_DESCRIPTION,
            images=concatenated_image,
            return_tensors="pt"
        ).to(model.paligemma.device)

        # Baseline (no CAST)
        with torch.no_grad():
            output_baseline = model.paligemma(**inputs)
        baseline_text = processor.decode(
            output_baseline.logits[0].argmax(dim=-1),
            skip_special_tokens=True
        )

        # With CAST
        hook.register()
        with torch.no_grad():
            output_cast = model.paligemma(**inputs)
        cast_text = processor.decode(
            output_cast.logits[0].argmax(dim=-1),
            skip_special_tokens=True
        )
        hook.remove()

        # Display comparison
        print(f"  Baseline: {baseline_text}")
        print(f"  With CAST: {cast_text}")

        if baseline_text != cast_text:
            print(f"  ✅ CAST modified output")
        else:
            print(f"  ⚠️  No change")

        # Save baseline results
        output_dir_baseline = create_output_directory(timestep, "baseline_vs_cast_comparison", use_cast=False)
        save_results(
            output_dir=output_dir_baseline,
            timestep=timestep,
            frames_dict=frames,
            concatenated_image=concatenated_image,
            output_text=baseline_text,
            use_cast=False,
            metadata={"phase": label}
        )

        # Save CAST results
        output_dir_cast = create_output_directory(timestep, "baseline_vs_cast_comparison", use_cast=True)
        save_results(
            output_dir=output_dir_cast,
            timestep=timestep,
            frames_dict=frames,
            concatenated_image=concatenated_image,
            output_text=cast_text,
            use_cast=True,
            metadata={"phase": label}
        )

    print(f"\n4. All comparison results saved to: {OUTPUT_DIR / 'baseline_vs_cast_comparison'}")
    print("\n" + "=" * 80 + "\n")


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("CAST WITH REAL ROBOT CAMERA FRAMES")
    print("Using frames from PiPER object transportation episode")
    print("=" * 80 + "\n")

    # Run examples
    try:
        example_cast_with_single_timestep(timestep=0)
    except Exception as e:
        print(f"Example 1 failed: {e}")
        import traceback
        traceback.print_exc()

    try:
        example_cast_over_episode()
    except Exception as e:
        print(f"Example 2 failed: {e}")
        import traceback
        traceback.print_exc()

    try:
        example_comparison_with_without_cast()
    except Exception as e:
        print(f"Example 3 failed: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "=" * 80)
    print("ALL EXAMPLES COMPLETED")
    print("=" * 80 + "\n")
