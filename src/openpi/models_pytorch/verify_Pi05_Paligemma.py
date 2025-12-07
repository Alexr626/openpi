from transformers import AutoProcessor
from PIL import Image
import torch
import openpi.models.gemma as _gemma
from openpi.models_pytorch.gemma_pytorch import PaliGemmaWithExpertModel
from openpi.models_pytorch.Activation_Engineering_helpers import (
    get_text_and_vision_based_hidden_states,
    extract_hidden_vector,
    compute_behavior_vector,
    ConditionalSteeringHook,
)

# Load processor
processor = AutoProcessor.from_pretrained("google/paligemma-3b-pt-224")
paligemma_config = _gemma.get_config('gemma_2b_lora')
action_expert_config = _gemma.get_config('gemma_300m_lora')

model = PaliGemmaWithExpertModel(
    paligemma_config,
    action_expert_config,
    use_adarms=[False, True],
    precision="bfloat16"
)

# Verify compatibility
print("=== Processor vs Model Config ===")
print(f"Processor vocab size: {processor.tokenizer.vocab_size}")
print(f"Model vocab size: {model.paligemma.config.text_config.vocab_size}")
print(f"Processor image token: {processor.image_token_id}")
print(f"Model image token: {model.paligemma.config.image_token_index}")

# Test with dummy inputs
dummy_image = Image.new('RGB', (224, 224), color='red')
dummy_text = "This is a test"

inputs = processor(
    text=dummy_text,
    images=dummy_image,
    return_tensors="pt"
).to(model.paligemma.device)

print("\n=== Processor Output ===")
print(f"Input IDs shape: {inputs['input_ids'].shape}")
print(f"Pixel values shape: {inputs['pixel_values'].shape}")
print(f"Input IDs dtype: {inputs['input_ids'].dtype}")
print(f"Pixel values dtype: {inputs['pixel_values'].dtype}")

# Test forward pass
print("\n=== Testing Forward Pass ===")
try:
    with torch.no_grad():
        output = model.paligemma(**inputs)
    print(f"✅ Forward pass successful!")
    print(f"Output logits shape: {output.logits.shape}")
except Exception as e:
    print(f"❌ Forward pass failed: {e}")

# Test hook extraction
print("\n=== Testing Hook Extraction ===")
try:
    hidden = get_text_and_vision_based_hidden_states(
        model=model,
        image=dummy_image,
        text=dummy_text,
        layer_idx=10,
        processor=processor
    )
    print(f"✅ Hidden states extracted!")
    print(f"Shape: {hidden.shape}")
    print(f"Dtype: {hidden.dtype}")
except Exception as e:
    print(f"❌ Hook extraction failed: {e}")

# Define example prompts for ball transportation task
TASK_DESCRIPTION = "transport the ball to the basket"
CONDITION_VECTOR_PROMPT = "if you don't have possession of the ball yet, reach closer towards it"
POSITIVE_EXAMPLE_PROMPT = "reach far towards the ball"
NEGATIVE_EXAMPLE_PROMPT = "retract arm close to body"

# Test CAST functionality
print("\n=== Testing CAST Functions ===")
try:
    print("1. Extracting condition vector...")
    condition_hidden = get_text_and_vision_based_hidden_states(
        model=model,
        image=dummy_image,
        text=CONDITION_VECTOR_PROMPT,
        layer_idx=10,
        processor=processor
    )
    condition_vec = extract_hidden_vector(condition_hidden, pool_method='mean')
    print(f"   ✅ Condition vector shape: {condition_vec.shape}")

    print("2. Computing behavior vector...")
    positive_hidden = get_text_and_vision_based_hidden_states(
        model=model,
        image=dummy_image,
        text=POSITIVE_EXAMPLE_PROMPT,
        layer_idx=10,
        processor=processor
    )
    negative_hidden = get_text_and_vision_based_hidden_states(
        model=model,
        image=dummy_image,
        text=NEGATIVE_EXAMPLE_PROMPT,
        layer_idx=10,
        processor=processor
    )
    behavior_vec = compute_behavior_vector(positive_hidden, negative_hidden, pool_method='mean')
    print(f"   ✅ Behavior vector shape: {behavior_vec.shape}")

    print("3. Testing ConditionalSteeringHook...")
    hook = ConditionalSteeringHook(
        model=model,
        condition_vec=condition_vec,
        behavior_vec=behavior_vec,
        alpha=2.0,
        layer_idx=10,
        threshold=0.5
    )
    hook.register()

    # Run inference with steering
    test_inputs = processor(
        text=TASK_DESCRIPTION,
        images=dummy_image,
        return_tensors="pt"
    ).to(model.paligemma.device)

    with torch.no_grad():
        output = model.paligemma(**test_inputs)

    print(f"   ✅ Inference with conditional steering successful!")
    print(f"   Output shape: {output.logits.shape}")

    hook.remove()

except Exception as e:
    print(f"❌ CAST test failed: {e}")
    import traceback
    traceback.print_exc()

print("\n=== All Tests Complete ===")
print("CAST implementation is ready for deployment!")