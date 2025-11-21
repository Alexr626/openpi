# Conditional Activation Steering (CAST) for PaliGemma

This directory contains an implementation of **Conditional Activation Steering (CAST)** as presented in "Programming Refusal with Conditional Activation Steering" by Lee et al., adapted for the PaliGemma vision-language model used in the π₀ VLA family.

## Overview

CAST enables dynamic behavioral modifications in language models based on conditional inputs (text + vision). Unlike traditional activation steering which applies the same intervention unconditionally, CAST only applies the steering vector when specific conditions are detected in the input.

### Key Formula

The modified hidden state `h'` is computed as:

```
h' ← h + f(sim(h, proj_c h)) · α · v
```

Where:
- `h` = original hidden state
- `c` = condition vector (from conditional prompt + vision)
- `v` = behavior vector (from positive - negative examples)
- `α` = scaling factor (steering strength)
- `f` = threshold function (binary switch)
- `proj_c h` = projection of h onto c
- `sim` = cosine similarity

## Implementation Files

### `CAST_helpers.py`

Core implementation with the following components:

#### 1. Hidden State Extraction
```python
get_text_and_vision_based_hidden_states(model, image, text, layer_idx, processor)
```
- Extracts hidden states from PaliGemma at a specific layer
- Supports both text and vision inputs
- Uses forward hooks to capture intermediate activations

#### 2. Vector Extraction
```python
extract_condition_vector(condition_hidden_states, pool_method='mean')
```
- Extracts condition vector from hidden states
- Pooling options: 'mean', 'last', 'max'
- Returns: `[hidden_dim]`

```python
compute_behavior_vector(positive_hidden_states, negative_hidden_states, pool_method='mean')
```
- Computes steering vector as difference: `positive - negative`
- Returns: `[hidden_dim]`

#### 3. CAST Core Functions

**Projection onto Condition Vector:**
```python
project_onto_condition(hidden_state, condition_vec, use_tanh=True)
```
- Projects hidden state onto condition direction: `proj_c(h) = (c·h / c·c) * c`
- Optional tanh non-linearity for stability
- Returns: same shape as `hidden_state`

**Similarity Computation:**
```python
compute_similarity(hidden_state, projection)
```
- Computes cosine similarity: `sim(h, g) = h·g / (|h||g|)`
- Returns: `[batch, seq_len]`

**Threshold Function:**
```python
apply_threshold_function(similarity, threshold=0.5)
```
- Binary switch: 1 if similarity > threshold, else 0
- Returns: `[batch, seq_len]`

**Full CAST Application:**
```python
apply_conditional_steering(hidden_state, condition_vec, behavior_vec, alpha, threshold=0.5, use_tanh=True)
```
- Applies complete CAST formula
- Returns: modified hidden state `h'`

#### 4. Hook Classes

**Static Conditional Steering:**
```python
ConditionalSteeringHook(model, condition_vec, behavior_vec, alpha, layer_idx, threshold=0.5, use_tanh=True)
```
- Uses pre-computed condition vector
- Good for testing and static conditions

**Dynamic Conditional Steering:**
```python
DynamicConditionalSteeringHook(model, processor, condition_text, behavior_vec, alpha, layer_idx, ...)
```
- Recomputes condition vector at each forward pass
- Uses `update_images()` to provide new camera frames
- **Recommended for robot deployment** - adapts to changing visual state

## Usage Examples

### Basic Usage (Static Condition)

```python
from openpi.models_pytorch.CAST_helpers import (
    get_text_and_vision_based_hidden_states,
    extract_condition_vector,
    compute_behavior_vector,
    ConditionalSteeringHook
)

# 1. Extract condition vector
condition_hidden = get_text_and_vision_based_hidden_states(
    model=model,
    image=camera_frame,
    text="if you don't have possession of the ball yet",
    layer_idx=15,
    processor=processor
)
condition_vec = extract_condition_vector(condition_hidden, pool_method='mean')

# 2. Compute behavior vector
positive_hidden = get_text_and_vision_based_hidden_states(
    model, camera_frame, "reach far towards the ball", 15, processor
)
negative_hidden = get_text_and_vision_based_hidden_states(
    model, camera_frame, "retract arm close to body", 15, processor
)
behavior_vec = compute_behavior_vector(positive_hidden, negative_hidden)

# 3. Apply conditional steering
hook = ConditionalSteeringHook(
    model=model,
    condition_vec=condition_vec,
    behavior_vec=behavior_vec,
    alpha=2.0,
    layer_idx=15,
    threshold=0.5
)
hook.register()

# Run inference
outputs = model.paligemma(**inputs)

# Clean up
hook.remove()
```

### Advanced Usage (Dynamic Condition for Robot Deployment)

```python
from openpi.models_pytorch.CAST_helpers import DynamicConditionalSteeringHook

# 1. Pre-compute behavior vector (static - done once)
positive_hidden = get_text_and_vision_based_hidden_states(
    model, dummy_image, "reach far", 15, processor
)
negative_hidden = get_text_and_vision_based_hidden_states(
    model, dummy_image, "retract close", 15, processor
)
behavior_vec = compute_behavior_vector(positive_hidden, negative_hidden)

# 2. Create dynamic hook
hook = DynamicConditionalSteeringHook(
    model=model,
    processor=processor,
    condition_text="if you don't have possession of the ball yet",
    behavior_vec=behavior_vec,
    alpha=2.0,
    layer_idx=15,
    threshold=0.5
)
hook.register()

# 3. Robot control loop
for timestep in range(num_steps):
    # Get current camera frames
    current_frames = get_camera_frames()  # Your camera capture function

    # Update hook with new frames (condition vector recomputed internally)
    hook.update_images(current_frames)

    # Run inference - steering adapts to current visual state
    outputs = model.paligemma(**inputs)
    actions = process_outputs(outputs)

    # Execute actions
    robot.execute(actions)

# Clean up
hook.remove()
```

## Integration with π₀ Deployment

To use CAST with π₀ on PiPER:

### 1. Modify `deploy_pi.py`

Add CAST initialization before the deployment loop:

```python
from openpi.models_pytorch.CAST_helpers import (
    get_text_and_vision_based_hidden_states,
    compute_behavior_vector,
    DynamicConditionalSteeringHook
)

# After model initialization
workspace = get_workspace(...)

# Define task-specific prompts
CONDITION_PROMPT = "if you don't have possession of the ball yet, reach closer towards it"
POSITIVE_EXAMPLE = "reach far towards the ball"
NEGATIVE_EXAMPLE = "retract arm close to body"
STEERING_LAYER = 15  # Choose based on experimentation
ALPHA = 2.0  # Steering strength
THRESHOLD = 0.5  # Similarity threshold

# Compute behavior vector (once at initialization)
dummy_image = Image.new('RGB', (224, 224))
processor = workspace.policy.model.processor

positive_hidden = get_text_and_vision_based_hidden_states(
    model=workspace.policy.model,
    image=dummy_image,
    text=POSITIVE_EXAMPLE,
    layer_idx=STEERING_LAYER,
    processor=processor
)

negative_hidden = get_text_and_vision_based_hidden_states(
    model=workspace.policy.model,
    image=dummy_image,
    text=NEGATIVE_EXAMPLE,
    layer_idx=STEERING_LAYER,
    processor=processor
)

behavior_vec = compute_behavior_vector(positive_hidden, negative_hidden)

# Create and register dynamic steering hook
cast_hook = DynamicConditionalSteeringHook(
    model=workspace.policy.model,
    processor=processor,
    condition_text=CONDITION_PROMPT,
    behavior_vec=behavior_vec,
    alpha=ALPHA,
    layer_idx=STEERING_LAYER,
    threshold=THRESHOLD
)
cast_hook.register()
```

### 2. Update in Deployment Loop

```python
# In the main deployment loop (in deploy_pi.py)
while True:
    # Get current observations
    camera_frames = get_camera_observations()  # Extract from observations dict

    # Update CAST hook with current frames
    cast_hook.update_images(camera_frames)

    # Get actions (CAST steering applied automatically during forward pass)
    actions = workspace.policy.get_actions(observations)

    # Execute actions
    # ... rest of deployment code
```

### 3. Clean up on Exit

```python
# Before exiting
cast_hook.remove()
```

## Hyperparameter Tuning Guide

### Critical Hyperparameters

1. **`alpha` (Steering Strength)**
   - Range: 0.5 - 5.0
   - Higher values = stronger behavioral modification
   - Start low (1.0-2.0) and increase if behavior change is insufficient
   - Too high can destabilize the model

2. **`threshold` (Similarity Threshold)**
   - Range: 0.0 - 1.0
   - Default: 0.5
   - Higher values = more selective steering (only when condition strongly met)
   - Lower values = more aggressive steering (applied more often)

3. **`layer_idx` (Which Layer to Steer)**
   - Early layers (0-8): More general features, less task-specific
   - Middle layers (9-17): Good balance, **recommended starting point**
   - Late layers (18-26): More task-specific, may be less effective for behavioral changes

4. **`pool_method` (Vector Extraction)**
   - `'mean'`: Average across sequence (default, most stable)
   - `'last'`: Use last token only (good for causal models)
   - `'max'`: Max pooling (captures strongest activations)

### Experimentation Tips

1. **Start Simple:**
   - Use layer 15, alpha=2.0, threshold=0.5
   - Test with clear positive/negative examples

2. **Monitor Similarity Values:**
   - Add logging to see when steering is actually applied
   - Check if threshold is too high/low based on similarity distribution

3. **A/B Testing:**
   - Compare performance with CAST on vs. off
   - Measure task success rate, smoothness, safety

4. **Visual Verification:**
   - Log camera frames when steering is applied
   - Verify condition is being detected correctly

## Troubleshooting

### Issue: No Behavioral Change

**Possible Causes:**
- Threshold too high (steering never activated)
- Alpha too low (steering too weak)
- Wrong layer chosen
- Condition vector not aligned with actual condition

**Solutions:**
- Lower threshold to 0.3
- Increase alpha to 3.0-5.0
- Try different layers (10-20)
- Add logging to check similarity values

### Issue: Unstable Behavior

**Possible Causes:**
- Alpha too high
- use_tanh=False causing large projections
- Behavior vector too large

**Solutions:**
- Reduce alpha to 0.5-1.0
- Enable use_tanh=True
- Normalize behavior vector

### Issue: Condition Not Detected

**Possible Causes:**
- Vision input not properly processed
- Condition text too vague
- Wrong pooling method

**Solutions:**
- Verify camera frames are correct format
- Make condition text more specific
- Try 'last' or 'max' pooling instead of 'mean'

## Testing

Run the example script to verify installation:

```bash
cd /home/alex/dev/piper_bimanual
uv run python openpi/src/openpi/models_pytorch/example_CAST_usage.py
```

This will run three examples:
1. Basic CAST with static condition
2. Dynamic CAST with changing images
3. Analysis of CAST mechanics

## References

- Paper: "Programming Refusal with Conditional Activation Steering" by Lee et al.
- PaliGemma: https://huggingface.co/google/paligemma-3b-pt-224
- π₀ Family: Physical Intelligence VLA models

## Future Improvements

- [ ] Multi-condition steering (combine multiple conditions)
- [ ] Adaptive threshold based on similarity distribution
- [ ] Layer-wise steering (apply different steering at different layers)
- [ ] Automatic hyperparameter tuning
- [ ] Integration with π₀.₅ fine-tuning pipeline
