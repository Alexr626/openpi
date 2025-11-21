# CAST Architecture and Data Flow

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                    CAST System Overview                              │
└─────────────────────────────────────────────────────────────────────┘

┌──────────────────┐
│  Initialization  │  (Done once at startup)
└────────┬─────────┘
         │
         ├─► 1. Define Prompts
         │   ├─ CONDITION_PROMPT: "if you don't have the ball yet"
         │   ├─ POSITIVE_EXAMPLE: "reach far"
         │   └─ NEGATIVE_EXAMPLE: "retract close"
         │
         ├─► 2. Compute Behavior Vector (Static)
         │   ├─ Get hidden states for POSITIVE_EXAMPLE
         │   ├─ Get hidden states for NEGATIVE_EXAMPLE
         │   └─ behavior_vec = positive - negative
         │
         └─► 3. Register Hook
             └─ DynamicConditionalSteeringHook.register()

┌──────────────────┐
│  Runtime Loop    │  (Every timestep)
└────────┬─────────┘
         │
         ├─► 1. Update Camera Frames
         │   └─ cast_hook.update_images(current_frames)
         │
         ├─► 2. Forward Pass (Policy Inference)
         │   │
         │   │   ┌────────────────────────────────────┐
         │   └──►│  PaliGemma Forward Pass            │
         │       │  (CAST hook triggered at layer 15) │
         │       └────────────────────────────────────┘
         │                      │
         │                      ▼
         │       ┌─────────────────────────────────────────────┐
         │       │  CAST Hook Execution (hook_fn)              │
         │       │                                             │
         │       │  A. Compute Condition Vector (Dynamic)      │
         │       │     └─ Based on current camera frames       │
         │       │                                             │
         │       │  B. Get Current Hidden State (h)            │
         │       │                                             │
         │       │  C. Project h onto Condition Vector         │
         │       │     └─ proj = (c·h / c·c) * c              │
         │       │                                             │
         │       │  D. Compute Similarity                      │
         │       │     └─ sim = cosine(h, proj)               │
         │       │                                             │
         │       │  E. Apply Threshold                         │
         │       │     └─ mask = 1 if sim > θ, else 0         │
         │       │                                             │
         │       │  F. Apply Steering                          │
         │       │     └─ h' = h + mask * α * v               │
         │       │                                             │
         │       │  G. Return Modified Hidden State (h')       │
         │       └─────────────────────────────────────────────┘
         │                      │
         │                      ▼
         ├─► 3. Get Actions
         │   └─ actions = process_logits(output)
         │
         └─► 4. Execute Actions
             └─ robot.execute(actions)
```

## Detailed Component Flow

### 1. Behavior Vector Computation (Initialization)

```
Dummy Image ──┐
              │
POSITIVE ─────┼──► get_text_and_vision_based_hidden_states()
              │    │
              │    ▼
              │    Hidden States [batch, seq_len, hidden_dim]
              │    │
              │    ▼
              │    extract_condition_vector(pool_method='mean')
              │    │
              │    ▼
              │    Positive Vector [hidden_dim]
              │
NEGATIVE ─────┼──► get_text_and_vision_based_hidden_states()
              │    │
              │    ▼
              │    Hidden States [batch, seq_len, hidden_dim]
              │    │
              │    ▼
              │    extract_condition_vector(pool_method='mean')
              │    │
              │    ▼
              │    Negative Vector [hidden_dim]
              │
              ▼
       compute_behavior_vector()
              │
              ▼
       Behavior Vector (v) = Positive - Negative
       [hidden_dim]
```

### 2. Dynamic Condition Vector Computation (Every Timestep)

```
Current Camera Frames ──┐
                        │
CONDITION_TEXT ─────────┼──► get_text_and_vision_based_hidden_states()
                        │    │
                        │    ▼
                        │    Hidden States [batch, seq_len, hidden_dim]
                        │    │
                        │    ▼
                        │    extract_condition_vector(pool_method='mean')
                        │    │
                        │    ▼
                        │    Condition Vector (c) [hidden_dim]
```

### 3. CAST Application (During Forward Pass)

```
Input: Hidden State (h) [batch, seq_len, hidden_dim]
       Condition Vector (c) [hidden_dim]
       Behavior Vector (v) [hidden_dim]
       Alpha (α): 2.0
       Threshold (θ): 0.5

Step 1: Project onto Condition
┌──────────────────────────────────┐
│ proj_c(h) = (c·h / c·c) * c      │
│                                  │
│ For each position in sequence:   │
│   h_dot_c = h · c                │
│   c_dot_c = c · c                │
│   proj = (h_dot_c / c_dot_c) * c │
│                                  │
│ Apply tanh for stability:        │
│   proj = tanh(proj)              │
└──────────────────────────────────┘
       │
       ▼
Projection [batch, seq_len, hidden_dim]

Step 2: Compute Similarity
┌──────────────────────────────────┐
│ sim(h, proj) = h·proj / |h||proj|│
│                                  │
│ Cosine similarity per position   │
└──────────────────────────────────┘
       │
       ▼
Similarity [batch, seq_len]
Values in range [-1, 1]

Step 3: Apply Threshold
┌──────────────────────────────────┐
│ f(sim) = 1 if sim > θ, else 0   │
│                                  │
│ Example with θ=0.5:              │
│   sim = [0.3, 0.6, 0.7, 0.2]     │
│   mask = [0, 1, 1, 0]            │
└──────────────────────────────────┘
       │
       ▼
Mask [batch, seq_len]
Binary values: 0 or 1

Step 4: Apply Steering
┌──────────────────────────────────┐
│ h' = h + mask * α * v            │
│                                  │
│ For each position:               │
│   if mask[pos] == 1:             │
│     h'[pos] = h[pos] + α * v     │
│   else:                          │
│     h'[pos] = h[pos]             │
└──────────────────────────────────┘
       │
       ▼
Modified Hidden State (h') [batch, seq_len, hidden_dim]
```

## Layer-wise View of PaliGemma with CAST

```
┌─────────────────────────────────────────────────────────────┐
│                    PaliGemma Model                          │
│                                                             │
│  ┌──────────────┐                                           │
│  │ Vision Tower │  Process camera frames                    │
│  │ (ViT)        │  Extract visual features                  │
│  └──────┬───────┘                                           │
│         │                                                   │
│         ▼                                                   │
│  ┌──────────────┐                                           │
│  │ Multimodal   │  Combine vision + text embeddings         │
│  │ Projector    │                                           │
│  └──────┬───────┘                                           │
│         │                                                   │
│         ▼                                                   │
│  ┌──────────────────────────────────────────────────────┐  │
│  │         Gemma Language Model (26 layers)             │  │
│  │                                                      │  │
│  │  Layer 0  ─┐                                         │  │
│  │  Layer 1   │                                         │  │
│  │  ...       │ Early layers                            │  │
│  │  Layer 8  ─┘ (General features)                      │  │
│  │                                                      │  │
│  │  Layer 9  ─┐                                         │  │
│  │  Layer 10  │                                         │  │
│  │  ...       │ Middle layers                           │  │
│  │  Layer 14  │ (Balance of general + task-specific)    │  │
│  │  Layer 15 ◄├─── CAST APPLIED HERE (default)          │  │
│  │  Layer 16  │                                         │  │
│  │  Layer 17 ─┘                                         │  │
│  │                                                      │  │
│  │  Layer 18 ─┐                                         │  │
│  │  ...       │ Later layers                            │  │
│  │  Layer 25 ─┘ (Task-specific features)                │  │
│  │                                                      │  │
│  └──────────────────────────────────────────────────────┘  │
│         │                                                   │
│         ▼                                                   │
│  ┌──────────────┐                                           │
│  │ Output Head  │  Generate logits for actions              │
│  └──────────────┘                                           │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## Data Shapes Throughout CAST

```
Component                          Shape                      Dtype
─────────────────────────────────────────────────────────────────────
Camera Frame (PIL)                 (224, 224, 3)              uint8
Processor Output                   [batch, seq_len]           long
  ├─ input_ids                     [1, ~260]                  long
  └─ pixel_values                  [1, 3, 224, 224]           float32

Hidden States (raw)                [batch, seq_len, hidden]   bfloat16
  └─ Example                       [1, 260, 2048]             bfloat16

Condition Vector (after pooling)   [hidden_dim]               bfloat16
  └─ Example                       [2048]                     bfloat16

Behavior Vector                    [hidden_dim]               bfloat16
  └─ Example                       [2048]                     bfloat16

During CAST Application:
├─ Hidden State (h)                [1, 260, 2048]             bfloat16
├─ Projection                      [1, 260, 2048]             bfloat16
├─ Similarity                      [1, 260]                   float32
├─ Mask                            [1, 260]                   float32
└─ Modified Hidden (h')            [1, 260, 2048]             bfloat16

Output Logits                      [1, seq_len, vocab_size]   bfloat16
  └─ Example                       [1, 260, 257152]           bfloat16
```

## Hook Registration Flow

```
┌─────────────────────────────────────────────────────────────┐
│  DynamicConditionalSteeringHook Lifecycle                   │
└─────────────────────────────────────────────────────────────┘

1. Initialization
   │
   ├─► Store references:
   │   ├─ self.model = PaliGemmaWithExpertModel
   │   ├─ self.processor = AutoProcessor
   │   ├─ self.condition_text = "if condition"
   │   ├─ self.behavior_vec = precomputed vector
   │   ├─ self.alpha = 2.0
   │   ├─ self.threshold = 0.5
   │   └─ self.layer_idx = 15
   │
   └─► Initialize state:
       └─ self.current_images = None

2. Registration
   │
   └─► hook.register()
       │
       └─► model.paligemma.language_model.layers[15].register_forward_hook(hook_fn)
           │
           └─► hook_fn will be called during forward pass

3. Runtime Updates
   │
   └─► For each timestep:
       │
       ├─► hook.update_images(new_frames)
       │   └─ self.current_images = new_frames
       │
       └─► During forward pass:
           │
           └─► hook_fn() automatically triggered
               │
               ├─► Compute condition_vec from current_images
               ├─► Apply CAST to hidden state
               └─► Return modified hidden state

4. Cleanup
   │
   └─► hook.remove()
       └─► Unregister forward hook
```

## Comparison: Static vs Dynamic CAST

```
┌─────────────────────────────────────────────────────────────┐
│                   Static CAST                               │
├─────────────────────────────────────────────────────────────┤
│ Condition Vector:  Pre-computed once at initialization     │
│ Behavior Vector:   Pre-computed once at initialization     │
│                                                             │
│ Pros:                                                       │
│   ✓ Fast (~5ms overhead per forward pass)                  │
│   ✓ Predictable behavior                                   │
│   ✓ Good for testing                                       │
│                                                             │
│ Cons:                                                       │
│   ✗ Doesn't adapt to visual changes                        │
│   ✗ Condition is static (not vision-based)                 │
│                                                             │
│ Use Case:                                                   │
│   Testing, text-only conditions, static environments       │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                   Dynamic CAST                              │
├─────────────────────────────────────────────────────────────┤
│ Condition Vector:  Recomputed every forward pass           │
│ Behavior Vector:   Pre-computed once at initialization     │
│                                                             │
│ Pros:                                                       │
│   ✓ Adapts to visual changes in real-time                  │
│   ✓ Vision-grounded condition evaluation                   │
│   ✓ Better for dynamic environments                        │
│                                                             │
│ Cons:                                                       │
│   ✗ Slower (~50-100ms overhead per forward pass)           │
│   ✗ More complex                                           │
│                                                             │
│ Use Case:                                                   │
│   Robot deployment, vision-dependent conditions,           │
│   dynamic environments (RECOMMENDED)                       │
└─────────────────────────────────────────────────────────────┘
```

## Example: Ball Transportation Timeline

```
Timestep 0: Ball is far from gripper
┌────────────────────────────────────────────────────────────┐
│ Camera: [Ball far from gripper]                            │
│                                                            │
│ Condition Vector Computation:                              │
│   "if you don't have possession of the ball yet"          │
│   + current_image → c₀                                    │
│                                                            │
│ CAST Application:                                          │
│   sim(h, proj_c₀(h)) = 0.75 > 0.5 ✓ Condition met         │
│   Apply steering: h' = h + 2.0 * v_reach                  │
│                                                            │
│ Result: Robot reaches far towards ball                     │
└────────────────────────────────────────────────────────────┘

Timestep 10: Ball is close to gripper
┌────────────────────────────────────────────────────────────┐
│ Camera: [Ball near gripper]                                │
│                                                            │
│ Condition Vector Computation:                              │
│   "if you don't have possession of the ball yet"          │
│   + current_image → c₁                                    │
│                                                            │
│ CAST Application:                                          │
│   sim(h, proj_c₁(h)) = 0.60 > 0.5 ✓ Condition still met   │
│   Apply steering: h' = h + 2.0 * v_reach                  │
│                                                            │
│ Result: Robot continues reaching                           │
└────────────────────────────────────────────────────────────┘

Timestep 20: Ball is grasped
┌────────────────────────────────────────────────────────────┐
│ Camera: [Ball in gripper]                                  │
│                                                            │
│ Condition Vector Computation:                              │
│   "if you don't have possession of the ball yet"          │
│   + current_image → c₂                                    │
│                                                            │
│ CAST Application:                                          │
│   sim(h, proj_c₂(h)) = 0.30 < 0.5 ✗ Condition NOT met     │
│   No steering: h' = h                                     │
│                                                            │
│ Result: Robot follows base policy (transport to basket)    │
└────────────────────────────────────────────────────────────┘
```

## Implementation Layers

```
┌─────────────────────────────────────────────────────────────┐
│                   Implementation Stack                       │
└─────────────────────────────────────────────────────────────┘

User API Layer
├─ DynamicConditionalSteeringHook  (High-level interface)
└─ ConditionalSteeringHook         (Static interface)

Core CAST Functions
├─ apply_conditional_steering()    (Main CAST logic)
├─ compute_similarity()            (Similarity computation)
├─ project_onto_condition()        (Vector projection)
└─ apply_threshold_function()      (Binary masking)

Vector Extraction
├─ compute_behavior_vector()       (Positive - Negative)
└─ extract_condition_vector()      (Pooling hidden states)

Hidden State Extraction
└─ get_text_and_vision_based_hidden_states()  (Forward hooks)

Foundation
├─ PyTorch (torch.nn.functional)
├─ Transformers (HuggingFace)
└─ PaliGemmaWithExpertModel (openpi)
```

## Memory and Compute Profiles

```
┌─────────────────────────────────────────────────────────────┐
│          Operation Costs (Per Forward Pass)                  │
├─────────────────────────────────────────────────────────────┤
│ Operation                    Time        Memory              │
├─────────────────────────────────────────────────────────────┤
│ Base PaliGemma forward       ~500ms     ~3GB                 │
│                                                              │
│ Static CAST:                                                 │
│ ├─ Projection                ~2ms       ~20MB                │
│ ├─ Similarity                ~1ms       ~1MB                 │
│ ├─ Threshold                 ~0.5ms     ~1MB                 │
│ └─ Apply steering            ~2ms       ~20MB                │
│ Total overhead               ~5-10ms    ~50MB                │
│                                                              │
│ Dynamic CAST:                                                │
│ ├─ Compute condition vector  ~50-100ms  ~50MB                │
│ ├─ Projection                ~2ms       ~20MB                │
│ ├─ Similarity                ~1ms       ~1MB                 │
│ ├─ Threshold                 ~0.5ms     ~1MB                 │
│ └─ Apply steering            ~2ms       ~20MB                │
│ Total overhead               ~55-105ms  ~100MB               │
│                                                              │
│ Total (PaliGemma + Dynamic)  ~555-605ms ~3.1GB               │
└─────────────────────────────────────────────────────────────┘

For 10 Hz control (100ms cycle):
├─ Static CAST:  ~1% overhead  ✓ Negligible
└─ Dynamic CAST: ~10-20% overhead  ✓ Acceptable
```
