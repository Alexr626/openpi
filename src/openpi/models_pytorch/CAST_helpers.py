import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoProcessor
from openpi.models_pytorch.pi0_pytorch import PI0Pytorch
import openpi.models.gemma as _gemma
from transformers.models.auto import CONFIG_MAPPING
from transformers.models.gemma import modeling_gemma

def pad_tokens(tokenizer: AutoTokenizer,
               prompt_add_raw: str,
               prompt_sub_raw: str):
    """Pad prompts to equal token length using the tokenizer's pad token.

    Args:
        tokenizer: A tokenizer loaded via AutoTokenizer.from_pretrained('<paligemma_path>')
        prompt_add_raw: The positive/additive prompt string
        prompt_sub_raw: The negative/subtractive prompt string

    Returns:
        Tuple of (padded_prompt_add, padded_prompt_sub) as strings
    """
    # Tokenize both prompts
    tokens_add = tokenizer(prompt_add_raw, return_tensors="pt")["input_ids"]
    tokens_sub = tokenizer(prompt_sub_raw, return_tensors="pt")["input_ids"]

    max_len = max(tokens_add.shape[1], tokens_sub.shape[1])

    # Get the pad token ID
    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id  # Fallback to EOS

    # Pad the shorter sequence
    if tokens_add.shape[1] < max_len:
        padding = torch.full((1, max_len - tokens_add.shape[1]), pad_token_id, dtype=tokens_add.dtype, device=tokens_add.device)
        tokens_add = torch.cat([tokens_add, padding], dim=1)
    elif tokens_sub.shape[1] < max_len:
        padding = torch.full((1, max_len - tokens_sub.shape[1]), pad_token_id, dtype=tokens_sub.dtype, device=tokens_sub.device)
        tokens_sub = torch.cat([tokens_sub, padding], dim=1)

    # Convert back to strings for compatibility with existing code
    prompt_add = tokenizer.decode(tokens_add[0], skip_special_tokens=False)
    prompt_sub = tokenizer.decode(tokens_sub[0], skip_special_tokens=False)

    return prompt_add, prompt_sub


def get_text_and_vision_based_hidden_states(model: PI0Pytorch, image, text, layer_idx, processor):
    """
    Get activations at specific layer in Paligemma decoder
    
    Args:
        image: PIL Image or torch.Tensor of shape [1, 3, H, W]
        text: String prompt (e.g., "Sharp knife")
        layer_idx: Which Gemma decoder layer to extract from
        processor: PaliGemmaProcessor for formatting inputs
    """
    # Prepare inputs using the PaliGemma processor
    # This handles image preprocessing + tokenization + special token insertion
    inputs = processor(
        text=text,
        images=image,
        return_tensors="pt"
    ).to(model.paligemma_with_expert.paligemma.device)
    
    # inputs will contain:
    # - input_ids: text tokens with image placeholder tokens
    # - pixel_values: preprocessed image tensor
    # - attention_mask: mask for the sequence
    
    activations = {}
    
    def hook(module, input, output):
        # Capture decoder layer output
        activations['hidden'] = (output[0] if isinstance(output, tuple) else output).detach()
    
    # Hook into the Gemma decoder layer (not vision encoder!)
    handle = model.paligemma_with_expert.paligemma.language_model.layers[layer_idx].mlp.down_proj.register_forward_hook(hook)
    
    with torch.no_grad():
        with torch.autocast('cuda'):
            _ = model.paligemma_with_expert.paligemma(**inputs)  # Forward pass triggers hook
    
    handle.remove()
    
    return activations['hidden']


def get_text_based_hidden_states(model: PI0Pytorch, text, layer_idx, tokenizer):
    """
    Get activations at specific layer in Paligemma decoder
    
    Args:
        image: PIL Image or torch.Tensor of shape [1, 3, H, W]
        text: String prompt (e.g., "Sharp knife")
        layer_idx: Which Gemma decoder layer to extract from
        processor: PaliGemmaProcessor for formatting inputs
    """
    # Prepare inputs using the PaliGemma processor
    # This handles image preprocessing + tokenization + special token insertion
    inputs = tokenizer(
        text=text,
        return_tensors="pt",
    ).to(model.paligemma_with_expert.paligemma.device)
    
    # inputs will contain:
    # - input_ids: text tokens with image placeholder tokens
    # - pixel_values: preprocessed image tensor
    # - attention_mask: mask for the sequence
    
    activations = {}
    
    def hook(module, input, output):
        # Capture decoder layer output
        activations['hidden'] = (output[0] if isinstance(output, tuple) else output).detach()
    
    # Hook into the Gemma decoder layer (not vision encoder!)
    handle = model.paligemma_with_expert.paligemma.language_model.layers[layer_idx].mlp.down_proj.register_forward_hook(hook)
    
    with torch.no_grad():
        with torch.autocast('cuda'):
            _ = model.paligemma_with_expert.paligemma(**inputs)  # Forward pass triggers hook
    
    handle.remove()
    
    return activations['hidden']


class SteeringHook:
    """Apply steering during generation"""
    def __init__(self, model: PI0Pytorch, steering_vec, alpha, layer_idx):
        self.model = model
        self.steering_vec = steering_vec.to(model.paligemma_with_expert.paligemma.device)
        self.alpha = alpha
        self.layer_idx = layer_idx
        self.handle = None
    
    def hook_fn(self, module, input, output):
        print("ADDING STEERING VECTOR")
        hidden = output[0] if isinstance(output, tuple) else output
        seq_len = hidden.shape[1]
        steer_len = self.steering_vec.shape[1]
        
        if seq_len >= steer_len:
            hidden[:, -steer_len:, :] += self.alpha * self.steering_vec
        else:
            hidden += self.alpha * self.steering_vec[:, :seq_len, :]
        
        return (hidden,) + output[1:] if isinstance(output, tuple) else hidden
    
    def register(self):
        self.handle = self.model.paligemma_with_expert.paligemma.language_model.layers[self.layer_idx].mlp.down_proj.register_forward_hook(self.hook_fn)
    
    
    def remove(self):
        if self.handle:
            self.handle.remove()


def generate(prompt, 
             model: PI0Pytorch,
             tokenizer, 
             steering_vec=None, 
             alpha=0, 
             layer=20,
             max_tokens=20):
    """Generate text with optional steering"""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.paligemma_with_expert.paligemma.device)
    
    hook = None
    if steering_vec is not None:
        hook = SteeringHook(steering_vec, alpha, layer)
        hook.register()
    
    try:
        with torch.no_grad():
            output_ids = model.paligemma_with_expert.paligemma.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=True,
                temperature=0.8,
                pad_token_id=tokenizer.eos_token_id
            )
        text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        return text.replace(prompt, "").strip()
    finally:
        if hook:
            hook.remove()


# ------------------------------------------- CAST HELPERS -----------------------------------

def compute_behavior_vector(
    positive_hidden_states: torch.Tensor,
    negative_hidden_states: torch.Tensor,
    pool_method: str = 'mean'
) -> torch.Tensor:
    """
    Compute behavior (steering) vector from positive and negative contrasting examples.

    Args:
        positive_hidden_states: Hidden states from positive prompt [batch, seq_len, hidden_dim]
        negative_hidden_states: Hidden states from negative prompt [batch, seq_len, hidden_dim]
        pool_method: Method to pool sequence dimension ('mean', 'last', 'max')

    Returns:
        Behavior vector [seq_len, hidden_dim] or [hidden_dim] depending on pooling
    """
    # Pool both hidden states
    positive_vec = extract_hidden_vector(positive_hidden_states, pool_method)
    negative_vec = extract_hidden_vector(negative_hidden_states, pool_method)

    # Compute steering vector as difference
    behavior_vec = positive_vec - negative_vec

    return behavior_vec


def extract_hidden_vector(hidden_states: torch.Tensor, pool_method: str = 'mean') -> torch.Tensor:
    """
    Extract vector from the activation values of the hidden states.

    Args:
        hidden_states: Hidden states [batch, seq_len, hidden_dim]
        pool_method: Method to pool sequence dimension ('mean', 'last', 'max')

    Returns:
        Vector [batch, hidden_dim] or [hidden_dim]
    """
    if pool_method == 'mean':
        # Average pool across sequence dimension
        vec = hidden_states.mean(dim=1)
    elif pool_method == 'last':
        # Take last token's hidden state
        vec = hidden_states[:, -1, :]
    elif pool_method == 'max':
        # Max pool across sequence dimension
        vec = hidden_states.max(dim=1)[0]
    else:
        raise ValueError(f"Unknown pool_method: {pool_method}")

    # Squeeze batch dimension if batch size is 1
    if vec.shape[0] == 1:
        vec = vec.squeeze(0)

    return vec


def project_onto_condition(hidden_state: torch.Tensor, condition_vec: torch.Tensor, use_tanh: bool = True) -> torch.Tensor:
    """
    Project hidden state onto condition vector using the formula:
    proj_c(h) = (c ⊗ c / c·c) h

    Args:
        hidden_state: Current hidden state [batch, seq_len, hidden_dim] or [seq_len, hidden_dim]
        condition_vec: Condition vector [hidden_dim]
        use_tanh: Apply tanh non-linearity as in the paper (default: True)

    Returns:
        Projection of hidden state onto condition vector (same shape as hidden_state)
    """
    # Ensure condition_vec is the right shape [hidden_dim]
    if condition_vec.dim() > 1:
        condition_vec = condition_vec.squeeze()

    # Compute c·c (scalar)
    c_dot_c = torch.dot(condition_vec, condition_vec)

    # Prevent division by zero
    if c_dot_c < 1e-8:
        raise ValueError("Condition vector has near-zero norm")

    # Reshape for broadcasting
    original_shape = hidden_state.shape
    if hidden_state.dim() == 3:  # [batch, seq_len, hidden_dim]
        h_flat = hidden_state.reshape(-1, hidden_state.shape[-1])  # [batch*seq_len, hidden_dim]
    elif hidden_state.dim() == 2:  # [seq_len, hidden_dim]
        h_flat = hidden_state
    else:
        raise ValueError(f"Unexpected hidden_state shape: {hidden_state.shape}")

    # Compute h·c for each position (element-wise dot product along hidden dim)
    h_dot_c = torch.matmul(h_flat, condition_vec)  # [batch*seq_len] or [seq_len]

    # Compute projection: (h·c / c·c) * c
    # This projects h onto the direction of c
    proj_coeff = h_dot_c / c_dot_c  # [batch*seq_len] or [seq_len]
    projection = proj_coeff.unsqueeze(-1) * condition_vec.unsqueeze(0)  # [batch*seq_len, hidden_dim]

    # Apply tanh non-linearity (as mentioned in the paper for stability)
    if use_tanh:
        projection = torch.tanh(projection)

    # Reshape back to original shape
    projection = projection.reshape(original_shape)

    return projection


def compute_similarity(hidden_state: torch.Tensor, projection: torch.Tensor) -> torch.Tensor:
    """
    Compute cosine similarity between hidden state and its projection:
    sim(h, proj_c(h)) = h·g / (|h||g|)

    Args:
        hidden_state: Current hidden state [batch, seq_len, hidden_dim] or [seq_len, hidden_dim]
        projection: Projection of hidden state [same shape as hidden_state]

    Returns:
        Cosine similarity values [batch, seq_len] or [seq_len]
    """
    # Flatten batch and sequence dimensions if needed
    original_shape = hidden_state.shape[:-1]  # Everything except hidden_dim

    if hidden_state.dim() == 3:  # [batch, seq_len, hidden_dim]
        h_flat = hidden_state.reshape(-1, hidden_state.shape[-1])
        p_flat = projection.reshape(-1, projection.shape[-1])
    elif hidden_state.dim() == 2:  # [seq_len, hidden_dim]
        h_flat = hidden_state
        p_flat = projection
    else:
        raise ValueError(f"Unexpected hidden_state shape: {hidden_state.shape}")

    # Compute cosine similarity using F.cosine_similarity
    # This handles the normalization: h·g / (|h||g|)
    similarity = F.cosine_similarity(h_flat, p_flat, dim=-1)  # [batch*seq_len] or [seq_len]

    # Reshape back to [batch, seq_len] or [seq_len]
    similarity = similarity.reshape(original_shape)

    return similarity


def apply_threshold_function(similarity: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    """
    Apply thresholding function f that outputs 1 if similarity > threshold, else 0.

    Args:
        similarity: Similarity values [batch, seq_len] or [seq_len]
        threshold: Threshold value θ (default: 0.5)

    Returns:
        Binary mask [batch, seq_len] or [seq_len]
    """
    return (similarity > threshold).float()


def apply_conditional_steering(
    hidden_state: torch.Tensor,
    condition_vec: torch.Tensor,
    behavior_vec: torch.Tensor,
    alpha: float,
    threshold: float = 0.5,
    use_tanh: bool = True
) -> torch.Tensor:
    """
    Apply Conditional Activation Steering using the formula:
    h' ← h + f(sim(h, proj_c h)) · α · v

    where:
    - h is the original hidden state
    - c is the condition vector
    - v is the behavior (steering) vector
    - α is the scaling factor
    - f is the thresholding function

    Args:
        hidden_state: Current hidden state [batch, seq_len, hidden_dim]
        condition_vec: Condition vector [hidden_dim]
        behavior_vec: Behavior/steering vector [seq_len, hidden_dim] or [hidden_dim]
        alpha: Scaling factor for steering strength
        threshold: Similarity threshold for f (default: 0.5)
        use_tanh: Apply tanh to projection (default: True)

    Returns:
        Modified hidden state h' (same shape as hidden_state)
    """
    # 1. Project hidden state onto condition vector
    projection = project_onto_condition(hidden_state, condition_vec, use_tanh=use_tanh)

    # 2. Compute similarity between hidden state and projection
    similarity = compute_similarity(hidden_state, projection)

    # 3. Apply threshold function
    mask = apply_threshold_function(similarity, threshold)

    # 4. Expand mask to match hidden dimension for broadcasting
    # mask is [batch, seq_len], we need [batch, seq_len, 1]
    if mask.dim() == 2:
        mask = mask.unsqueeze(-1)
    elif mask.dim() == 1:
        mask = mask.unsqueeze(-1)

    # 5. Prepare behavior vector for broadcasting
    if behavior_vec.dim() == 1:  # [hidden_dim]
        # Broadcast to [1, 1, hidden_dim] or [1, seq_len, hidden_dim]
        behavior_vec = behavior_vec.unsqueeze(0).unsqueeze(0)
    elif behavior_vec.dim() == 2:  # [seq_len, hidden_dim]
        behavior_vec = behavior_vec.unsqueeze(0)  # [1, seq_len, hidden_dim]

    # 6. Apply steering: h' = h + f(sim) · α · v
    modified_hidden = hidden_state + mask * alpha * behavior_vec

    return modified_hidden.to(torch.bfloat16)


class ConditionalSteeringHook:
    """Apply conditional activation steering during generation"""
    def __init__(self,
                 model: PI0Pytorch,
                 condition_vec: torch.Tensor,
                 behavior_vec: torch.Tensor,
                 alpha: float,
                 layer_idx: int,
                 threshold: float = 0.5,
                 use_tanh: bool = True):
        """
        Initialize conditional steering hook.

        Args:
            model: PaliGemmaWithExpertModel instance
            condition_vec: Condition vector [hidden_dim]
            behavior_vec: Behavior/steering vector [seq_len, hidden_dim] or [hidden_dim]
            alpha: Steering strength
            layer_idx: Which layer to apply steering to
            threshold: Similarity threshold for activation (default: 0.5)
            use_tanh: Apply tanh to projection (default: True)
        """
        self.model = model
        self.condition_vec = condition_vec.to(self.model.device)
        self.behavior_vec = behavior_vec.to(self.model.device)
        self.alpha = alpha
        self.layer_idx = layer_idx
        self.threshold = threshold
        self.use_tanh = use_tanh
        self.handle = None

    def hook_fn(self, module, input, output):
        """Hook function that applies conditional steering"""
        hidden = output[0] if isinstance(output, tuple) else output

        # Apply conditional activation steering
        modified_hidden = apply_conditional_steering(
            hidden_state=hidden,
            condition_vec=self.condition_vec,
            behavior_vec=self.behavior_vec,
            alpha=self.alpha,
            threshold=self.threshold,
            use_tanh=self.use_tanh
        )

        return (modified_hidden,) + output[1:] if isinstance(output, tuple) else modified_hidden

    def register(self):
        """Register the hook to the specified layer"""
        self.handle = self.model.paligemma_with_expert.paligemma.language_model.layers[self.layer_idx].mlp.down_proj.register_forward_hook(self.hook_fn)

    def remove(self):
        """Remove the hook"""
        if self.handle:
            self.handle.remove()


if __name__=='__main__':
    # Load tokenizer for PaliGemma (update path as needed)
    tokenizer = AutoTokenizer.from_pretrained("google/paligemma-3b-pt-224")

    # print(pad_tokens(tokenizer=tokenizer,
    #                  prompt_add_raw=POSITIVE_EXAMPLE,
    #                  prompt_sub_raw=NEGATIVE_EXAMPLE))