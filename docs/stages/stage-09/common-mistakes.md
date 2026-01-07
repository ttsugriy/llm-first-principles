# Stage 9: Common Mistakes

## Mistake 1: Training Base Weights with LoRA

**Symptom**: Memory usage same as full fine-tuning

**Wrong**:
```python
# Forgot to freeze base weights!
for param in model.parameters():
    param.requires_grad = True  # Everything trainable

lora.forward(x, W)  # W still getting gradients
```

**The fix**: Freeze base model first
```python
# Freeze all base model parameters
for param in model.parameters():
    param.requires_grad = False

# Only LoRA parameters are trainable
lora_params = lora.parameters()  # A and B matrices
```

---

## Mistake 2: Wrong LoRA Initialization

**Symptom**: Model changes behavior immediately after adding LoRA

**Wrong**:
```python
# Both A and B initialized with random values
self.A = np.random.randn(rank, in_features) * 0.1
self.B = np.random.randn(out_features, rank) * 0.1  # Wrong!
```

**The fix**: Initialize B to zeros
```python
# A: random (Gaussian)
self.A = np.random.randn(rank, in_features) * 0.01

# B: zeros (LoRA starts as identity)
self.B = np.zeros((out_features, rank))

# This way, BA = 0 initially, so W + BA = W (no change)
```

---

## Mistake 3: Forgetting the Scaling Factor

**Symptom**: LoRA has weak effect or requires different learning rates

**Wrong**:
```python
def forward(self, x, W):
    base = x @ W.T
    lora = (x @ self.A.T) @ self.B.T
    return base + lora  # No scaling!
```

**The fix**: Always apply alpha/rank scaling
```python
def forward(self, x, W):
    base = x @ W.T
    lora = (x @ self.A.T) @ self.B.T
    return base + (self.alpha / self.rank) * lora  # Scaled!
```

---

## Mistake 4: Applying LoRA to Wrong Layers

**Symptom**: Poor performance despite correct implementation

**Wrong**:
```python
# Only applying LoRA to output projection
lora_layers = {'o': LoRALayer(...)}  # Not enough!
```

**The fix**: Apply to query and value (most common)
```python
# Empirically, Q and V benefit most from LoRA
lora_layers = {
    'q': LoRALayer(...),  # Query projection
    'v': LoRALayer(...),  # Value projection
}

# For some tasks, also include K and O
```

---

## Mistake 5: LoRA Rank Too High

**Symptom**: Overfitting, no memory savings

**Wrong**:
```python
# rank = 256 for d_model = 512
lora = LoRALayer(in_features=512, out_features=512, rank=256)
# This is almost as many params as the original!
```

**The fix**: Keep rank small (4-64 typically)
```python
# Typical ranks:
# - Simple tasks: rank=4
# - Medium tasks: rank=8-16
# - Complex tasks: rank=32-64

lora = LoRALayer(in_features=512, out_features=512, rank=8)
```

---

## Mistake 6: Not Merging for Inference

**Symptom**: Slow inference with LoRA overhead

**Wrong**:
```python
# Keeping LoRA separate during inference
def forward(self, x):
    base = x @ self.W.T
    lora = (x @ self.A.T) @ self.B.T * self.scaling
    return base + lora  # Two matrix mults!
```

**The fix**: Merge after training
```python
# Before deployment
W_merged = self.W + self.scaling * (self.B @ self.A)

# Now inference is just
def forward(self, x):
    return x @ W_merged.T  # Single matrix mult!
```

---

## Mistake 7: Prompt Tuning Position

**Symptom**: Prompt tuning doesn't work well

**Wrong**:
```python
# Adding prompts after embeddings wrong
output = model(input_embeds)
output_with_prompt = concat(prompt, output)  # Wrong place!
```

**The fix**: Prepend prompts to input embeddings
```python
# Prompts go at the start of the input
input_embeds = model.embed(tokens)
input_with_prompt = concat(self.soft_prompts, input_embeds, axis=1)
output = model(input_with_prompt)
```

---

## Mistake 8: Adapter Residual Wrong

**Symptom**: Adapters break model performance

**Wrong**:
```python
def forward(self, x):
    down = x @ self.W_down
    up = relu(down) @ self.W_up
    return up  # Forgot residual!
```

**The fix**: Always include residual connection
```python
def forward(self, x):
    down = x @ self.W_down
    up = relu(down) @ self.W_up
    return x + up  # Residual connection!
```

---

## Mistake 9: Wrong Learning Rate for PEFT

**Symptom**: Training unstable or too slow

**Wrong**:
```python
# Using same LR as full fine-tuning
lora_lr = 1e-5  # Too low for LoRA!
```

**The fix**: PEFT methods often need higher LR
```python
# LoRA typically uses 10-100x higher LR than full fine-tuning
lora_lr = 1e-4  # or even 1e-3

# Prompt tuning may need even higher
prompt_lr = 1e-2
```

---

## Mistake 10: Not Saving LoRA Separately

**Symptom**: Huge checkpoint files, can't swap adapters

**Wrong**:
```python
# Saving entire model including frozen weights
save_checkpoint(model)  # 7GB for a 7B model
```

**The fix**: Save only LoRA weights
```python
def save_lora(lora_layers, path):
    """Save only LoRA parameters (few MB)."""
    lora_state = {}
    for name, layer in lora_layers.items():
        lora_state[f'{name}_A'] = layer.A
        lora_state[f'{name}_B'] = layer.B
    np.savez(path, **lora_state)

def load_lora(lora_layers, path):
    """Load LoRA weights into existing layers."""
    state = np.load(path)
    for name, layer in lora_layers.items():
        layer.A = state[f'{name}_A']
        layer.B = state[f'{name}_B']
```

---

## Mistake 11: Prefix Tuning Mask Errors

**Symptom**: Model attends incorrectly to prefix

**Wrong**:
```python
# Not adjusting attention mask for prefix
K = concat(prefix_K, K)
V = concat(prefix_V, V)
attention = softmax(Q @ K.T)  # Mask doesn't account for prefix!
```

**The fix**: Extend attention mask
```python
# Prefix tokens should always be attendable
prefix_len = self.prefix_length
seq_len = Q.shape[1]

# Extend mask to cover prefix
if mask is not None:
    prefix_mask = np.zeros((seq_len, prefix_len))  # Can attend to prefix
    extended_mask = np.concatenate([prefix_mask, mask], axis=1)
    attention = softmax(Q @ K.T + extended_mask)
```

---

## Mistake 12: Gradient Computation for Frozen Weights

**Symptom**: Slow backward pass, memory waste

**Wrong**:
```python
# Computing gradients for frozen weights unnecessarily
def backward(self, grad_output):
    # Still computing grad for W even though frozen
    grad_W = x.T @ grad_output  # Waste!
    grad_lora = compute_lora_grad(grad_output)
```

**The fix**: Skip frozen weight gradients
```python
def backward(self, grad_output):
    # Only compute gradients for LoRA parameters
    grad_B = (x @ self.A.T).T @ (grad_output * self.scaling)
    grad_A = self.B.T @ (grad_output * self.scaling).T @ x

    # Gradient w.r.t. input still needed for chain rule
    grad_x = grad_output @ (self.W + self.scaling * self.B @ self.A)

    return grad_x  # Don't return grad_W
```
