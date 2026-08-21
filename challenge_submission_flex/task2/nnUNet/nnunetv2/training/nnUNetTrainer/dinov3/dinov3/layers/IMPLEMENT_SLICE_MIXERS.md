# Implementation Spec: Cross-Slice Token Mixer Ablations for 2D-to-3D DINO Adaptation

## Goal

We want to modify the current `SliceSelfAttentionBlock` / DINO-style transformer pipeline to support a lightweight cross-slice mixing module for 3D medical volumes represented as slice videos.

The input 3D volume is treated as a sequence of 2D slices. Each slice is processed by a pretrained 2D DINOv3-like transformer. After each selected DINO layer, we collect the global tokens from each slice:

* 1 CLS token
* 4 register tokens

For a 16-slice volume, this gives:

```text
D = 16 slices
T = 5 global tokens per slice = 1 CLS + 4 registers
N = D * T = 80 global tokens
C = 768 feature dimension
```

The cross-slice mixer receives these global tokens, mixes information across slices, and returns residual offsets to update the CLS/register tokens before the next slice-wise DINO layer.

The key design principle is:

```text
Do NOT perform full 3D attention over all patch tokens.
Only mix compact CLS/register tokens across slices.
```

This keeps the 2D DINO backbone mostly unchanged and injects 3D volumetric context efficiently.

---

## Expected Input and Output

The cross-slice mixer should support input in either of the following shapes:

```python
# Preferred structured shape
x: Tensor  # [B, D, T, C]

# Optional flattened shape
x: Tensor  # [B, D*T, C]
```

where:

```text
B = batch size
D = number of slices, e.g. 16
T = number of global tokens per slice, e.g. 5
C = embedding dimension, e.g. 768
```

The output should have the same shape as the input.

The mixer should behave like a residual adapter:

```python
x_new = x + gate * delta
```

where:

```text
delta = predicted cross-slice offset
gate is a learnable scalar initialized to 0
```

This ensures that at initialization the model behaves exactly like the original 2D DINO model.

---

## Where to Insert the Mixer

In each selected DINO layer:

1. Run the normal 2D slice-wise DINO block independently on each slice.
2. Extract the CLS token and register tokens from each slice.
3. Stack them into `[B, D, T, C]`.
4. Apply the cross-slice mixer.
5. Replace the original CLS/register tokens with the updated ones.
6. Continue to the next DINO layer.

Patch tokens are not directly cross-slice mixed in this module.

The updated CLS/register tokens should participate in the next slice-wise attention layer, allowing 3D context to gradually influence patch tokens indirectly.

---

## Positional and Token-Type Embeddings

The cross-slice mixer should include two optional embeddings:

### 1. Slice positional embedding

Each slice index should receive a learnable or sinusoidal embedding:

```python
slice_pos_embed: [1, D, 1, C]
```

This tells the mixer which slice each token belongs to.

### 2. Token-type embedding

Each global token type should receive an embedding:

```python
token_type_embed: [1, 1, T, C]
```

Token types are:

```text
0: CLS
1: register token 1
2: register token 2
3: register token 3
4: register token 4
```

Before mixing, use:

```python
x_pos = x + slice_pos_embed[:, :D] + token_type_embed[:, :, :T]
```

If input is flattened `[B, D*T, C]`, reshape to `[B, D, T, C]` first if possible.

---

## Important Residual Rule

The cross-slice mixer should output an offset, not a full replacement.

Use:

```python
x_pos = x + pos
out = mixer_core(x_pos)
delta = out - x_pos
x_new = x + gate * delta
```

Do not use:

```python
x_new = x + gate * out
```

because `out` already contains the residual path if the mixer core is a transformer block.

This is especially important for the layer-copied LoRA mixer.

---

# Required Ablation Variants

Implement three mixer variants.

---

## Variant 1: Bottleneck Attention Adapter

### Name

```python
BottleneckCrossSliceAttentionMixer
```

### Purpose

A lightweight fully trainable adapter that does not copy DINO weights.

### Structure

Input:

```python
x: [B, D, T, C]
```

Flatten tokens:

```python
x_flat: [B, D*T, C]
```

Then:

```text
LayerNorm(C)
Linear C -> r
Self-attention in r dimensions
MLP in r dimensions
Linear r -> C
zero-initialized gated residual update
```

Recommended defaults:

```python
C = 768
r = 128
num_heads = 4
mlp_ratio = 2.0
gate initialized to 0
```

Pseudo-forward:

```python
def forward(x):
    # x: [B, D, T, C]
    x_input = x
    x_pos = add_slice_and_type_pos(x)

    B, D, T, C = x_pos.shape
    y = x_pos.reshape(B, D * T, C)

    y = norm(y)
    y = down(y)                  # [B, D*T, r]
    y = y + attn(norm_attn(y))    # attention in r dim
    y = y + mlp(norm_mlp(y))      # MLP in r dim
    delta = up(y)                # [B, D*T, C]

    delta = delta.reshape(B, D, T, C)
    return x_input + gate * delta
```

Note: This variant predicts the offset directly, so `delta = up(y)` is fine. It does not need `out - x_pos`.

### Trainable Parameters

All parameters in this module are trainable.

---

## Variant 2: Layer-Copied LoRA Mixer

### Name

```python
LayerCopiedLoRACrossSliceMixer
```

### Purpose

A 768-dimensional cross-slice transformer block initialized from a pretrained DINOv3 transformer layer.

This variant should copy the corresponding DINOv3 layer weights, freeze the copied base weights, and train only LoRA parameters plus small positional embeddings and gate.

### Core idea

For DINO layer `l`, create a cross-slice mixer by deep-copying DINO block `l`:

```python
mixer_l = deepcopy(dino_blocks[l])
```

Then:

1. Freeze all copied parameters.
2. Inject LoRA into selected linear layers.
3. Train only LoRA parameters, slice/type embeddings, and gate.

### Target LoRA Layers

Use LoRA rank 16 by default.

Apply LoRA to:

```text
attention q projection
attention v projection
MLP fc1
MLP fc2
```

Optional ablation:

```text
also apply LoRA to attention output projection
```

But the default should be Q/V + MLP only.

### Important Implementation Detail

Many DINO attention implementations store QKV in one linear layer:

```python
qkv: Linear(C, 3*C)
```

If this is the case, implement one of the following:

### Option A: Replace qkv with a LoRA-aware QKV wrapper

The wrapper should preserve the frozen pretrained qkv output and add LoRA offsets only to q and v:

```python
qkv_base = frozen_qkv(x)  # [B, N, 3*C]
q_base, k_base, v_base = split(qkv_base)

q = q_base + lora_q(x)
k = k_base
v = v_base + lora_v(x)

qkv = concat(q, k, v)
```

### Option B: Split qkv into q, k, v linear layers

Copy the relevant weight slices from pretrained qkv:

```python
q.weight = qkv.weight[:C]
k.weight = qkv.weight[C:2*C]
v.weight = qkv.weight[2*C:]
```

Then wrap q and v with LoRA and freeze q/k/v base weights.

Option A is preferred because it minimally changes the original block.

### Forward Rule

Input:

```python
x: [B, D, T, C]
```

Then:

```python
x_input = x
x_pos = x + slice_pos_embed + token_type_embed
x_flat = x_pos.reshape(B, D*T, C)

out = copied_dino_block_with_lora(x_flat)
delta = out - x_flat

x_new = x_input.reshape(B, D*T, C) + gate * delta
x_new = x_new.reshape(B, D, T, C)
```

Important:

```python
delta = out - x_flat
```

because the copied transformer block already contains internal residual connections.

### Gate

Use a learnable scalar gate initialized to zero:

```python
self.gate = nn.Parameter(torch.zeros(1))
```

Optionally use one gate per layer:

```python
self.gate_attn_or_block = nn.Parameter(torch.zeros(1))
```

Do not initialize gate to 1.

### Trainable Parameters

Only these should be trainable:

```text
LoRA parameters
slice_pos_embed
token_type_embed
gate
optional LayerNorm affine parameters if desired
```

The copied DINO block base weights should remain frozen.

### Expected Parameter Count

For `C=768`, `mlp_ratio=4`, `rank=16`, LoRA on Q/V + MLP:

```text
Q LoRA:       16 * (768 + 768) = 24,576
V LoRA:       16 * (768 + 768) = 24,576
MLP fc1 LoRA: 16 * (768 + 3072) = 61,440
MLP fc2 LoRA: 16 * (3072 + 768) = 61,440

Total ≈ 172,032 trainable parameters per mixer layer
```

If LoRA is also added to output projection:

```text
+ 24,576
Total ≈ 196,608
```

---

## Variant 3: Token-Only MLP-Mixer

### Name

```python
TokenOnlyMLPMixer
```

### Purpose

A very lightweight mixer that only mixes along the token dimension and does not perform channel mixing.

This tests whether simple cross-slice global-token mixing is sufficient.

### Structure

Input:

```python
x: [B, D, T, C]
```

Flatten tokens:

```python
x_flat: [B, N, C], where N = D*T = 80
```

Apply token-mixing MLP across the token dimension:

```python
x_t = x_flat.transpose(1, 2)  # [B, C, N]
x_t = token_mlp(x_t)          # MLP from N -> hidden -> N
x_t = x_t.transpose(1, 2)     # [B, N, C]
delta = x_t
```

Recommended defaults:

```python
N = 80
token_mlp_hidden = 4 * N or 2 * N
gate initialized to 0
```

Pseudo-forward:

```python
def forward(x):
    # x: [B, D, T, C]
    x_input = x
    x_pos = add_slice_and_type_pos(x)

    B, D, T, C = x_pos.shape
    N = D * T

    y = x_pos.reshape(B, N, C)
    y = norm(y)

    y = y.transpose(1, 2)     # [B, C, N]
    y = token_mlp(y)          # mix tokens only
    y = y.transpose(1, 2)     # [B, N, C]

    delta = y.reshape(B, D, T, C)
    return x_input + gate * delta
```

No channel MLP should be used in the default token-only variant.

### Trainable Parameters

For `N=80` and hidden dimension `4N=320`:

```text
80 * 320 + 320 * 80 = 51,200
```

plus biases, positional embeddings, and gate.

---

# Suggested API

Implement a unified wrapper:

```python
class CrossSliceGlobalTokenMixer(nn.Module):
    def __init__(
        self,
        mode: str,
        dim: int = 768,
        num_slices: int = 16,
        num_global_tokens: int = 5,
        num_heads: int = 12,
        bottleneck_dim: int = 128,
        mlp_ratio: float = 4.0,
        lora_rank: int = 16,
        pretrained_block: Optional[nn.Module] = None,
        use_slice_pos_embed: bool = True,
        use_token_type_embed: bool = True,
    ):
        ...
```

Supported modes:

```python
mode = "bottleneck_attention"
mode = "layer_copied_lora"
mode = "token_only_mlp_mixer"
```

The forward method should support:

```python
x = mixer(x)  # x: [B, D, T, C]
```

and optionally:

```python
x = mixer(x, D=D, T=T)  # if x is flattened [B, D*T, C]
```

---

# Integration with Existing DINO Block

The existing `SliceSelfAttentionBlock` should not be heavily rewritten.

Preferred integration:

1. Keep `SliceSelfAttentionBlock` as the standard 2D DINO block.
2. Add a separate module that operates only on collected CLS/register tokens.
3. Insert the mixer between DINO blocks in the outer model forward pass.

Pseudo-code:

```python
for layer_idx, block in enumerate(self.blocks):
    # x: [B, D, num_tokens_per_slice, C]
    # reshape slices into batch dimension for normal 2D DINO
    x_2d = x.reshape(B * D, num_tokens_per_slice, C)

    x_2d = block(x_2d, rope_or_rope_list)

    x = x_2d.reshape(B, D, num_tokens_per_slice, C)

    if layer_idx in self.cross_slice_mixer_layers:
        global_tokens = x[:, :, :num_global_tokens, :]  # [B, D, T, C]

        global_tokens = self.cross_slice_mixers[layer_idx](global_tokens)

        x[:, :, :num_global_tokens, :] = global_tokens
```

Assumption:

```text
CLS/register tokens are placed before patch tokens.
```

If token order is different, adapt the slicing accordingly.

---

# Important Design Constraints

## 1. Do not mix patch tokens directly

This module should only mix CLS/register tokens across slices.

Patch tokens should remain processed by the original 2D DINO slice-wise attention.

## 2. Preserve pretrained behavior at initialization

All cross-slice mixers should use zero-initialized residual gates:

```python
gate = nn.Parameter(torch.zeros(1))
```

At initialization:

```python
x_new == x
```

or as close as possible.

## 3. Avoid using 2D RoPE inside cross-slice mixer

The copied DINO block may expect 2D RoPE for image patch tokens. For cross-slice global tokens, do not reuse the original 2D image RoPE.

Instead use:

```text
slice_pos_embed
token_type_embed
```

or later implement 1D RoPE along the slice dimension.

## 4. Freeze copied DINO base weights in LoRA variant

For `layer_copied_lora`, copied pretrained weights must be frozen. Only LoRA parameters and small adapter parameters should be trainable.

## 5. Keep ablations comparable

All three variants should have the same input/output behavior and should be inserted at the same layers.

---

# Recommended Experiments

Use the following ablation names in logs/configs:

```yaml
cross_slice_mixer: none
cross_slice_mixer: bottleneck_attention
cross_slice_mixer: layer_copied_lora
cross_slice_mixer: token_only_mlp_mixer
```

Recommended comparisons:

```text
1. No cross-slice mixer
2. Bottleneck attention, fully trainable
3. Layer-copied LoRA mixer, rank=16
4. Token-only MLP-Mixer
```

Optional additional ablations:

```text
- insert mixer every layer
- insert mixer every 2 layers
- insert mixer every 3 layers
- CLS-only mixing
- register-only mixing
- CLS + register mixing
- with vs without slice positional embedding
- with vs without token-type embedding
- LoRA on Q/V only vs Q/V + MLP vs Q/V + projection + MLP
```

---

# Expected Main Contribution

The main method should be described as:

```text
We introduce a cross-slice global-token mixing module that adapts a pretrained 2D DINOv3 model to 3D medical volumes. Instead of applying expensive full 3D attention over all patch tokens, the proposed module exchanges volumetric context only through compact CLS and register tokens. The updated global tokens are injected back into subsequent slice-wise DINO layers, enabling progressive 3D context integration with minimal trainable parameters.
```

For the LoRA variant:

```text
We further propose a layer-copied LoRA mixer, where each cross-slice mixer is initialized from the corresponding pretrained DINOv3 transformer block. The copied base weights are frozen, and only low-rank LoRA updates are trained on Q/V and MLP projections. This preserves the pretrained token-mixing prior while learning efficient volumetric adaptation.
```
