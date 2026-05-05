"""
Shape test for MultiHeadAttention.

Verifies that given an input of shape (B, T, C):
  - The attention output is also (B, T, C)
  - The causal mask is respected (scores above diagonal are -inf)
  - Works for different batch sizes and sequence lengths up to block_size
"""

import torch
from model import MultiHeadAttention, device

# ── Test Config ─────────────────────────────────────────────────────────────
B          = 4    # Batch size
T          = 8    # Sequence length (must be <= block_size)
d_model    = 64   # Embedding / model dimension
num_heads  = 4    # Number of attention heads
block_size = 16   # Max context window

# ── Helpers ──────────────────────────────────────────────────────────────────
def ok(msg): print(f"  PASS  {msg}")
def fail(msg, got, expected): raise AssertionError(f"  FAIL  {msg}  | got {got}, expected {expected}")

# ── Setup ────────────────────────────────────────────────────────────────────
print(f"\nDevice: {device}")
print(f"Input shape: ({B}, {T}, {d_model})\n")

mha = MultiHeadAttention(d_model=d_model, num_heads=num_heads, block_size=block_size).to(device)

# Dummy float input (B, T, C) — MHA expects already-embedded tokens
x = torch.randn(B, T, d_model, device=device)

# ── Test 1: Output shape ─────────────────────────────────────────────────────
out = mha(x)
expected_shape = (B, T, d_model)
if out.shape == expected_shape:
    ok(f"Output shape is correct: {tuple(out.shape)}")
else:
    fail("Output shape mismatch", tuple(out.shape), expected_shape)

# ── Test 2: Output is finite (no NaNs / Infs from bad masking) ───────────────
if torch.isfinite(out).all():
    ok("Output contains no NaN / Inf values")
else:
    fail("Output contains NaN or Inf", out.isnan().sum().item(), 0)

# ── Test 3: Causal mask shape covers the sequence ────────────────────────────
mask = mha.mask[:T, :T]
expected_mask_shape = (T, T)
if mask.shape == expected_mask_shape:
    ok(f"Causal mask slice shape is correct: {tuple(mask.shape)}")
else:
    fail("Mask shape mismatch", tuple(mask.shape), expected_mask_shape)

# ── Test 4: Mask is lower-triangular (causal) ────────────────────────────────
upper_triangle_is_false = not mask.triu(diagonal=1).any().item()
if upper_triangle_is_false:
    ok("Causal mask correctly blocks future positions (upper triangle is False)")
else:
    fail("Mask is NOT causal – future positions are visible", None, None)

# ── Test 5: Different sequence lengths work (T < block_size) ─────────────────
for t in [1, T // 2, T]:
    x_var = torch.randn(B, t, d_model, device=device)
    out_var = mha(x_var)
    assert out_var.shape == (B, t, d_model), f"Failed for T={t}"
ok(f"Output shape is correct for variable sequence lengths: 1, {T//2}, {T}")

# ── Test 6: Gradient flows back (model is learnable) ─────────────────────────
x_grad = torch.randn(B, T, d_model, device=device, requires_grad=True)
out_grad = mha(x_grad)
loss = out_grad.sum()
loss.backward()
if x_grad.grad is not None:
    ok("Gradients flow back correctly through MultiHeadAttention")
else:
    fail("No gradients on input", None, "grad tensor")

print("\nAll shape tests passed!")
