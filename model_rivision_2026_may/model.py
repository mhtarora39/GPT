import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from dateFeed import DataLoader

# Determine device: Check environment variable first, then fallback to GPU/CPU detection
env_device = os.environ.get("FORCE_DEVICE")
if env_device:
    device = torch.device(env_device)
else:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Projection(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model
        self.projection = nn.Linear(d_model, d_model)

    def forward(self, x):
        return self.projection(x)


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, block_size):
        super().__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        self.head_dim = d_model // num_heads
        self.q_proj = Projection(d_model)
        self.k_proj = Projection(d_model)
        self.v_proj = Projection(d_model)
        self.linier_transformation = Projection(d_model)
        tril_matrix = torch.tril(torch.ones(block_size, block_size)).bool()
        # Register as buffer (non-trainable, moves with .to(device))
        self.register_buffer('mask', tril_matrix)

    def forward(self, x):
        # Shape = B,T,C = Batch_Size, Time, embedding_dim
        q = self.q_proj(x)  # (B,T,C)
        k = self.k_proj(x)  # (B,T,C)
        v = self.v_proj(x)  # (B,T,C)
        B, T, C = q.shape
        ## QK^T / sqrt(d_k)
        score = (q @ k.transpose(-2, -1)) / self.d_model ** 0.5
        score = score.masked_fill(~self.mask[:T, :T], float('-inf'))
        ## Softmax (turn scores into probabilities)
        probs = torch.softmax(score, dim=-1)  # (B, T, T)
        ## Weight by Value (V)
        out = self.linier_transformation(probs @ v)  # (B, T, C)
        return out


class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, block_size):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.multiheadattention = MultiHeadAttention(d_model, num_heads, block_size)
        self.ln2 = nn.LayerNorm(d_model)
        self.ff1 = nn.Linear(d_model, 4 * d_model)
        self.ff2 = nn.Linear(4 * d_model, d_model)

    def forward(self, x):
        # Pre-norm + residual for attention
        x = self.multiheadattention(self.ln1(x)) + x
        # Pre-norm + residual for feed-forward
        x1 = self.ln2(x)
        x1 = self.ff2(F.gelu(self.ff1(x1)))
        x = x + x1
        return x


class LLMTemplate(nn.Module):
    def __init__(self, vocab_size, block_size, d_model, num_layers, num_heads):
        super().__init__()
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.d_model = d_model

        self.embedding = nn.Embedding(vocab_size, d_model)
        ## Learned positional embeddings: positions 0 -> block_size-1
        self.position_embedding = nn.Embedding(block_size, d_model)
        # Buffer: position indices — not a parameter, but moves with .to(device)
        self.register_buffer("position", torch.arange(block_size, dtype=torch.long))

        self.transformer_blocks = nn.ModuleList(
            [TransformerBlock(d_model, num_heads, block_size) for _ in range(num_layers)]
        )
        self.ln_final = nn.LayerNorm(d_model)
        # Language model head: project from d_model -> vocab_size
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, x):
        B, T = x.shape
        token_embedding = self.embedding(x)              # (B, T, C)
        positions = self.position[:T]
        position_embedding = self.position_embedding(positions)  # (T, C)
        x = token_embedding + position_embedding         # (B, T, C)
        for block in self.transformer_blocks:
            x = block(x)
        x = self.ln_final(x)                             # (B, T, C)
        logits = self.lm_head(x)                         # (B, T, vocab_size)
        return logits


def evaluate(model, loader, block_size, batch_size, vocab_size, criterion, eval_steps=20):
    """Run eval_steps batches on val split and return average loss."""
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for _ in range(eval_steps):
            x, y = loader.get_batch(batch_size, block_size, split='val')
            x = torch.tensor(x, dtype=torch.long, device=device)
            y = torch.tensor(y, dtype=torch.long, device=device)
            logits = model(x)
            logits = logits.reshape(-1, vocab_size)
            y      = y.reshape(-1)
            total_loss += criterion(logits, y).item()
    model.train()
    return total_loss / eval_steps


def train(filepath):
    loader = DataLoader(filepath, val_split=0.2)
    vocab_size = loader.tokenizer.vocab_size

    # ── Hyperparameters ───────────────────────────────────────────────────
    block_size      = 64
    d_model         = 256
    num_layers      = 6
    num_heads       = 8
    batch_size      = 32
    epochs          = 5
    steps_per_epoch = 500   # real updates per epoch (was: 1 — useless)
    lr              = 1e-4
    grad_clip       = 1.0   # prevent exploding gradients
    log_every       = 50    # print loss every N steps
    eval_every      = 200   # run validation every N steps
    eval_steps      = 20    # number of val batches to average over
    # ─────────────────────────────────────────────────────────────────────

    model = LLMTemplate(vocab_size, block_size, d_model, num_layers, num_heads)
    model = model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    global_step = 0
    print(f"Training on {device} | vocab={vocab_size} | steps={epochs * steps_per_epoch}\n")

    model.train()  # set model to training mode

    for epoch in range(epochs):
        epoch_loss = 0.0

        for step in range(steps_per_epoch):
            # ── Get batch (Python lists → tensors on device) ──────────────
            x, y = loader.get_batch(batch_size, block_size, split='train')
            x = torch.tensor(x, dtype=torch.long, device=device)  # (B, T)
            y = torch.tensor(y, dtype=torch.long, device=device)  # (B, T)

            # ── Forward pass ──────────────────────────────────────────────
            logits = model(x)                          # (B, T, vocab_size)
            logits = logits.reshape(-1, vocab_size)    # (B*T, vocab_size)
            y      = y.reshape(-1)                     # (B*T,)
            loss   = criterion(logits, y)

            # ── Backward pass ─────────────────────────────────────────────
            optimizer.zero_grad()
            loss.backward()
            # Gradient clipping: prevents Transformer gradient explosions
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

            epoch_loss  += loss.item()
            global_step += 1

            # ── Step-level logging ────────────────────────────────────────
            if global_step % log_every == 0:
                avg = epoch_loss / (step + 1)
                print(f"Epoch {epoch+1}/{epochs} | Step {global_step} | "
                      f"Loss: {loss.item():.4f} | Avg: {avg:.4f}")

            # ── Periodic validation ───────────────────────────────────────
            if global_step % eval_every == 0:
                val_loss = evaluate(model, loader, block_size, batch_size,
                                    vocab_size, criterion, eval_steps)
                print(f"  >> Val Loss: {val_loss:.4f}  (step {global_step})")
                model.train()

        epoch_avg = epoch_loss / steps_per_epoch
        print(f"\n=== Epoch {epoch+1} done | Avg Train Loss: {epoch_avg:.4f} ===\n")


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python model.py <path_to_dataset.txt>")
        sys.exit(1)
    train(sys.argv[1])
