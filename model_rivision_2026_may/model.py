import os
import torch
import torch.nn as nn
import torch.nn.functional as F

# Determine device: Check environment variable first, then fallback to GPU/CPU detection
env_device = os.environ.get("FORCE_DEVICE")
if env_device:
    device = torch.device(env_device)
else:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class  Projection(nn.Module):
    def __init__(self,d_model):
        super().__init__()
        self.d_model = d_model
        self.projection = nn.Linear(d_model,d_model)
    
    
    def forward(self,x):
        return self.projection(x) 


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads,block_size):
        super().__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        ##PlcaeHolder for self.mask in projection layer 
        self.head_dim = d_model // num_heads
        self.q_proj = Projection(d_model)
        self.k_proj = Projection(d_model)
        self.v_proj = Projection(d_model)
        tril_matrix = torch.tril(torch.ones(block_size,block_size)).bool()
        self.linier_transformation = Projection(d_model)
        # 2. Register it as a buffer (non-trainable)
        self.register_buffer('mask', tril_matrix)
    
    def forward(self, x):
        #Shape = B,T,C = Batch_Size, Time, embedding_dim
        #embedding_dim = C
        q = self.q_proj(x) # (B,T,C)
        k = self.k_proj(x) # (B,T,C)
        v = self.v_proj(x) # (B,T,C)
        B,T,C = q.shape
        ##QK^T / d_k ** 0.5
        score = (q @ k.transpose(-2,-1))/self.d_model **0.5
        score = score.masked_fill(~self.mask[:T, :T], float('-inf'))
        ## Softmax (to turn scores into probabilities)
        probs = torch.softmax(score, dim=-1) # (B, T, T)
        ## Weight by Value (V)
        out = self.linier_transformation(probs @ v) # (B, T, C)

        return out
        

class TrasFormerBlock(nn.Module):
    def __init__(self, d_model, num_heads,block_size):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)  
        self.multiheadattention = MultiHeadAttention(d_model, num_heads,block_size)
        self.FeedForward = nn.Linear(d_model,4*d_model)
        self.ln2 = nn.LayerNorm(d_model)  
        self.FeedForward2 = nn.Linear(4*d_model,d_model)
    
    def forward(self,x):
        x = self.multiheadattention(self.ln1(x)) + x
        x1 = self.ln2(x)
        x1 = self.FeedForward(x1)
        x1 = F.gelu(x1)
        x1 = self.FeedForward2(x1)
        x = x + x1
        return x
        
    

class LLMTemplate(nn.Module):  
    def __init__(self, vocab_size, block_size, d_model):
        # Crucial step: Initialize the parent nn.Module class
        super().__init__()
        
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.d_model = d_model
        
        self.embedding = nn.Embedding(vocab_size, d_model)
        
        ## Let's learn position embeddings
        self.position_embedding = nn.Embedding(block_size, d_model)
        
        # register_buffer saves this tensor as part of the module state.
        # We also enforce it is created on our global device.
        self.register_buffer("position", torch.arange(block_size, dtype=torch.long, device=device))
        self.k_proj = Projection(d_model)
        self.q_proj = Projection(d_model)
        self.v_proj = Projection(d_model)
        
    def forward(self, x):
        B, T = x.shape
        
        token_embedding = self.embedding(x)
        positions = self.position[:T]
        position_embedding = self.position_embedding(positions)
        x = token_embedding + position_embedding
        
        
