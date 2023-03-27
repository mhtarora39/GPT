BLOCK_SIZE = 8
import torch
import torch.nn as nn
import torch.nn.functional as F

from tokenizer import Tokenizer
from dataloader import DataLoader , BLOCK_SIZE

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Running on {device}")
torch.manual_seed(1337)
dropout = 0.2
num_head=6
block_layer_size=6

class Head(nn.Module):
    def __init__(self,n_embed,head_size):
        super().__init__()
        self.head_size = head_size
        self.key   = nn.Linear(n_embed,head_size,bias=False)
        self.query = nn.Linear(n_embed,head_size,bias=False)
        self.value = nn.Linear(n_embed,head_size,bias=False)
        self.register_buffer('trill',torch.tril(torch.ones(BLOCK_SIZE,BLOCK_SIZE)))
        self.dropout = nn.Dropout(dropout)

    def forward(self,x):
        B,T,C = x.shape
        #print(x.shape)
        key = self.key(x)
        query = self.query(x)
        weight = query@key.transpose(-2,-1) + C**-0.5
        weight = weight.masked_fill(self.trill[:T, :T] ==0 ,float('-inf'))
        weight = F.softmax(weight,dim=-1)
        weight = self.dropout(weight)
        v = self.value(x)
        out = weight @ v
        return out

class MULTI_HEAD(nn.Module):

    def __init__(self,n_embed,head_size,num_head):
        super().__init__()
        self.heads = nn.ModuleList([Head(n_embed,head_size) for _ in range(num_head)])
        self.proj  = nn.Linear(n_embed,n_embed)
        self.dropout = nn.Dropout(dropout)

    def forward(self,x):
        x = torch.cat([h(x) for h in self.heads],dim = -1)
        return self.dropout(self.proj(x))
        
class FeedForward(nn.Module):
    def __init__(self,n_in,n_out):
        super().__init__()
        self.layer = nn.Sequential(nn.Linear(n_in,n_out),
                                   nn.ReLU(),
                                   nn.Linear(n_in,n_out),
                                   nn.Dropout(dropout)) 
        
    def forward(self,x):
        return self.layer(x)
    
class MultiHeadBlock(nn.Module):
    def __init__(self,embed_size,head_size,num_head) -> None:
        super().__init__()
        self.ff = FeedForward(embed_size,embed_size)
        self.multi_head = MULTI_HEAD(embed_size,head_size,num_head)
        self.lm_sa = nn.LayerNorm(embed_size)
        self.lm_ff = nn.LayerNorm(embed_size)
    
    def forward(self,x):
        x = x + self.multi_head(self.lm_sa(x))
        x = x + self.ff(self.lm_ff(x))
        return x

class BigramLanguageModel(nn.Module):

    def __init__(self,data_loader,n_embd=384):
        super().__init__()

        self.data_loader   = data_loader
        self.vocab_size    = data_loader.vocab_size
        # Embedding size
        self.n_embd        = n_embd
        # Create Matrix of the self.vocab size x self.n_embed
        self.embeddings    = nn.Embedding(self.vocab_size,self.n_embd)
        # Block Size X embedding layer as this will add notion of space in self attentions
        self.pos_embedding = nn.Embedding(BLOCK_SIZE,self.n_embd) 
        # Take the embedding and change it vocab size vector 
        self.lm_head      = nn.Linear(self.n_embd,self.vocab_size)
        # Mask softmax head
        self.multi_head    = nn.Sequential(*[MultiHeadBlock(n_embd,n_embd//num_head,num_head) for _ in range(block_layer_size)])
        self.ln_head       = nn.LayerNorm(n_embd)
        self.drop_out      = nn.Dropout(dropout)


    @torch.no_grad()
    def estimate_loss(self,eval_iters):
        out = {}
        self.eval()
        for split in ["train","test"]:
            losses  = torch.zeros(eval_iters)
            for i in range(eval_iters):
                x,y = self.data_loader.get_batch(split,device)
                _, loss = self(x,y)
                losses[i] =  loss.item()
            out[split+"_loss"] = losses.mean()
        self.train()
        return out     

    def forward(self,index,target=None):
        # Here T is time and B is Batch
        # pytorch forward method changing inputs
        # if len(index) == 1:
        #     index,target  = data_loader.get_batch('train',device) 
        B,T           = index.shape
            
        ## Mapping each token id to token embeddings
        tok_embd      = self.embeddings(index) #B,T,C
        ## Each location (0,t-1) feeds into and there spacial arrangement will be captured.
        ## As 0 will represent the 0th position and 
        pos_embedding = self.pos_embedding(torch.arange(T,device=device)) #TXC
        final_embedding = pos_embedding + tok_embd
        # print(final_embedding.shape)
        final_embedding = self.multi_head(final_embedding)
        final_embedding = self.ln_head(final_embedding)
        #final_embedding = self.drop_out(final_embedding)

        logits    = self.lm_head(final_embedding)   
        loss      = None

        if target is not None:
            B,T,C   = logits.shape
            target  = target.view(B*T)
            logits  = logits.view(B*T,C)
            loss    = F.cross_entropy(logits,target)
        return logits ,  loss
    
    def generate(self,idx,num_sample):
        for _ in range(num_sample):
            logits , _ = self(idx[:,-BLOCK_SIZE:]) #B , T , C
            last_sample = logits[:,-1,:] # B,C
            last_sample_prob = F.softmax(last_sample,dim=-1) # B,C
            idx_next =  torch.multinomial(last_sample_prob,num_samples=1) #B,1
            idx = torch.cat([idx,idx_next],dim=1)
        return idx

if __name__ == "__main__":
    data_loader = DataLoader('./input.txt',32)

    model = BigramLanguageModel(data_loader)
    model = model.to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(),lr=3e-4)
    
    
    train_iter = 15000
    for i in range(train_iter):
        x, y  = data_loader.get_batch('train',device)
        logits, loss = model(x,y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if i%1000 == 0:
            print(model.estimate_loss(100))
    import pdb;pdb.set_trace()
    # logits, loss = model(x)
    # print(logits.shape)
    # print(loss)
    print(data_loader.tokenizer.decode(model.generate(idx = torch.zeros((1, 1), dtype=torch.long,device=device), num_sample=100)[0].tolist()))


