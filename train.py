BLOCK_SIZE = 8
import torch
import torch.nn as nn
import torch.nn.functional as F

from tokenizer import Tokenizer
from dataloader import DataLoader

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Running on {device}")
torch.manual_seed(1337)


class BigramLanguageModel(nn.Module):

    def __init__(self,data_loader):
        super().__init__()
        self.data_loader = data_loader
        self.vocab_size  = data_loader.vocab_size
        self.embeddings = nn.Embedding(self.vocab_size,self.vocab_size)

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
        logits  = self.embeddings(index) #B,T,C
        loss    = None

        if target is not None:
            B,T,C   = logits.shape
            target  = target.view(B*T)
            logits  = logits.view(B*T,C)
            loss    = F.cross_entropy(logits,target)
        return logits ,  loss
    
    def generate(self,idx,num_sample):
        for _ in range(num_sample):
            logits , loss = self(idx) #B , T , C
            last_sample = logits[:,-1,:] # B,C
            last_sample_prob = F.softmax(last_sample,dim=1) # B,C
            idx_next =  torch.multinomial(last_sample_prob,num_samples=1) #B,1
            idx = torch.cat([idx,idx_next],dim=-1)
        return idx


    




if __name__ == "__main__":
    data_loader = DataLoader('./input.txt',1)

    model = BigramLanguageModel(data_loader)
    model = model.to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(),lr=1e-3)
    
    
    train_iter = 100000
    for i in range(train_iter):
        x, y  = data_loader.get_batch('train',device)
        logits, loss = model(x,y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if i%1000 == 0:
            print(model.estimate_loss(100))
    
    # logits, loss = model(x)
    # print(logits.shape)
    # print(loss)
    print(data_loader.tokenizer.decode(model.generate(idx = torch.zeros((1, 1), dtype=torch.long,device=device), num_sample=100)[0].tolist()))


