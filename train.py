BLOCK_SIZE = 8
import torch
import torch.nn as nn
import torch.nn.functional as F

from tokenizer import Tokenizer
from dataloader import DataLoader

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Running on {device}")
class BigramLanguageModel(nn.Module):

    def __init__(self,vocab_size):
        super().__init__()
        self.embeddings = nn.Embedding(vocab_size,vocab_size)

    def forward(self,index,target=None):
        logits  = self.embeddings(index) #B,T,C
        loss    = None

        if target:
            B,T,C   = logits.shape
            targets = targets.view(B*T)
            logits  = logits.view(B*T,C)
            loss    = F.cross_entropy(logits,target)
        return logits ,  loss
    
    def generate(self,idx,num_sample):
        for _ in range(num_sample):
            logits , loss = self(idx) #B , T , C
            last_sample = logits[:,-1,:]
            last_sample_prob = F.softmax(last_sample,dim=1)
            idx_next =  torch.multinomial(last_sample_prob,num_samples=1)
            idx = torch.cat([idx,idx_next])
        return idx


    




if __name__ == "__main__":
    data_loader = DataLoader('./input.txt',1)

    model = BigramLanguageModel(data_loader.vocab_size)
    model = model.to(device)
    x, y  = data_loader.get_batch('train',device)

    logits, loss = model(x)
    print(logits.shape)
    print(loss)
    print(data_loader.tokenizer.decode(model.generate(idx = torch.zeros((1, 1), dtype=torch.long,device=device), num_sample=100)[0].tolist()))


