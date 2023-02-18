BLOCK_SIZE = 8
import torch
import torch.nn as nn
import torch.nn.functional as F

from tokenizer import Tokenizer
Batch = 4

class BigramLanguageModel(nn.Model):

    def __init__(self,vocab_size):
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
            logits = self(idx) #B , T , C
            last_sample = logits[:,-1,:]
            last_sample_prob = F.softmax(last_sample,dim=1)
            idx_next =  torch.multinomial(last_sample_prob,dim=1)
            idx = torch.cat([idx,idx_next])
        return idx


    




if __name__ == "__main__":

    with open("./input.txt",'r',encoding='utf-8') as f:
        text = f.read()

    tokenizer = Tokenizer(text)
    tensor = torch.tensor(tokenizer.encode(text),dtype = torch.long)
    train_len = int(tensor.size().numel()*0.9)
    train_tensor = tensor[:train_len]
    val_tensor = tensor[train_len:]

    print("Below representation how data is formatted")
    train_data = train_tensor[:BLOCK_SIZE]
    target_data = train_tensor[1:BLOCK_SIZE+1]
    for i in range(BLOCK_SIZE):
        print(f"training data {train_data[:i+1]} target data is {target_data[i]}")



