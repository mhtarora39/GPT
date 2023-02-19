import torch
from tokenizer import Tokenizer
BLOCK_SIZE = 8

class DataLoader:
    def __init__(self, file_path, batch=1,split=0.9) -> None:
        with open(file_path,'r',encoding='utf-8') as f:
            text = f.read()
        self.tokenizer = Tokenizer(text)
        tensor = self.tokenizer.encode(text)
        self.vocab_size = len(set(tensor))
        train_len = int(len(tensor)*split)
        self.train_data = torch.tensor(tensor[:train_len],dtype=torch.long)
        self.val_data = torch.tensor(tensor[train_len:],dtype=torch.long)
        self.batch = batch
        

    def get_batch(self,split):
        # generate a small batch of data of inputs x and targets y
        data = self.train_data if split == 'train' else self.val_data
        ix = torch.randint(len(data) - BLOCK_SIZE, (self.batch,))
        x = torch.stack([data[i:i+BLOCK_SIZE] for i in ix])
        y = torch.stack([data[i+1:i+BLOCK_SIZE+1] for i in ix])
        return x, y
    
    def get_vocab_size(self):
        return self.vocab_size
    
