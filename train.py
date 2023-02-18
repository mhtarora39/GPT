BLOCK_SIZE = 8
import torch
from tokenizer import Tokenizer

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



