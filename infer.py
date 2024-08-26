import onnx
import sys
import torch
import torch.nn.functional as F
import onnxruntime
import numpy as np

from dataloader import BLOCK_SIZE, DataLoader
data_loader = DataLoader('./input.txt',32)
path = "./assets/GPTNANO.onnx"
BATCH_SIZE = 32


if len(sys.argv) > 1: 
    path = sys.argv[1]


device = 'cuda' if torch.cuda.is_available() else 'cpu'
session = onnxruntime.InferenceSession(path)

def extrapolate(inputs):
    inputs = np.array([inputs*BLOCK_SIZE]*BATCH_SIZE).astype(np.int64)
    return inputs
    
def softmax(x,axis):
    return np.exp(x) / np.sum(np.exp(x), axis=axis, keepdims=True)


def run(input_data):
    output = session.run([], {'input': input_data})
    return output[0]

def generate(idx,num_sample):
    idx = extrapolate(idx)
    for i in range(num_sample):
         
        logits = run(idx[:,-BLOCK_SIZE:]) #B , T , C
        #Logits = B,T,VOCAB_SIZE

        last_sample = logits[:,-1,:] # B,C
        last_sample_prob = softmax(last_sample,axis=-1) # B,C
        idx_next =  np.random.choice(np.argsort(last_sample_prob[0])[::-1][2:7], size=1, replace=False)
        idx[:,i] = idx_next
    return idx
generated_idx = generate(idx = [7], num_sample=100)
output = data_loader.tokenizer.decode(generated_idx[0])
print(output)
