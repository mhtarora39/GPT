import torch
import torch.nn.functional as F

ATTN_SIZE = 3
def self_attn_simplified():
    wei = torch.zeros((ATTN_SIZE,ATTN_SIZE))
    print("Zero Weight Matrix")
    print(wei)
    lower_tril = torch.tril(torch.ones((ATTN_SIZE,ATTN_SIZE)))
    print("Lower triangular matrix!")
    print(lower_tril)
    print("Masking Zero matrix upper triangle with -inf")
    wei = wei.masked_fill(lower_tril == 0,float('-inf'))
    print(wei)
    print("doing softmax")
    wei = F.softmax(wei,dim=-1)
    print(wei)

if __name__ == "__main__":
    self_attn_simplified()

