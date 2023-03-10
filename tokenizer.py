import torch

class Tokenizer():
    def __init__(self,text) -> None:
        self.trained_text =  text
        unique_ch = sorted(list(set(text)))
        print("".join(unique_ch))
        print(len(unique_ch))
        self.stoi = {ch : i  for i,ch in enumerate(unique_ch)}
        self.itos = {i : ch  for i,ch in enumerate(unique_ch)}

     
    
    def encode(self,text,dtype=torch.long):
        return [self.stoi[x] for x in text]
    
    def decode(self,seq,dtype=torch.long):
        return "".join([self.itos[x] for x in seq])
    
    

if __name__ == "__main__":
    with open("./input.txt",'r',encoding='utf-8') as f:
        text = f.read()

    tokenizer = Tokenizer(text)
   

    # print("=======ENCODED TEXT======")
    # tensor = tokenizer.encode(text)
    # print(tensor)
    # print("=======DECODED TEXT======")
    # print(tokenizer.decode(tensor))




