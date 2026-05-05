

class Tokenizer:
    def __init__(self, text=""):
        # Find all unique characters in the provided dataset/text
        chars = sorted(list(set(text)))
        self.vocab_size = len(chars)
        
        # Create lookup dictionaries: char-to-int (stoi) and int-to-char (itos)
        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for i, ch in enumerate(chars)}
        
    def encode(self, text):
        # Convert a string into a list of integers
        return [self.stoi[c] for c in text]  
    
    def decode(self, int_list):
        # Convert a list of integers back into a string
        return ''.join([self.itos[i] for i in int_list])