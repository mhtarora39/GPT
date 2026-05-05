#import torch
from tokenizer import Tokenizer
import random 

class DataLoader:
    def __init__(self,filepath,val_split):
        self.text = self.load_data(filepath)
        self.tokenizer = Tokenizer(self.text)
        self.train_data = self.tokenizer.encode(self.text[:int(len(self.text)*(1-val_split))])
        self.val_data = self.tokenizer.encode(self.text[int(len(self.text)*(1-val_split)):])
        

    def load_data(self, filepath):
        with open(filepath, 'r', encoding='utf-8') as f:
            data = f.read()
        return data
    
    def get_batch(self, batch_size, block_size, split='train'):
        # Get a batch of data for the specified split
        data = self.train_data if split == 'train' else self.val_data
        
        # Check for out of range errors before generating indices
        if len(data) <= block_size:
            raise ValueError(f"Dataset split '{split}' is too small (length {len(data)}) for block_size {block_size}.")
            
        x = []
        y = []
        for b in range(batch_size):
            ix = random.randint(0, len(data) - block_size - 1)
            x.append(data[ix:ix+block_size])
            y.append(data[ix+1:ix+block_size+1])

        return x, y

if __name__ == "__main__":
    import os
    import sys
    
    if len(sys.argv) > 1:
        test_file = sys.argv[1]
        cleanup_needed = False
    else:
        test_file = "dummy_dataset.txt"
        cleanup_needed = True
        sample_text = "Hello world! This is a simple test dataset for our GPT dataloader. We need enough characters to test both the training and validation splits properly."
        with open(test_file, "w", encoding="utf-8") as f:
            f.write(sample_text)
            
    # 2. Initialize DataLoader
    print(f"Initializing DataLoader with dataset: {test_file}")
    loader = DataLoader(filepath=test_file, val_split=0.2)
    
    print(f"Total Text Length: {len(loader.text)}")
    print(f"Train Data Tokens: {len(loader.train_data)}")
    print(f"Val Data Tokens:   {len(loader.val_data)}")
    
    # 3. Get a batch
    batch_size = 2
    block_size = 5
    x_t, y_t = loader.get_batch(batch_size, block_size, split='train')
    x_v, y_v = loader.get_batch(batch_size, block_size, split='val')
    
    print(f"\n--- Batch Output (Batch Size={batch_size}, Block Size={block_size}) ---")
    print(f"x_train: {x_t}")
    print(f"y_train: {y_t}")
    print(f"x_val:   {x_v}")
    print(f"y_val:   {y_v}")
    
    # 4. Cleanup
    if cleanup_needed:
        os.remove(test_file)
    print("\nTest completed successfully!")
