from datasets import load_dataset
from transformers import AutoTokenizer
import torch
from torch.utils.data import Dataset, DataLoader, Subset



class IMDbDataset(Dataset):
    """
    Custom dataset class for IMDB movie review
    """
    
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels
        
    def __getitem__(self, idx):
        # return a dict of input_ids, attetion_mask, and labels
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item
    
    def __len__(self):
        return len(self.labels)
    
def load_and_preprocess_data(model_name:str = 'distilbert-base-uncased', max_length: int = 128):
        """
        Load the IMDb dataset, tokenizes it, and create Pytorch DataLoaders
        
        Args:
            model_name
            max_length: the maximum sequence length for  tokenization

        returns:
            tuple: A tuple contains:
                - train_dataloader
                - val_dataloader
                - test_dataloader
                - tokenizer
                
        """
        # Load the dataset from HuggingFace dataset library
        dataset = load_dataset('imdb')
        
        # Load the tokenizer based on the chosen model
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        def tokenize_function(examples):
            return tokenizer(examples['text'], truncation=True, padding = 'max_length', max_length = max_length)
        
        # Apply tokenization on the entire dataset
        tokenized_dataset = dataset.map(tokenize_function, batched=True)
        
        # Rename to label to match pytorch naming expectations
        tokenized_dataset = tokenized_dataset.rename_column("label", "labels")
        
        # Set the correct format
        tokenized_dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
        
        # Create PyTorch dataset instances
        train_dataset = IMDbDataset(
            encodings= {'input_ids': tokenized_dataset['train']['input_ids'],
                        'attention_mask': tokenized_dataset['train']['attention_mask']},
            labels = tokenized_dataset['train']['labels']
        )
        
        # For simplicity, we'll use the 'test' split as our validation set during training
        # and then as a final test set. In a real scenario, you might split 'train' further.
        
        val_dataset = IMDbDataset(
            encodings= {'input_ids': tokenized_dataset['test']['input_ids'],
                        'attention_mask': tokenized_dataset['test']['attention_mask']},
            labels= tokenized_dataset['test']['labels']
        )
        
        test_dataset= IMDbDataset(
            encodings={'input_ids': tokenized_dataset['test']['input_ids'],
                      'attention_mask': tokenized_dataset['test']['attention_mask']},
            labels= tokenized_dataset['test']['labels']
        )

        # batch size
        batch_size = 16
        
        # Only a small subset
        # Take only 10 examples from each dataset
        train_subset = Subset(train_dataset, list(range(64)))
        val_subset = Subset(val_dataset, list(range(64)))
        test_subset = Subset(test_dataset, list(range(64)))
        
        # crate DataLoader
        train_dataloader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
        val_dataloader = DataLoader(val_subset, batch_size=batch_size, shuffle=True)
        test_dataloader = DataLoader(test_subset, batch_size=batch_size, shuffle=True)
        
        print("Loading and pre-processing is completed")
        return train_dataloader, val_dataloader, test_dataloader, tokenizer
    
if __name__ == "__main__":
    # Example of Usage
    train_dl, val_dl, test_dl, tokinizer_obj = load_and_preprocess_data()
    print(f"Number of training  batches: {len(train_dl)}")
    print(f"Number of validation batches: {len(val_dl)}")
    print(f"Number of test batches: {len(test_dl)}")
    
    
    # Inspect one batch
    for batch in train_dl:
        print("\nSample batch structure:")
        print(f"Input IDs shape: {batch['input_ids'].shape}")
        print(f"Attention mask shape: {batch['attention_mask'].shape}")
        print(f"Label shape: {batch['Labels'].shape}")
        print(f"First 5 Input IDs :{batch['input_ids'][0, :5]}")
        print(f"First 5 labels:{batch['labels'][0:5]}")
        
    # decode tokens back to text
    print("\nDecoded first sample:")
    sample_text = tokinizer_obj.decode(batch['input_ids'][0], skip_special_tokens=True)
    print(sample_text)