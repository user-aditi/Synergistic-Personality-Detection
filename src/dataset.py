# src/dataset.py

import torch
from torch.utils.data import Dataset

class PersonalityDataset(Dataset):
    """
    Custom PyTorch Dataset to handle text, enhanced linguistic features, and labels.
    This class structures the data for the DataLoader.
    """
    def __init__(self, texts, linguistic_features, labels, tokenizer, max_len=256):
        self.texts = texts
        self.linguistic_features = torch.tensor(linguistic_features, dtype=torch.float)
        
        # ✅ CHANGED: Explicitly convert labels to a float type before creating the tensor.
        # This resolves the TypeError by ensuring the data is in a supported numerical format.
        self.labels = torch.tensor(labels.astype(float).values, dtype=torch.float)
        
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts.iloc[idx])
        linguistic_vec = self.linguistic_features[idx]
        label = self.labels[idx]

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'linguistic_features': linguistic_vec,
            'labels': label
        }