# src/model.py
import torch
import torch.nn as nn
from transformers import AutoModel

# The incorrect "from src.model import HybridPersonalityModel" line has been removed.

class HybridPersonalityModel(nn.Module):
    """
    This is the correct model architecture that matches the training script.
    It takes text embeddings from a transformer and combines them with linguistic features
    processed by an MLP.
    """
    def __init__(self, n_linguistic_features, model_name='roberta-base'):
        super(HybridPersonalityModel, self).__init__()
        self.transformer = AutoModel.from_pretrained(model_name)
        
        self.linguistic_mlp = nn.Sequential(
            nn.Linear(n_linguistic_features, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(self.transformer.config.hidden_size + 64, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 5) # Outputs raw logits for 5 traits
        )

    def forward(self, input_ids, attention_mask, linguistic_features):
        transformer_output = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        pooled_output = transformer_output.last_hidden_state[:, 0]
        linguistic_output = self.linguistic_mlp(linguistic_features)
        combined_features = torch.cat((pooled_output, linguistic_output), dim=1)
        
        return self.classifier(combined_features)