# src/utils.py

import torch
import pandas as pd

def get_class_weights(df: pd.DataFrame, label_columns: list, device: str) -> torch.Tensor:
    """
    Calculates class weights to handle imbalanced datasets for multi-label classification.
    The weight for a class is the ratio of non-occurrences to occurrences.
    """
    total_samples = len(df)
    
    # ✅ CHANGED: Explicitly convert the label columns to a numeric type before summing.
    # This prevents the TypeError by ensuring class_sums is a series of numbers, not strings.
    class_sums = df[label_columns].astype(float).sum(axis=0)
    
    # Calculate weight for the positive class (1) for each label
    pos_weight = (total_samples - class_sums) / class_sums
    
    # Convert to a tensor to be used in the loss function
    return torch.tensor(pos_weight.values, dtype=torch.float, device=device)