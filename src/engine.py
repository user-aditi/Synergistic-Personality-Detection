# src/engine.py

import torch
from tqdm.auto import tqdm

def train_epoch(model, data_loader, loss_fn, optimizer, device, scheduler):
    """Performs one complete training epoch."""
    model.train()
    total_loss = 0
    for batch in tqdm(data_loader, desc="Training"):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        linguistic_features = batch['linguistic_features'].to(device)
        labels = batch['labels'].to(device)

        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask, linguistic_features)
        loss = loss_fn(outputs, labels)
        total_loss += loss.item()
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) # Gradient clipping
        optimizer.step()
        scheduler.step() # Update learning rate

    return total_loss / len(data_loader)

def eval_model(model, data_loader, loss_fn, device):
    """Evaluates the model on a validation or test set."""
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            linguistic_features = batch['linguistic_features'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(input_ids, attention_mask, linguistic_features)
            loss = loss_fn(outputs, labels)
            total_loss += loss.item()
            
    return total_loss / len(data_loader)
