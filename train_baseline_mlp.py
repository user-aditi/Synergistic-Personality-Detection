# train_baseline_mlp.py
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import AdamW
from sklearn.metrics import classification_report
import numpy as np
from tqdm.auto import tqdm

# Import config for batch size
from config import MODEL_CONFIG, OCEAN_TRAITS

def run_baseline_training():
    print("--- 🧠 Starting Baseline MLP Training ---")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # --- Step 1: Load Pre-processed Data ---
    print("Step 1: Loading pre-processed embeddings and labels...")
    train_embeddings = torch.load('models/train_embeddings.pt')
    train_labels = torch.load('models/train_labels.pt')
    val_embeddings = torch.load('models/val_embeddings.pt')
    val_labels = torch.load('models/val_labels.pt')
    test_embeddings = torch.load('models/test_embeddings.pt')
    test_labels = torch.load('models/test_labels.pt')

    # Create TensorDatasets and DataLoaders
    train_dataset = TensorDataset(train_embeddings, train_labels)
    val_dataset = TensorDataset(val_embeddings, val_labels)
    test_dataset = TensorDataset(test_embeddings, test_labels)

    train_loader = DataLoader(train_dataset, batch_size=MODEL_CONFIG['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=MODEL_CONFIG['batch_size'])
    test_loader = DataLoader(test_dataset, batch_size=MODEL_CONFIG['batch_size'])
    print("✅ Data loaded successfully.")

    # --- Step 2: Define the MLP Model ---
    class SimpleMLP(nn.Module):
        def __init__(self, input_size, output_size=5):
            super(SimpleMLP, self).__init__()
            self.layers = nn.Sequential(
                nn.Linear(input_size, 256),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(128, output_size)
            )
        
        def forward(self, x):
            return self.layers(x)

    # The input size is the dimension of our RoBERTa embeddings (768)
    model = SimpleMLP(input_size=train_embeddings.shape[1]).to(device)
    print("✅ MLP Model defined.")

    # --- Step 3: Training Loop ---
    print("Step 3: Starting training...")
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = AdamW(model.parameters(), lr=1e-4)
    epochs = 20 # We can use more epochs since training is fast
    best_val_loss = float('inf')
    early_stop_counter = 0
    patience = 5 # More patience

    for epoch in range(epochs):
        model.train()
        total_train_loss = 0
        for embeddings, labels in train_loader:
            embeddings, labels = embeddings.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(embeddings)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()
        
        avg_train_loss = total_train_loss / len(train_loader)

        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for embeddings, labels in val_loader:
                embeddings, labels = embeddings.to(device), labels.to(device)
                outputs = model(embeddings)
                loss = loss_fn(outputs, labels)
                total_val_loss += loss.item()
        
        avg_val_loss = total_val_loss / len(val_loader)
        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), 'models/baseline_mlp_model.pth')
            print(f"✔️ Validation loss improved. Model saved.")
            early_stop_counter = 0
        else:
            early_stop_counter += 1
            if early_stop_counter >= patience:
                print("--- Early stopping triggered. ---")
                break
    
    # --- Step 4: Final Evaluation ---
    print("\nStep 4: Evaluating best model on the test set...")
    model.load_state_dict(torch.load('models/baseline_mlp_model.pth'))
    model.eval()

    all_preds = []
    all_labels = []
    with torch.no_grad():
        for embeddings, labels in test_loader:
            embeddings, labels = embeddings.to(device), labels.to(device)
            outputs = model(embeddings)
            preds = torch.sigmoid(outputs) > 0.5
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    ocean_cols = list(OCEAN_TRAITS.keys())

    print("\n--- 📊 Final Performance Report (Baseline MLP) ---")
    print(classification_report(all_labels, all_preds, target_names=ocean_cols, zero_division=0))
    print("--- ✅ Baseline training complete! ---")


if __name__ == "__main__":
    run_baseline_training()