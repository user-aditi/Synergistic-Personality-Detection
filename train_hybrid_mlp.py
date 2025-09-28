# train_hybrid_mlp.py
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import AdamW
from sklearn.metrics import classification_report
import numpy as np
from tqdm.auto import tqdm
import joblib
from sklearn.model_selection import train_test_split
from datasets import load_dataset

# Import your feature extractor and config
from src.features import extract_enhanced_linguistic_features
from config import MODEL_CONFIG, OCEAN_TRAITS

def run_hybrid_training():
    print("--- 🧠 Starting Improved Hybrid MLP Training ---")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # --- Step 1: Load All Data (Embeddings + Linguistic Features) ---
    print("Step 1: Loading all data sources...")
    # Load pre-processed embeddings and labels
    train_embeddings = torch.load('models/train_embeddings.pt')
    train_labels = torch.load('models/train_labels.pt')
    val_embeddings = torch.load('models/val_embeddings.pt')
    val_labels = torch.load('models/val_labels.pt')
    test_embeddings = torch.load('models/test_embeddings.pt')
    test_labels = torch.load('models/test_labels.pt')

    # Load and process linguistic features
    dataset = load_dataset("jingjietan/essays-big5", split="train")
    df = dataset.to_pandas()
    X = df['text']
    y = df[OCEAN_TRAITS.keys()]

    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    print("Extracting and scaling linguistic features...")
    X_train_ling = extract_enhanced_linguistic_features(X_train)
    X_val_ling = extract_enhanced_linguistic_features(X_val)
    X_test_ling = extract_enhanced_linguistic_features(X_test)
    
    feature_columns = joblib.load('models/feature_columns.pkl')
    X_train_ling = X_train_ling.reindex(columns=feature_columns, fill_value=0)
    X_val_ling = X_val_ling.reindex(columns=feature_columns, fill_value=0)
    X_test_ling = X_test_ling.reindex(columns=feature_columns, fill_value=0)

    scaler = joblib.load('models/linguistic_feature_scaler.pkl')
    X_train_scaled = torch.tensor(scaler.transform(X_train_ling), dtype=torch.float)
    X_val_scaled = torch.tensor(scaler.transform(X_val_ling), dtype=torch.float)
    X_test_scaled = torch.tensor(scaler.transform(X_test_ling), dtype=torch.float)

    # Combine embeddings and linguistic features
    train_combined = torch.cat((train_embeddings, X_train_scaled), dim=1)
    val_combined = torch.cat((val_embeddings, X_val_scaled), dim=1)
    test_combined = torch.cat((test_embeddings, X_test_scaled), dim=1)
    
    # Create TensorDatasets and DataLoaders
    train_dataset = TensorDataset(train_combined, train_labels)
    val_dataset = TensorDataset(val_combined, val_labels)
    test_dataset = TensorDataset(test_combined, test_labels)

    train_loader = DataLoader(train_dataset, batch_size=MODEL_CONFIG['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=MODEL_CONFIG['batch_size'])
    test_loader = DataLoader(test_dataset, batch_size=MODEL_CONFIG['batch_size'])
    print("✅ All data loaded and combined successfully.")

    # --- Step 2: Define the Hybrid MLP Model ---
    class HybridMLP(nn.Module):
        def __init__(self, input_size, output_size=5):
            super(HybridMLP, self).__init__()
            self.layers = nn.Sequential(
                nn.Linear(input_size, 512),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(256, output_size)
            )
        
        def forward(self, x):
            return self.layers(x)

    model = HybridMLP(input_size=train_combined.shape[1]).to(device)
    print("✅ Hybrid MLP Model defined.")

    # --- Step 3: Training Loop ---
    print("Step 3: Starting training...")
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = AdamW(model.parameters(), lr=1e-4)
    epochs = 20
    best_val_loss = float('inf')
    early_stop_counter = 0
    patience = 5

    for epoch in range(epochs):
        model.train()
        for features, labels in train_loader:
            features, labels = features.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(features)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()
        
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for features, labels in val_loader:
                features, labels = features.to(device), labels.to(device)
                outputs = model(features)
                loss = loss_fn(outputs, labels)
                total_val_loss += loss.item()
        
        avg_val_loss = total_val_loss / len(val_loader)
        print(f"Epoch {epoch+1}/{epochs} | Val Loss: {avg_val_loss:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), 'models/hybrid_mlp_model.pth')
            print(f"✔️ Validation loss improved. Model saved.")
            early_stop_counter = 0
        else:
            early_stop_counter += 1
            if early_stop_counter >= patience:
                print("--- Early stopping triggered. ---")
                break
    
    # --- Step 4: Final Evaluation ---
    print("\nStep 4: Evaluating best hybrid model on the test set...")
    model.load_state_dict(torch.load('models/hybrid_mlp_model.pth'))
    model.eval()

    all_preds = []
    all_labels = []
    with torch.no_grad():
        for features, labels in test_loader:
            features, labels = features.to(device), labels.to(device)
            outputs = model(features)
            preds = torch.sigmoid(outputs) > 0.5
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    ocean_cols = list(OCEAN_TRAITS.keys())

    print("\n--- 📊 Final Performance Report (Improved Hybrid MLP) ---")
    print(classification_report(all_labels, all_preds, target_names=ocean_cols, zero_division=0))
    print("--- ✅ Hybrid training complete! ---")

if __name__ == "__main__":
    run_hybrid_training()