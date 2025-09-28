import torch
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from datasets import load_dataset
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from torch.optim import AdamW
from torch.utils.data import DataLoader
import os

# Assuming other files are in the src directory
from src.dataset import PersonalityDataset
from src.features import extract_enhanced_linguistic_features
from src.model import HybridPersonalityModel
from src.engine import train_epoch, eval_model
from src.utils import get_class_weights
from config import MODEL_CONFIG, OCEAN_TRAITS

def run_training():
    print("--- 🧠 Starting Model Training Pipeline ---")

    # Basic setup
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # --- Step 1: Loading dataset ---
    print("Step 1: Loading dataset...")
    dataset = load_dataset("jingjietan/essays-big5")
    df = dataset['train'].to_pandas()
    print("Dataset loaded successfully.")

    # Define feature (X) and labels (y)
    ocean_cols = ['O', 'C', 'E', 'A', 'N']
    X = df['text']
    y = df[ocean_cols]

    # Split data
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
    print(f"Data split complete. Training samples: {len(X_train)}, Validation: {len(X_val)}, Test: {len(X_test)}")

    # --- Step 2: Extracting and scaling linguistic features ---
    print("\nStep 2: Extracting and scaling linguistic features...")
    X_train_ling = extract_enhanced_linguistic_features(X_train)
    X_val_ling = extract_enhanced_linguistic_features(X_val)
    X_test_ling = extract_enhanced_linguistic_features(X_test)

    # ✅ ADDED: Get the column order from the training set
    feature_columns = X_train_ling.columns

    # ✅ ADDED: Reindex validation and test sets to match the training set's columns
    # This ensures column consistency and prevents the ValueError.
    X_val_ling = X_val_ling.reindex(columns=feature_columns, fill_value=0)
    X_test_ling = X_test_ling.reindex(columns=feature_columns, fill_value=0)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_ling)
    X_val_scaled = scaler.transform(X_val_ling)
    X_test_scaled = scaler.transform(X_test_ling)
    
    # Save the scaler
    os.makedirs('models', exist_ok=True)
    scaler_path = 'models/linguistic_feature_scaler.pkl'
    joblib.dump(scaler, scaler_path)
    print(f"Linguistic features extracted and scaler saved to '{scaler_path}'")

    columns_path = 'models/feature_columns.pkl'
    joblib.dump(feature_columns.tolist(), columns_path)
    print(f"Linguistic features extracted, scaler saved to '{scaler_path}', and columns saved to '{columns_path}'")

    # --- Step 3: Creating PyTorch DataLoaders ---
    print("\nStep 3: Creating PyTorch DataLoaders...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_CONFIG['model_name'])
    
    train_dataset = PersonalityDataset(X_train, X_train_scaled, y_train, tokenizer, MODEL_CONFIG['max_length'])
    val_dataset = PersonalityDataset(X_val, X_val_scaled, y_val, tokenizer, MODEL_CONFIG['max_length'])
    test_dataset = PersonalityDataset(X_test, X_test_scaled, y_test, tokenizer, MODEL_CONFIG['max_length'])
    
    train_loader = DataLoader(train_dataset, batch_size=MODEL_CONFIG['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=MODEL_CONFIG['batch_size'])
    test_loader = DataLoader(test_dataset, batch_size=MODEL_CONFIG['batch_size'])
    print("DataLoaders created.")

    # --- Step 4: Initializing model and training components ---
    print("\nStep 4: Initializing model and training components...")
    n_features = X_train_scaled.shape[1]
    model = HybridPersonalityModel(n_linguistic_features=n_features).to(device)

    class_weights = get_class_weights(y_train, ocean_cols, device)
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=class_weights)
    print("Using BCEWithLogitsLoss with weights to handle class imbalance.")

    optimizer = AdamW(model.parameters(), lr=MODEL_CONFIG['learning_rate'], weight_decay=0.01)

    total_steps = len(train_loader) * MODEL_CONFIG['epochs']
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=total_steps
    )

    # --- Step 5: Starting model training loop ---
    print("\nStep 5: Starting model training loop...")
    best_val_loss = float('inf')
    early_stop_counter = 0
    patience = 3
    model_path = 'models/personality_classifier_model.pth'

    for epoch in range(MODEL_CONFIG['epochs']):
        print(f"--- Epoch {epoch + 1}/{MODEL_CONFIG['epochs']} ---")
        train_loss = train_epoch(model, train_loader, loss_fn, optimizer, device, scheduler)
        val_loss = eval_model(model, val_loader, loss_fn, device)
        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), model_path)
            print(f"✔️ Validation loss improved. Model saved to '{model_path}'")
            early_stop_counter = 0
        else:
            early_stop_counter += 1
            print(f"❌ Validation loss did not improve. Counter: {early_stop_counter}/{patience}")
            if early_stop_counter >= patience:
                print("--- Early stopping triggered. Halting training. ---")
                break
    
    print("\n--- Training Complete ---")

    # --- Step 6: Evaluating the best model on the test set ---
    print("\nStep 6: Evaluating the best model on the test set...")
    model.load_state_dict(torch.load(model_path))
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            linguistic_features = batch['linguistic_features'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(input_ids, attention_mask, linguistic_features)
            preds = torch.sigmoid(outputs) > 0.5
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    print("\n--- 📊 Final Performance Report ---")
    report = classification_report(all_labels, all_preds, target_names=ocean_cols, output_dict=True, zero_division=0)
    
    print("\n--- Overall (Macro Average) ---")
    print(f"  - F1-Score:  {report['macro avg']['f1-score']:.4f}")
    print(f"  - Precision: {report['macro avg']['precision']:.4f}")
    print(f"  - Recall:    {report['macro avg']['recall']:.4f}")

    for trait in ocean_cols:
        trait_name = OCEAN_TRAITS[trait]['name']
        trait_report = report[trait]
        accuracy = np.mean(all_preds[:, ocean_cols.index(trait)] == all_labels[:, ocean_cols.index(trait)])
        print(f"\n--- Trait: {trait} ({trait_name}) ---")
        print(f"  - F1-Score:  {trait_report['f1-score']:.4f}")
        print(f"  - Precision: {trait_report['precision']:.4f}")
        print(f"  - Recall:    {trait_report['recall']:.4f}")
        print(f"  - Accuracy:  {accuracy:.4f}")
        
    print("\n-------------------------------------------------")
    print("✅ Pipeline finished.")

if __name__ == "__main__":
    run_training()