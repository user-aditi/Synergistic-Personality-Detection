# preprocess_embeddings.py
import torch
import pandas as pd
import os
import joblib
from sklearn.model_selection import train_test_split
from datasets import load_dataset
from transformers import AutoTokenizer
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm

# Import your existing model and dataset classes
from src.model import HybridPersonalityModel
from config import MODEL_CONFIG, OCEAN_TRAITS

def run_extraction():
    print("--- 🧠 Starting Embedding Extraction ---")

    # Basic setup
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    os.makedirs("models", exist_ok=True)

    # --- Step 1: Load and Split Dataset ---
    print("Step 1: Loading and splitting dataset...")
    dataset = load_dataset("jingjietan/essays-big5")
    df = dataset['train'].to_pandas()
    ocean_cols = ['O', 'C', 'E', 'A', 'N']
    
    X = df['text']
    y = df[ocean_cols]

    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
    
    # --- Step 2: Load Your Pre-trained Model to Access RoBERTa ---
    print("Step 2: Loading your trained model to access the transformer...")
    # We need the number of linguistic features to initialize the model,
    # even though we won't use that part of it.
    scaler = joblib.load('models/linguistic_feature_scaler.pkl')
    n_features = scaler.n_features_in_
    
    model = HybridPersonalityModel(n_linguistic_features=n_features)
    model.load_state_dict(torch.load(MODEL_CONFIG['model_path'], map_location=device))
    model.to(device)
    model.eval() # Set model to evaluation mode

    tokenizer = AutoTokenizer.from_pretrained(MODEL_CONFIG['model_name'])

    # --- Step 3: Define a function to process data and extract embeddings ---
    def extract_embeddings(texts, labels_df):
        all_embeddings = []
        
        # Convert pandas DataFrame to a torch Tensor
        labels_tensor = torch.tensor(labels_df.astype(float).values, dtype=torch.float)
        
        # Create a DataLoader for batching
        dataset = TensorDataset(torch.arange(len(texts))) # Dummy tensor for indexing
        loader = DataLoader(dataset, batch_size=MODEL_CONFIG['batch_size'])

        with torch.no_grad():
            for batch in tqdm(loader, desc="Extracting Embeddings"):
                indices = batch[0]
                batch_texts = texts.iloc[indices].tolist()

                encoding = tokenizer(
                    batch_texts,
                    truncation=True,
                    padding='max_length',
                    max_length=MODEL_CONFIG['max_length'],
                    return_tensors='pt'
                )
                input_ids = encoding['input_ids'].to(device)
                attention_mask = encoding['attention_mask'].to(device)

                # Get the transformer output (we only need the pooled output)
                transformer_output = model.transformer(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                # The [CLS] token embedding is a good representation of the whole sequence
                pooled_output = transformer_output.last_hidden_state[:, 0]
                all_embeddings.append(pooled_output.cpu())

        return torch.cat(all_embeddings, dim=0), labels_tensor

    # --- Step 4: Process each data split and save the results ---
    print("\nProcessing training data...")
    train_embeddings, train_labels = extract_embeddings(X_train, y_train)
    torch.save(train_embeddings, 'models/train_embeddings.pt')
    torch.save(train_labels, 'models/train_labels.pt')
    print(f"✅ Saved training embeddings and labels. Shape: {train_embeddings.shape}")

    print("\nProcessing validation data...")
    val_embeddings, val_labels = extract_embeddings(X_val, y_val)
    torch.save(val_embeddings, 'models/val_embeddings.pt')
    torch.save(val_labels, 'models/val_labels.pt')
    print(f"✅ Saved validation embeddings and labels. Shape: {val_embeddings.shape}")

    print("\nProcessing test data...")
    test_embeddings, test_labels = extract_embeddings(X_test, y_test)
    torch.save(test_embeddings, 'models/test_embeddings.pt')
    torch.save(test_labels, 'models/test_labels.pt')
    print(f"✅ Saved test embeddings and labels. Shape: {test_embeddings.shape}")

    print("\n--- ✅ All embeddings extracted and saved! ---")

if __name__ == "__main__":
    run_extraction()