# explain_model.py

import torch
import torch.nn as nn
import joblib
import pandas as pd
from transformers import AutoTokenizer, AutoModel
from captum.attr import IntegratedGradients
import sys
import os

# --- Ensure the script can find your 'src' modules ---
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.features import extract_enhanced_linguistic_features, clean_text

# --- Define the HybridMLP class exactly as it is in personality_service.py ---
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

# --- Configuration ---
DEVICE = torch.device("cpu") # Run on CPU for simplicity
OCEAN_TRAITS = {
    0: 'Openness',
    1: 'Conscientiousness',
    2: 'Extraversion',
    3: 'Agreeableness',  # <-- CORRECTED THIS LINE
    4: 'Neuroticism'
}

# --- 1. Load all necessary models and assets ---
print("--- Loading models and assets for XAI analysis ---")
try:
    # Load Tokenizer and RoBERTa for embeddings
    tokenizer = AutoTokenizer.from_pretrained('roberta-base')
    embedding_model = AutoModel.from_pretrained('roberta-base').to(DEVICE)
    embedding_model.eval()

    # Load Scaler and Feature Columns
    scaler = joblib.load('models/linguistic_feature_scaler.pkl')
    feature_columns = joblib.load('models/feature_columns.pkl')

    # Load your trained Hybrid MLP model
    n_embeddings = 768
    n_linguistic_features = len(feature_columns)
    total_input_size = n_embeddings + n_linguistic_features

    prediction_model = HybridMLP(input_size=total_input_size).to(DEVICE)
    prediction_model.load_state_dict(torch.load('models/hybrid_mlp_model.pth', map_location=DEVICE))
    prediction_model.eval()
    print("✅ All assets loaded successfully.")
except Exception as e:
    print(f"❌ Error loading assets: {e}")
    print("Please ensure all model files are in the 'models/' directory.")
    exit()


def prepare_inputs_for_text(text):
    """Processes a single raw text into the combined tensor the model expects."""
    cleaned_text = clean_text(text)
    
    # 1. Get RoBERTa embeddings
    encoding = tokenizer(
        cleaned_text, 
        truncation=True, 
        max_length=512, 
        return_tensors='pt'
    ).to(DEVICE)
    
    with torch.no_grad():
        transformer_output = embedding_model(**encoding)
        text_embedding = transformer_output.last_hidden_state[:, 0]

    # 2. Get linguistic features
    raw_features = extract_enhanced_linguistic_features(pd.Series([cleaned_text]))
    aligned_features = raw_features.reindex(columns=feature_columns, fill_value=0)
    scaled_features = scaler.transform(aligned_features)
    linguistic_features = torch.tensor(scaled_features, dtype=torch.float).to(DEVICE)
    
    # 3. Combine into a single input tensor
    combined_input = torch.cat((text_embedding, linguistic_features), dim=1)
    return combined_input

# --- 2. Select a sample text to analyze ---
# You can change this text to any example you want to test.
sample_text = """
I am a very organized person. I plan my week every Sunday, making sure all my assignments are scheduled and my goals are clear. 
I enjoy quiet evenings reading a book more than going to loud parties, but I do love spending quality time with a few close friends. 
I try to be helpful and cooperative, but I get frustrated when things are inefficient. I worry a lot about the future, 
and sometimes I can get a little anxious about deadlines, even though I always meet them.
"""

print(f"\n--- Analyzing Sample Text ---\n'{sample_text[:100]}...'")

# --- 3. Perform the XAI Attribution ---
# Prepare the input and a zero-baseline for Integrated Gradients
model_input = prepare_inputs_for_text(sample_text)
baseline = torch.zeros_like(model_input)

# Initialize the Integrated Gradients algorithm with our prediction model
ig = IntegratedGradients(prediction_model)

print("\n--- XAI Analysis Results ---")

# Analyze attribution for EACH of the five traits
for trait_index, trait_name in OCEAN_TRAITS.items():
    # Calculate attributions for the specific trait
    attributions = ig.attribute(
        model_input, 
        baselines=baseline, 
        target=trait_index # This is CRITICAL: 0 for O, 1 for C, etc.
    )
    
    # Separate the attributions for the text embedding vs. the linguistic features
    attr_embedding = attributions[:, :n_embeddings]
    attr_linguistic = attributions[:, n_embeddings:]
    
    print(f"\n--- Trait: {trait_name} ---")
    print(f"  - Overall Text Contribution: {attr_embedding.sum():.4f}")
    print(f"  - Overall Linguistic Feature Contribution: {attr_linguistic.sum():.4f}")
    
    # Find the top 5 most influential linguistic features for this trait
    top_indices = torch.topk(attr_linguistic.abs(), 5).indices.squeeze(0)
    
    print("  - Top 5 Most Influential Linguistic Features:")
    for idx in top_indices:
        feature_name = feature_columns[idx]
        feature_attr_score = attr_linguistic[0, idx].item()
        direction = "(supports)" if feature_attr_score > 0 else "(opposes)"
        print(f"    - {feature_name:<25} | Score: {feature_attr_score:.4f} {direction}")

print("\n--- Analysis Complete ---")