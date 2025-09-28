import os
import torch
from dotenv import load_dotenv

# Load environment variables from a .env file if it exists
load_dotenv()

# --- API Keys & Environment Settings ---
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "YOUR_API_KEY_HERE")
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "http://127.0.0.1:5500,http://localhost:5500").split(",")

# --- Model & Training Configuration (Single Source of Truth) ---
MODEL_CONFIG = {
    'model_name': 'roberta-base',
    'max_length': 256,
    'batch_size': 16,
    'learning_rate': 2e-5,
    'epochs': 10,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'early_stopping_patience': 3,
    'min_delta': 0.001,
    'model_path': "models/personality_classifier_model.pth",
}

API_CONFIG = {
    'host': '127.0.0.1',
    'port': 8000,
    'log_level': 'info',
}


# --- File & Path Configuration ---
os.makedirs("models", exist_ok=True)
MODEL_PATH = "models/personality_classifier_model.pth"
SCALER_PATH = "models/linguistic_feature_scaler.pkl"

# --- Dataset Configuration ---
DATASET_NAME = "jingjietan/essays-big5"
TEXT_COLUMN = 'text'
LABEL_COLUMNS = ['O', 'C', 'E', 'A', 'N']

# --- Trait Information (for API and reporting) ---
OCEAN_TRAITS = {
    'O': {'name': 'Openness'},
    'C': {'name': 'Conscientiousness'},
    'E': {'name': 'Extraversion'},
    'A': {'name': 'Agreeableness'},
    'N': {'name': 'Neuroticism'}
}