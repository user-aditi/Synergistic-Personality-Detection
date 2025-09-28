# src/personality_service.py
import torch
import torch.nn as nn
import os
import sys
import joblib
import pandas as pd
from typing import Dict, Optional
from transformers import AutoTokenizer, AutoModel

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.features import extract_enhanced_linguistic_features, clean_text
from src.session_manager import SessionManager
from src.llm_service import LLMService
from config import MODEL_CONFIG, OCEAN_TRAITS

# --- NEW: Define the Hybrid MLP class directly in this file ---
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

class PersonalityService:
    def __init__(self, device: str = 'cpu'):
        self.device = device
        self.tokenizer = None
        self.scaler = None
        self.feature_columns = None
        self.embedding_model = None # For RoBERTa
        self.prediction_model = None # For our Hybrid MLP
        
        self.session_manager = SessionManager()
        try:
            self.llm_service = LLMService()
            print("✅ LLM Service initialized successfully.")
        except ValueError as e:
            print(f"⚠️ WARNING: LLM Service failed to initialize: {e}")
            self.llm_service = None

        self.load_models()

    def get_session_data(self, session_id: str) -> Optional[Dict]:
        """Safely retrieves session data for state restoration."""
        session = self.session_manager.get_session_info(session_id)
        if not session:
            return None
        return {
            "ocean_scores": session["running_avg_scores"],
            "trait_descriptions": self._get_trait_descriptions(session["running_avg_scores"]),
            "sentiment_history": session["sentiment_history"],
        }

    def load_models(self):
        try:
            print("--- Loading all necessary models for the application ---")
            
            # 1. Load tokenizer and RoBERTa for embedding extraction
            self.tokenizer = AutoTokenizer.from_pretrained(MODEL_CONFIG['model_name'])
            self.embedding_model = AutoModel.from_pretrained(MODEL_CONFIG['model_name']).to(self.device)
            self.embedding_model.eval()
            print("✅ RoBERTa embedding model loaded.")

            # 2. Load scaler and feature columns for linguistic features
            self.scaler = joblib.load('models/linguistic_feature_scaler.pkl')
            self.feature_columns = joblib.load('models/feature_columns.pkl')
            print("✅ Scaler and feature columns loaded.")
            
            # 3. Load our best-performing Hybrid MLP model
            n_embeddings = 768 # RoBERTa base model output size
            n_linguistic_features = len(self.feature_columns)
            total_input_size = n_embeddings + n_linguistic_features
            
            self.prediction_model = HybridMLP(input_size=total_input_size).to(self.device)
            self.prediction_model.load_state_dict(torch.load('models/hybrid_mlp_model.pth', map_location=self.device))
            self.prediction_model.eval()
            print("✅ Best-performing Hybrid MLP model loaded.")
            print("--- ✅ All models loaded successfully! ---")

        except Exception as e:
            print(f"❌ Error loading models: {str(e)}")
            # Set all to None to prevent the app from running in a broken state
            self.tokenizer = self.embedding_model = self.scaler = self.feature_columns = self.prediction_model = None

    def predict_personality(self, text: str) -> Dict[str, int]:
        if not all([self.tokenizer, self.embedding_model, self.scaler, self.feature_columns, self.prediction_model]):
            print("⚠️ A required model is not loaded. Using demo scores.")
            return {'O': 55, 'C': 65, 'E': 45, 'A': 75, 'N': 35}

        cleaned_text = clean_text(text)

        # --- Step 1: Extract Text Embedding ---
        encoding = self.tokenizer(cleaned_text, truncation=True, padding='max_length', max_length=MODEL_CONFIG['max_length'], return_tensors='pt')
        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)
        
        with torch.no_grad():
            transformer_output = self.embedding_model(input_ids=input_ids, attention_mask=attention_mask)
            text_embedding = transformer_output.last_hidden_state[:, 0]

        # --- Step 2: Extract Linguistic Features ---
        raw_features_df = extract_enhanced_linguistic_features(pd.Series([cleaned_text]))
        aligned_features_df = raw_features_df.reindex(columns=self.feature_columns, fill_value=0)
        scaled_features = self.scaler.transform(aligned_features_df)
        linguistic_features = torch.tensor(scaled_features, dtype=torch.float).to(self.device)
        
        # --- Step 3: Combine Features and Predict ---
        combined_features = torch.cat((text_embedding, linguistic_features), dim=1)
        
        with torch.no_grad():
            predictions_tensor = self.prediction_model(combined_features)
        
        scores = torch.sigmoid(predictions_tensor).cpu().numpy().flatten()
        results = {trait: int(round(score * 100)) for trait, score in zip(OCEAN_TRAITS.keys(), scores)}
        return results

    # --- NO CHANGES NEEDED BELOW THIS LINE ---
    # The analyze_with_history, get_final_summary, and _get_trait_descriptions methods remain the same.

    def analyze_with_history(self, text: str, session_id: Optional[str] = None) -> Dict:
        session_id = self.session_manager.get_or_create_session(session_id)
        sentiment = "NEUTRAL"
        if self.llm_service:
            sentiment = self.llm_service.analyze_sentiment(text)
        self.session_manager.add_user_message(session_id, text, sentiment)
        session = self.session_manager.get_session_info(session_id)
        current_scores = self.predict_personality(text)
        prev_avg = session["running_avg_scores"]
        count = session["input_count"]
        final_scores = {}
        for trait in current_scores:
            avg_score = ((prev_avg.get(trait, 50) * (count - 1)) + current_scores[trait]) / count
            final_scores[trait] = int(round(avg_score))
        session["running_avg_scores"] = final_scores
        final_followup = "Thank you. What else is on your mind?"
        if self.llm_service:
            final_followup = self.llm_service.generate_followup(session["conversation_turns"], final_scores)
        self.session_manager.add_bot_message(session_id, final_followup)
        return {
            "session_id": session_id,
            "ocean_scores": final_scores,
            "trait_descriptions": self._get_trait_descriptions(final_scores),
            "followup_question": final_followup,
            "sentiment_history": session["sentiment_history"],
            "conversation_stats": {"input_count": count}
        }

    def get_final_summary(self, session_id: str) -> str:
        session = self.session_manager.get_session_info(session_id)
        if not session:
            return "Session not found. Unable to generate summary."
        if self.llm_service:
            return self.llm_service.generate_final_summary(session["conversation_turns"], session["running_avg_scores"])
        return "Summary service is currently unavailable."

    def _get_trait_descriptions(self, scores: Dict[str, int]) -> Dict[str, Dict]:
        descriptions = {}
        for trait, score in scores.items():
            trait_info = OCEAN_TRAITS.get(trait, {})
            level = "Low" if score < 40 else "High" if score > 60 else "Moderate"
            descriptions[trait] = {
                "name": trait_info.get('name', "Unknown"),
                "score": score,
                "level": level,
                "interpretation": f"Your profile indicates a {level.lower()} level of {trait_info.get('name', 'this trait')}."
            }
        return descriptions