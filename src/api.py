# src/api.py
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Dict
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.personality_service import PersonalityService
from config import MODEL_PATH, SCALER_PATH 

app = FastAPI(title="OCEAN Personality Analyzer API")

# CORS middleware setup
origins = ["http://127.0.0.1:5000", "http://localhost:5000", "http://127.0.0.1:5500", "http://localhost:5500"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic Models
class TextInput(BaseModel):
    text: str
    session_id: Optional[str] = None

class TraitExplanationRequest(BaseModel):
    trait_name: str
    score: int

# Global service instance
personality_service: Optional[PersonalityService] = None

@app.on_event("startup")
async def startup_event():
    global personality_service
    personality_service = PersonalityService()

# API Endpoints
@app.get("/health")
async def health_check():
    # Check for the new, specific model attributes
    model_loaded = personality_service.prediction_model is not None if personality_service else False
    llm_available = personality_service.llm_service is not None if personality_service else False
    return {"status": "healthy", "model_loaded": model_loaded, "llm_available": llm_available}


# src/api.py

@app.post("/analyze")
async def analyze_personality(input_data: TextInput):
    if personality_service is None:
        raise HTTPException(status_code=503, detail="Service not available")
    
    user_text = input_data.text.strip()
    
    # NEW: Better handling for short or empty inputs
    if len(user_text.split()) < 3:
        follow_up = "Hello there! To get a sense of your personality, I'll need a little more to go on. Could you tell me a bit about your day or something that's on your mind?"
        if personality_service.llm_service:
            try:
                # Ask the LLM to craft a more creative prompt
                prompt = "The user has provided a very short response. Ask them a gentle, open-ended question to encourage them to share more about themselves. Keep it under 30 words."
                response = personality_service.llm_service.model.generate_content(prompt)
                follow_up = response.text.strip()
            except Exception:
                # Fallback to the hardcoded message if the LLM call fails
                pass
        
        # Return a standard structure but with the prompt for more text
        session_id = personality_service.session_manager.get_or_create_session(input_data.session_id)
        session_data = personality_service.get_session_data(session_id)
        if session_data is None: # Handle case where session might not be found
            session_data = {"ocean_scores": {}, "trait_descriptions": {}, "sentiment_history": []}

        return {
            "session_id": session_id,
            "ocean_scores": session_data["ocean_scores"],
            "trait_descriptions": session_data["trait_descriptions"],
            "followup_question": follow_up,
            "sentiment_history": session_data["sentiment_history"],
            "conversation_stats": {}
        }

    try:
        result = personality_service.analyze_with_history(input_data.text, input_data.session_id)
        return result
    except Exception as e:
        print(f"ANALYSIS ERROR: {e}")
        raise HTTPException(status_code=500, detail=f"An internal error occurred: {e}")

@app.post("/explain-trait")
async def explain_trait(request: TraitExplanationRequest):
    """New endpoint to generate personalized explanations for a trait score."""
    if personality_service is None or personality_service.llm_service is None:
        raise HTTPException(status_code=503, detail="LLM Service not available")
    
    try:
        explanation = personality_service.llm_service.explain_trait_score(
            request.trait_name,
            request.score
        )
        return {"explanation": explanation}
    except Exception as e:
        print(f"EXPLANATION ERROR: {e}")
        raise HTTPException(status_code=500, detail="Failed to generate explanation.")

# NEW: Endpoint to get data for an existing session
@app.get("/session/{session_id}")
async def get_session(session_id: str):
    """Retrieves all data for a given session to restore state."""
    if personality_service is None:
        raise HTTPException(status_code=503, detail="Service not available")
    session_data = personality_service.get_session_data(session_id)
    if session_data:
        return session_data
    raise HTTPException(status_code=404, detail="Session not found")

# NEW: Endpoint to generate the final summary
@app.get("/summary/{session_id}")
async def get_summary(session_id: str):
    """Generates and returns a final personality summary for the session."""
    if personality_service is None or personality_service.llm_service is None:
        raise HTTPException(status_code=503, detail="LLM Service not available")
    try:
        summary = personality_service.get_final_summary(session_id)
        return {"summary": summary}
    except Exception as e:
        print(f"SUMMARY ERROR: {e}")
        raise HTTPException(status_code=500, detail="Failed to generate summary.")

@app.post("/reset/{session_id}")
async def reset_session(session_id: str):
    if personality_service is None:
        raise HTTPException(status_code=503, detail="Service not available")
    success = personality_service.session_manager.clear_session(session_id)
    if success:
        return {"message": "Session reset successfully"}
    else:
        raise HTTPException(status_code=404, detail="Session not found")