# src/session_manager.py
import uuid
from datetime import datetime
from typing import Dict, List, Optional

class SessionManager:
    """Manages user conversation sessions with structured history and sentiment tracking."""
    
    def __init__(self):
        self.sessions: Dict[str, Dict] = {}
    
    def get_or_create_session(self, session_id: Optional[str] = None) -> str:
        """Get existing session or create a new one."""
        if session_id is None or session_id not in self.sessions:
            session_id = str(uuid.uuid4())
            self.sessions[session_id] = {
                "created_at": datetime.now(),
                "running_avg_scores": {trait: 0 for trait in ['O', 'C', 'E', 'A', 'N']},
                "input_count": 0,
                "conversation_turns": [],
                "sentiment_history": [] # NEW: To track sentiment over time
            }
        return session_id
    
    def get_session_info(self, session_id: str) -> Optional[Dict]:
        """Get complete information for a specific session."""
        return self.sessions.get(session_id)
        
    def add_user_message(self, session_id: str, text: str, sentiment: str):
        """Adds a user's message and its sentiment to the history."""
        session = self.get_session_info(session_id)
        if session:
            session["conversation_turns"].append({"role": "user", "text": text})
            session["sentiment_history"].append(sentiment)
            session["input_count"] += 1

    def add_bot_message(self, session_id: str, text: str):
        """Adds the bot's response to the conversation history."""
        session = self.get_session_info(session_id)
        if session:
            session["conversation_turns"].append({"role": "bot", "text": text})

    def clear_session(self, session_id: str) -> bool:
        """Deletes a specific session from memory."""
        if session_id in self.sessions:
            del self.sessions[session_id]
            return True
        return False
