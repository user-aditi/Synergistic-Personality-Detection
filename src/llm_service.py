# src/llm_service.py
import google.generativeai as genai
from config import GOOGLE_API_KEY, OCEAN_TRAITS
from typing import List, Dict

class LLMService:
    """Handles communication with the Google Gemini API to generate conversational responses."""
    def __init__(self):
        if not GOOGLE_API_KEY or GOOGLE_API_KEY == "YOUR_API_KEY_HERE":
            raise ValueError("Google API Key not found or not configured in config.py")
        
        genai.configure(api_key=GOOGLE_API_KEY)
        self.model = genai.GenerativeModel('gemini-1.5-flash')

    # NEW: Function to analyze sentiment
    def analyze_sentiment(self, text: str) -> str:
        """Analyzes sentiment of a given text."""
        try:
            prompt = f"""
            Analyze the sentiment of the following text. 
            Respond with only a single word: POSITIVE, NEGATIVE, or NEUTRAL.

            Text: "{text}"
            """
            response = self.model.generate_content(prompt)
            # Clean up the response to ensure it's one of the three words
            result = response.text.strip().upper()
            if result in ["POSITIVE", "NEGATIVE", "NEUTRAL"]:
                return result
            return "NEUTRAL" # Default fallback
        except Exception as e:
            print(f"LLM sentiment analysis failed: {e}")
            return "NEUTRAL"

    # REVISED: Enhanced prompt for better conversational flow
    def generate_followup(self, conversation_turns: List[Dict], ocean_scores: dict) -> str:
        """
        Generates a contextual acknowledgement and a probing follow-up question
        using the LLM, based on the conversation history.
        """
        try:
            history_str = "\n".join(
                [f"{turn['role'].capitalize()}: {turn['text']}" for turn in conversation_turns[-6:]]
            )

            focus_trait_key = min(ocean_scores, key=ocean_scores.get)
            focus_trait_name = OCEAN_TRAITS[focus_trait_key]['name']

            prompt = f"""
            You are a warm, insightful, and empathetic personality coach named Kai. Your goal is to facilitate a natural, flowing conversation that helps a user explore their own personality.

            CONVERSATION HISTORY (Most Recent Turns):
            ---
            {history_str}
            ---

            ANALYSIS: The user's current personality profile shows the lowest score in '{focus_trait_name}'. This is an area for gentle exploration.

            YOUR TASK:
            Based on the **user's most recent message** and the overall conversation flow, generate a response that does the following:
            1.  Write a brief, thoughtful acknowledgement (1-2 sentences) that proves you are listening and connects directly to what the user just said.
            2.  Craft a creative, open-ended follow-up question. This question should feel like a natural next step in the conversation, but subtly guide them to reflect on something related to '{focus_trait_name}'.
            3.  Ensure your entire response is a single, natural paragraph under 50 words.

            RULES:
            - **DO NOT** be repetitive. If the previous topic was about 'work', ask about something else like 'hobbies', 'relationships', or 'challenges'.
            - **NEVER** sound like a generic chatbot. Avoid phrases like "As an AI...", "That's interesting," or "I'm here to help."
            - **BE PERSONALIZED.** Use "you" and "I" to create a connection.

            Example of a GOOD response (if user talked about a solo trip and their lowest trait is 'Extraversion'):
            "That sense of freedom you described on your solo trip sounds incredible. It makes me curious, in your day-to-day life, what kind of social situations leave you feeling the most drained or the most energized?"

            Example of a BAD response:
            "Thanks for sharing. Tell me more about your experiences with {focus_trait_name}."

            Generate your response now.
            """

            response = self.model.generate_content(prompt)
            return response.text.strip()
        except Exception as e:
            print(f"LLM generation failed: {e}")
            return "That's really insightful. What's another experience that has shaped who you are today?"

    # NEW: Function to explain a trait score, as required by api.py
    def explain_trait_score(self, trait_name: str, score: int) -> str:
        """Generates a personalized explanation for a given trait score."""
        try:
            level = "low" if score < 40 else "high" if score > 60 else "moderate"
            prompt = f"""
            You are an expert personality psychologist.
            A user has a score of {score} (out of 100) for the personality trait '{trait_name}', which is a {level} level.
            In 2-3 encouraging and insightful sentences, explain what a {level} score in '{trait_name}' might mean for them in their daily life, focusing on common behaviors and perspectives associated with this level.
            Do not use jargon. Speak directly to the user ("Your score suggests...").
            """
            response = self.model.generate_content(prompt)
            return response.text.strip()
        except Exception as e:
            print(f"LLM explanation failed: {e}")
            return "Could not generate an explanation at this time."

    # NEW: Function to generate the final summary report
    def generate_final_summary(self, conversation_turns: List[Dict], ocean_scores: dict) -> str:
        """Generates a final, holistic summary of the user's personality."""
        try:
            history_str = "\n".join(
                [f"{turn['role'].capitalize()}: {turn['text']}" for turn in conversation_turns]
            )
            scores_str = "\n".join([f"- {OCEAN_TRAITS[t]['name']}: {s}%" for t, s in ocean_scores.items()])

            prompt = f"""
            You are a personality psychologist writing a final summary report for a client based on an insightful conversation and their final OCEAN personality scores.
            Your task is to synthesize all the information into a thoughtful, multi-paragraph summary.

            CLIENT'S FINAL SCORES:
            ---
            {scores_str}
            ---

            FULL CONVERSATION HISTORY:
            ---
            {history_str}
            ---

            INSTRUCTIONS:
            Write a comprehensive, encouraging, and insightful personality summary. Structure it as follows:
            1.  **Opening:** Start with a warm opening that acknowledges the conversation you've had.
            2.  **Key Traits:** Discuss two or three of their most prominent traits (the highest and lowest scores). Use examples or themes from the conversation to illustrate how these traits might manifest in their life. This is crucial – connect the scores to the chat content.
            3.  **Potential & Growth:** Offer a final, encouraging paragraph that suggests how they can leverage their strengths and be mindful of their areas for growth.
            4.  **Tone:** Maintain a positive, professional, and empathetic tone throughout. Address the user directly as "you."

            Generate the final summary now.
            """
            response = self.model.generate_content(prompt)
            return response.text.strip().replace('\n', '<br>')
        except Exception as e:
            print(f"LLM summary generation failed: {e}")
            return "An error occurred while generating your final summary. Please try again later."