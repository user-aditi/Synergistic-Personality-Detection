# src/features.py

import pandas as pd
import numpy as np
import textstat
import spacy
from tqdm.auto import tqdm
from empath import Empath
from nrclex import NRCLex
from textblob import TextBlob
import re # ✅ ADDED: Required for the new clean_text function

# Initialize lexicons once to be reused globally
try:
    nlp = spacy.load('en_core_web_sm')
    lexicon = Empath()
    print("✅ spaCy and Empath models loaded successfully.")
except OSError:
    print("⚠️ spaCy model 'en_core_web_sm' not found. Please run: python -m spacy download en_core_web_sm")
    nlp = None

# ✅ ADDED: The missing clean_text function
def clean_text(text: str) -> str:
    """
    Cleans input text by lowercasing, removing special characters and extra whitespace.
    """
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'\[.*?\]', '', text)  # Remove text in brackets
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Remove non-alphabetic characters except spaces
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra whitespace
    return text

def extract_enhanced_linguistic_features(text_series: pd.Series) -> pd.DataFrame:
    """
    Extracts a rich set of linguistic, psycholinguistic, and emotional features from text.
    """
    if nlp is None:
        raise RuntimeError("spaCy model is not loaded. Cannot extract features.")

    # ✅ ADDED: Define all possible NRC emotions to ensure column consistency.
    nrc_emotions = [
        'fear', 'anger', 'anticipation', 'trust', 'surprise', 
        'positive', 'negative', 'sadness', 'disgust', 'joy'
    ]
    
    features_list = []
    
    if isinstance(text_series, str):
        text_series = [text_series]

    for text in tqdm(text_series, desc="Extracting Features"):
        doc = nlp(text)
        blob = TextBlob(text)

        word_count = len([token for token in doc if not token.is_punct])
        avg_word_len = np.mean([len(token.text) for token in doc if not token.is_punct]) if word_count > 0 else 0
        readability = textstat.flesch_reading_ease(text)

        sentiment_polarity = blob.sentiment.polarity
        sentiment_subjectivity = blob.sentiment.subjectivity

        empath_features = lexicon.analyze(text, normalize=True) or {cat: 0 for cat in lexicon.cats}

        # ✅ CHANGED: This logic now guarantees all emotion columns are present.
        nrc_object = NRCLex(text)
        # 1. Create a base dictionary with all emotions set to 0.
        emotion_features = {emotion: 0 for emotion in nrc_emotions}
        # 2. Update it with the scores found in the current text.
        emotion_features.update(nrc_object.raw_emotion_scores)


        features = {
            'word_count': word_count,
            'avg_word_len': avg_word_len,
            'readability': readability,
            'sentiment_polarity': sentiment_polarity,
            'sentiment_subjectivity': sentiment_subjectivity,
        }
        features.update({f"empath_{cat}": val for cat, val in empath_features.items()})
        # 3. Use the complete, zero-filled dictionary for the final features.
        features.update({f"nrc_{emotion}": val for emotion, val in emotion_features.items()})
        
        features_list.append(features)

    return pd.DataFrame(features_list).fillna(0)