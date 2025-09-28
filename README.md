# Synergistic Personality Detection using Hybrid AI

![Project Banner](https://user-images.githubusercontent.com/29746973/149817726-24e0f49c-8e62-42e1-80a5-e362c8234857.png)

This project is a web-based application that analyzes user-provided text to predict the **Big Five (OCEAN) personality traits**. It uses a sophisticated hybrid model that combines the deep contextual understanding of **RoBERTa embeddings** with rich, hand-crafted **linguistic features**. The user interacts with an AI assistant named Kai through a chat interface, receiving real-time personality insights and sentiment analysis.



---

## ✨ Key Features

* **Hybrid Personality Model:** Fuses RoBERTa sentence embeddings with over 200 linguistic features (readability, sentiment, emotion, etc.) for nuanced and accurate predictions.
* **Interactive Chat Interface:** A user-friendly frontend where users can have a natural conversation with an AI assistant to reveal personality insights.
* **Generative AI Integration:** Uses the Google Gemini API to generate empathetic, context-aware follow-up questions and detailed final summaries.
* **Real-time Sentiment Analysis:** Tracks and visualizes the user's sentiment trend throughout the conversation.
* **Model Explainability (XAI):** Includes a script (`explain_model.py`) using Captum's Integrated Gradients to understand which linguistic features most influence the model's predictions for each trait.
* **Decoupled Architecture:** A robust backend built with **FastAPI** serving the AI model and a separate frontend built with **Flask** and vanilla JavaScript.

---

## 🚀 Tech Stack

* **Backend:** Python, FastAPI, PyTorch, Transformers, scikit-learn, spaCy, NLTK
* **Frontend:** Flask, HTML, CSS, JavaScript (with Chart.js)
* **AI & ML:** Google Gemini, RoBERTa, Captum, Pandas, NumPy

---

## 🛠️ Setup and Installation

Follow these steps to get the project running on your local machine.

### 1. **Clone the Repository**
```bash
git clone [https://github.com/user-aditi/Synergistic-Personality-Detection.git](https://github.com/user-aditi/Synergistic-Personality-Detection.git)
cd Synergistic-Personality-Detection
```

### 2. **Create a Virtual Environment**
It's recommended to use a virtual environment to manage dependencies.
```bash
python -m venv penv
source penv/bin/activate  # On Windows, use `penv\Scripts\activate`
```

### 3. **Install Dependencies**
Install all the required Python packages.
```bash
pip install -r requirements.txt
```

### 4. **Download spaCy Model**
The linguistic feature extractor requires a spaCy model.
```bash
python -m spacy download en_core_web_sm
```

### 5. **Set Up Environment Variables**
The application requires a Google Gemini API key to function.

* Create a file named `.env` in the root directory.
* Add your API key to this file as shown below:

```env
# .env
GOOGLE_API_KEY="YOUR_GOOGLE_API_KEY_HERE"
```

---

## 🏃‍♀️ Running the Project

Because the necessary model files are included in this repository, you do not need to run any training scripts to start the application.

### 1. **Start the Backend API Server**
This will start the FastAPI server that serves the model and handles the analysis logic.
```bash
python run_api.py
```
The API will be available at `http://127.0.0.1:8000`. You can see the documentation at `http://127.0.0.1:8000/docs`.

### 2. **Start the Frontend Server**
In a **new terminal**, start the Flask server for the user interface.
```bash
python frontend/app.py
```
The web application will be accessible at **`http://127.0.0.1:5000`**. Open this URL in your browser to use the app!

---

### (Optional) How to Retrain the Model
If you want to train the model from scratch, you can run the following scripts in order:

1.  `python preprocess_embeddings.py`
2.  `python train_hybrid_mlp.py`

---

## 📁 Project Structure

```
Synergistic-Personality-Detection/
├── .env                  # Stores secret keys (ignored by Git)
├── .gitignore            # Specifies files for Git to ignore
├── config.py             # Central configuration for models, APIs
├── requirements.txt      # Project dependencies
├── README.md             # This file
│
├── frontend/             # All frontend code (HTML, CSS, JS)
│   ├── app.py            # Flask server for the frontend
│   ├── static/
│   └── templates/
│
├── src/                  # All backend source code
│   ├── api.py            # FastAPI endpoints
│   ├── features.py       # Linguistic feature extraction logic
│   ├── llm_service.py    # Handles calls to Google Gemini API
│   ├── model.py          # PyTorch model architecture
│   ├── personality_service.py # Core logic for prediction and session management
│   └── ...
│
├── models/               # Saved models, scalers, etc. (only essentials are tracked)
│
├── preprocess_embeddings.py # Script to generate and save text embeddings
├── train_baseline_mlp.py    # Script to train a simpler baseline model
├── train_hybrid_mlp.py      # Script to train the main hybrid model
└── explain_model.py         # Script to run XAI on the model
```