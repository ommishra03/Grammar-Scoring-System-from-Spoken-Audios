# 🎙️ Grammar Scoring System from Spoken Audios

A smart, AI-powered Streamlit web app that transcribes audio files and evaluates spoken English for grammar, fluency, vocabulary, and error types using NLP and grammar-checking APIs. This tool is especially helpful for learners, educators, and public speakers to evaluate and improve their spoken language quality.

---

## 📌 Features

- 🔊 **Audio Transcription** using `faster-whisper` (efficient Whisper model)
- 🧠 **Grammar Evaluation** via LanguageTool API and spaCy NLP
- 📊 **Fluency & Vocabulary Scoring**
- 📉 **Detailed Error Type Breakdown** (Grammar, Spelling, Style, Redundancy, etc.)
- 📈 **Visual Charts** for better insights

---

## 🚀 Technologies Used

| Category       | Tools & Libraries                         |
|----------------|-------------------------------------------|
| Frontend       | Streamlit, Matplotlib, Seaborn            |
| Audio Model    | [faster-whisper](https://github.com/guillaumekln/faster-whisper) |
| NLP            | spaCy (`en_core_web_sm`), LanguageTool API |
| Backend Logic  | Python, pandas, numpy, tempfile, requests |
| Visualization  | Matplotlib, Seaborn                       |

---

## 🔧 Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/your-username/grammar-audio-scoring.git
   cd grammar-audio-scoring
   
2. **Install dependencies**
    
        pip install -r requirements.txt
        
        Download spaCy English model
        
        python -m spacy download en_core_web_sm
        
        Run the Streamlit app
        
        streamlit run app.py

🔑 Replace the "paste api" in the evaluate_grammar() function with your actual LanguageTool API endpoint.

📥 Input

    Accepts .mp3 or .wav files.
    
    Transcribes audio to text.
    
    Analyzes grammatical structure, fluency, and vocabulary.

📤 Output

    📜 Transcribed Text
    
    📊 Individual and Total Scores
    
    ❌ Error Type Distribution
    
    🎯 Final Grammar Score (out of 10)

📚 About the Author

    👨‍🎓 Om Mishra
    
    📍 Third-year Student, Chandigarh University
    
    🏆 Reliance Foundation Scholar
    
    🧠 Mentor at Reliance Foundation, guiding peers in C, C++, DSA, and Python
    
    🌐 Blockchain & AI Enthusiast: Skilled in Solidity, Ethereum, React, and Machine Learning
    
    🛠️ Hackathon Finalist (NASA Space App Challenge, NITs, BITS)
    
    📢 Public Speaker and Presentation Expert: Reconstructed complete Math curriculum for an ed-tech platform
    
    🧑‍🏫 Teaching Experience: Trained merchant navy aspirants in math

🔗 [Connect on LinkedIn](https://www.linkedin.com/in/om-mishra-a62991289)

✨ Acknowledgements

    🤖 Whisper by OpenAI
    
    🧪 spaCy for Natural Language Processing
    
    🔍 LanguageTool API for grammar checking
    

📜 License
This project is open-source and available under the MIT License.

📧 Contact
For feedback or collaboration: ommishra1729@gmail.com

Let me know if you'd like me to include a `requirements.txt` or modify the README for deployment (e.g., Streamlit Cloud, Docker, Hugging Face, etc.).
