# AI-Powered Fake Movie Review Detector

This project implements an AI-powered system to detect fake movie reviews using Natural Language Processing (NLP) and Machine Learning techniques. The system analyzes text patterns, sentiment, and user behavior to classify reviews as genuine or fake.

## Features

- Fake Review Detection using ML/NLP
- Real-time Review Analysis
- Interactive Dashboard
- Sentiment Analysis
- User-friendly Web Interface

## Tech Stack

- Backend: Python, Flask
- Frontend: HTML, CSS, JavaScript
- Machine Learning: scikit-learn, NLTK
- Data Processing: pandas, numpy

## Setup Instructions

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Initialize NLTK data:
```python
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet')"
```

4. Run the application:
```bash
python app.py
```

5. Access the application at `http://localhost:5000`

## Project Structure

- `app.py`: Main Flask application
- `model/`: ML model and preprocessing scripts
- `static/`: CSS, JavaScript, and other static files
- `templates/`: HTML templates
- `data/`: Training data and model files 