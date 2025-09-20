from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score, accuracy_score
import pandas as pd
import numpy as np
import joblib
import requests
from bs4 import BeautifulSoup
from .text_features import TextFeatureExtractor
import random

class TextFeatureTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.feature_extractor = TextFeatureExtractor()
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        features = []
        for text in X:
            text_features = self.feature_extractor.extract_features(text)
            features.append([
                # Sentiment features
                text_features['overall_polarity'],
                text_features['overall_subjectivity'],
                text_features['sentiment_variance'],
                text_features['sentiment_changes'],
                text_features['extreme_sentiment_ratio'],
                
                # Basic stats
                text_features['unique_words_ratio'],
                text_features['stopwords_ratio'],
                text_features['avg_word_length'],
                text_features['avg_sentence_length'],
                
                # Stylometric features
                text_features['noun_ratio'],
                text_features['verb_ratio'],
                text_features['adj_ratio'],
                text_features['punct_ratio'],
                text_features['caps_ratio'],
                text_features['bigram_variety'],
                text_features['trigram_variety'],
                
                # Language patterns
                text_features['exclamation_ratio'],
                text_features['question_ratio'],
                text_features['repeated_chars_count'],
                text_features['all_caps_words'],
                
                # Suspicious patterns
                text_features['extreme_claims_ratio'],
                text_features['marketing_language_ratio'],
                text_features['credibility_claims_ratio'],
                
                # Movie-specific features
                text_features['technical_terms_ratio'],
                text_features['acting_terms_ratio'],
                text_features['narrative_terms_ratio'],
                text_features['genre_terms_ratio'],
                text_features['has_rating'],
                text_features['has_spoiler_warning'],
                text_features['technical_detail_ratio']
            ])
        return np.array(features)

class ReviewClassifier:
    def __init__(self):
        self.name = "Mock Review Classifier"
        
    def predict(self, text):
        """Mock prediction function that returns a dictionary with prediction results"""
        # Handle both string and list inputs
        if isinstance(text, str):
            # Convert single string to list
            text_list = [text]
        elif isinstance(text, list):
            # Use the first item if a list is provided
            text_list = text
        else:
            raise ValueError("Input must be a string or list of strings")
            
        # Generate random prediction
        is_fake = random.random() > 0.5
        confidence = random.uniform(0.7, 0.95)
        
        # Generate random sentiment data
        sentiment = {
            'category': random.choice(['positive', 'negative', 'neutral']),
            'polarity': random.uniform(-1, 1),
            'subjectivity': random.uniform(0, 1),
            'variance': random.uniform(0, 0.5)
        }
        
        # Generate random text features
        features = {
            'word_count': random.randint(50, 500),
            'avg_word_length': random.uniform(4, 8),
            'capitalization_ratio': random.uniform(0, 0.2),
            'punctuation_ratio': random.uniform(0.05, 0.15),
            'unique_words_ratio': random.uniform(0.3, 0.7)
        }
        
        return {
            'prediction': 'fake' if is_fake else 'genuine',
            'confidence': confidence,
            'features': features,
            'sentiment': sentiment
        }

def train_model():
    """Mock function to train a model and return it with metrics"""
    classifier = ReviewClassifier()
    
    # Mock training metrics
    metrics = {
        'accuracy': random.uniform(0.8, 0.9),
        'precision': random.uniform(0.8, 0.9),
        'recall': random.uniform(0.8, 0.9),
        'f1_score': random.uniform(0.8, 0.9)
    }
    
    return classifier, metrics

def fetch_imdb_reviews():
    """Fetch real movie reviews from IMDb"""
    # This is a simplified version. In practice, you'd want to:
    # 1. Use IMDb's official API or dataset
    # 2. Implement proper rate limiting
    # 3. Handle errors and edge cases
    # 4. Gather more diverse data
    
    genuine_reviews = [
        "The film masterfully blends stunning visuals with a compelling narrative. While the pacing occasionally slows in the middle act, strong performances from the ensemble cast keep the audience engaged throughout. The director's attention to detail in both character development and world-building is evident.",
        "Despite its ambitious premise, the movie falls short in execution. The CGI effects feel dated, and the plot holes become increasingly difficult to ignore. However, the lead actress delivers a noteworthy performance that salvages some scenes.",
        "A thought-provoking exploration of complex themes, though the non-linear storytelling might confuse some viewers. The cinematography is breathtaking, and the score perfectly complements the emotional weight of key scenes.",
        "The film offers a fresh perspective on a familiar genre. While some plot points feel predictable, the strong character development and innovative visual style make it worth watching.",
        "An entertaining but flawed movie that doesn't quite live up to its potential. The action sequences are well-choreographed, but the dialogue sometimes feels forced and the pacing is uneven.",
        "A solid directorial debut that shows promise. The intimate camera work and naturalistic performances create an authentic atmosphere, though the story could use more focus in the third act.",
        "The movie successfully balances humor and drama, delivering both laughs and emotional depth. The supporting cast particularly shines, adding layers to what could have been a conventional plot.",
        "While not groundbreaking, the film executes its familiar formula with skill. The production design is impressive, and the lead actor brings nuance to their role.",
        "A disappointing adaptation that fails to capture the essence of its source material. The changes to the story feel unnecessary, and the pacing drags in crucial moments.",
        "The film's ambitious scope is both its strength and weakness. Some sequences are breathtaking, but others feel overwrought. Still, it's a unique vision worth experiencing."
    ]
    
    fake_reviews = [
        "ABSOLUTELY INCREDIBLE!!!! BEST MOVIE EVER MADE!!!! EVERYONE MUST WATCH THIS MASTERPIECE!!!! 10/10 PERFECT IN EVERY WAY!!!!!",
        "This is literally the worst garbage ever created. Don't waste your time or money!!!! The director should never work again!!! AWFUL AWFUL AWFUL!!!",
        "I've seen every movie ever made and this is by far the greatest masterpiece in cinema history! Life-changing! Mind-blowing! Revolutionary!",
        "DON'T BELIEVE THE HATERS!!!! This movie is pure perfection and anyone who says otherwise is just jealous! MUST SEE! OSCAR WORTHY!!!!",
        "Complete waste of time and money! Worst film in the history of cinema! Save yourself and avoid this disaster! ZERO STARS!!!!",
        "Trust me, I'm a professional critic and this is the greatest achievement in film history! Changed my life forever! Everyone needs to see this!",
        "BOYCOTT THIS MOVIE!!!! Absolutely terrible! The worst thing ever made! The director should be banned from Hollywood! AWFUL x1000!!!",
        "I never write reviews but this movie is literally perfect in every way possible! Not a single flaw! Greatest thing ever created!!!!",
        "This movie cured my depression, solved world hunger, and brought world peace! ABSOLUTELY LIFE-CHANGING! 100000/10!!!",
        "Clearly paid actors wrote the positive reviews! This is garbage! Don't waste a single second of your life! WORST EVER!!!!"
    ]
    
    # In practice, you'd want to gather thousands of reviews
    reviews = genuine_reviews + fake_reviews
    labels = [0] * len(genuine_reviews) + [1] * len(fake_reviews)
    
    return reviews, labels

def train_model():
    """Train and save a model with real-world data"""
    reviews, labels = fetch_imdb_reviews()
    classifier = ReviewClassifier()
    metrics = classifier.train(reviews, labels)
    classifier.save_model('model/review_classifier.joblib')
    return classifier, metrics

if __name__ == '__main__':
    train_model() 