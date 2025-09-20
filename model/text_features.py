import numpy as np
from textblob import TextBlob
import re
from collections import Counter
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.util import ngrams
import nltk
from typing import Dict, Any

# Download additional NLTK data
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')

class TextFeatureExtractor:
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        
        # Movie-specific patterns
        self.movie_terms = {
            'technical': {'cinematography', 'direction', 'screenplay', 'script', 'editing', 'score', 
                        'soundtrack', 'visual effects', 'cgi', 'practical effects', 'production', 
                        'costume design', 'set design', 'lighting', 'camera work'},
            'acting': {'performance', 'actor', 'actress', 'cast', 'ensemble', 'portrayal', 'role', 
                      'character development', 'chemistry', 'dialogue', 'delivery'},
            'narrative': {'plot', 'story', 'pacing', 'narrative', 'character arc', 'development', 
                         'twist', 'ending', 'climax', 'resolution', 'subplot', 'theme'},
            'genre': {'action', 'drama', 'comedy', 'thriller', 'horror', 'sci-fi', 'romance', 
                     'adventure', 'fantasy', 'mystery', 'documentary'}
        }
        
        # Patterns indicating potential fake reviews
        self.suspicious_patterns = {
            'extreme_claims': [
                r'best (?:movie|film) (?:ever|of all time)',
                r'worst (?:movie|film) (?:ever|of all time)',
                r'masterpiece',
                r'complete garbage',
                r'waste of (?:time|money)',
                r'changed my life',
                r'perfect in every way'
            ],
            'marketing_language': [
                r'must(?:-| )see',
                r'don\'t miss',
                r'run to see',
                r'instant classic',
                r'oscar-worthy'
            ],
            'credibility_claims': [
                r'trust me',
                r'believe me',
                r'take my word',
                r'i\'ve seen (?:everything|all)',
                r'i never write reviews'
            ]
        }
    
    def extract_features(self, text: str) -> Dict[str, Any]:
        """Extract comprehensive features from text"""
        features = {}
        
        # Basic text statistics
        features.update(self._get_basic_stats(text))
        
        # Sentiment features
        features.update(self._get_sentiment_features(text))
        
        # Stylometric features
        features.update(self._get_stylometric_features(text))
        
        # Language patterns
        features.update(self._get_language_patterns(text))
        
        # Movie-specific features
        features.update(self._get_movie_specific_features(text))
        
        return features
    
    def _get_basic_stats(self, text: str) -> Dict[str, float]:
        """Extract basic statistical features"""
        sentences = sent_tokenize(text)
        words = word_tokenize(text.lower())
        words_no_stop = [w for w in words if w not in self.stop_words]
        
        return {
            'num_sentences': len(sentences),
            'num_words': len(words),
            'num_chars': len(text),
            'avg_word_length': np.mean([len(w) for w in words]) if words else 0,
            'avg_sentence_length': np.mean([len(word_tokenize(s)) for s in sentences]) if sentences else 0,
            'unique_words_ratio': len(set(words)) / len(words) if words else 0,
            'stopwords_ratio': len([w for w in words if w in self.stop_words]) / len(words) if words else 0
        }
    
    def _get_sentiment_features(self, text: str) -> Dict[str, float]:
        """Extract advanced sentiment features using TextBlob"""
        blob = TextBlob(text)
        sentences = blob.sentences
        
        # Get sentence-level sentiments
        sentence_sentiments = [sentence.sentiment for sentence in sentences]
        polarities = [s.polarity for s in sentence_sentiments]
        subjectivities = [s.subjectivity for s in sentence_sentiments]
        
        # Calculate sentiment consistency
        polarity_changes = sum(1 for i in range(1, len(polarities)) 
                             if (polarities[i] > 0 and polarities[i-1] < 0) or 
                                (polarities[i] < 0 and polarities[i-1] > 0))
        
        return {
            'overall_polarity': blob.sentiment.polarity,
            'overall_subjectivity': blob.sentiment.subjectivity,
            'sentiment_variance': np.var(polarities) if polarities else 0,
            'max_polarity_diff': max(polarities) - min(polarities) if polarities else 0,
            'subjectivity_variance': np.var(subjectivities) if subjectivities else 0,
            'sentiment_changes': polarity_changes,
            'extreme_sentiment_ratio': sum(1 for p in polarities if abs(p) > 0.8) / len(polarities) if polarities else 0
        }
    
    def _get_stylometric_features(self, text: str) -> Dict[str, float]:
        """Extract stylometric features"""
        words = word_tokenize(text.lower())
        pos_tags = nltk.pos_tag(words)
        
        # Count POS tags
        pos_counts = Counter(tag for word, tag in pos_tags)
        
        # Calculate punctuation ratios
        punct_ratio = len([c for c in text if c in '.,!?;:']) / len(text) if text else 0
        
        # Calculate capitalization features
        cap_ratio = sum(1 for c in text if c.isupper()) / len(text) if text else 0
        
        # Calculate word variety
        if words:
            bigrams = list(ngrams(words, 2))
            trigrams = list(ngrams(words, 3))
            bigram_ratio = len(set(bigrams)) / len(bigrams) if bigrams else 0
            trigram_ratio = len(set(trigrams)) / len(trigrams) if trigrams else 0
        else:
            bigram_ratio = trigram_ratio = 0
        
        return {
            'noun_ratio': (pos_counts.get('NN', 0) + pos_counts.get('NNS', 0)) / len(words) if words else 0,
            'verb_ratio': sum(pos_counts.get(tag, 0) for tag in ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']) / len(words) if words else 0,
            'adj_ratio': (pos_counts.get('JJ', 0) + pos_counts.get('JJR', 0) + pos_counts.get('JJS', 0)) / len(words) if words else 0,
            'punct_ratio': punct_ratio,
            'caps_ratio': cap_ratio,
            'bigram_variety': bigram_ratio,
            'trigram_variety': trigram_ratio
        }
    
    def _get_language_patterns(self, text: str) -> Dict[str, float]:
        """Extract features related to language patterns"""
        text_lower = text.lower()
        
        # Look for suspicious patterns
        pattern_counts = {
            category: sum(len(re.findall(pattern, text_lower)) 
                        for pattern in patterns)
            for category, patterns in self.suspicious_patterns.items()
        }
        
        # Calculate ratios
        total_words = len(word_tokenize(text))
        pattern_ratios = {
            f'{category}_ratio': count / total_words if total_words else 0
            for category, count in pattern_counts.items()
        }
        
        # Additional patterns
        return {
            **pattern_ratios,
            'exclamation_ratio': text.count('!') / len(text) if text else 0,
            'question_ratio': text.count('?') / len(text) if text else 0,
            'ellipsis_ratio': text.count('...') / len(text) if text else 0,
            'repeated_chars_count': len(re.findall(r'(.)\1{2,}', text)),
            'all_caps_words': len(re.findall(r'\b[A-Z]{2,}\b', text)) / total_words if total_words else 0
        }
    
    def _get_movie_specific_features(self, text: str) -> Dict[str, float]:
        """Extract movie-specific features"""
        words = set(word_tokenize(text.lower()))
        
        # Calculate coverage of movie-specific terms
        term_coverage = {
            f'{category}_terms_ratio': len(words.intersection(terms)) / len(words) if words else 0
            for category, terms in self.movie_terms.items()
        }
        
        # Check for specific movie elements discussion
        has_rating = bool(re.search(r'\b(?:(?:\d+(?:\.\d+)?)|(?:one|two|three|four|five|six|seven|eight|nine|ten))\s*(?:\/|\s*out\s*of\s*)\s*(?:10|5)\b', text.lower()))
        has_spoiler_warning = bool(re.search(r'spoiler(?:s|\s+warning|\s+alert)?', text.lower()))
        
        return {
            **term_coverage,
            'has_rating': float(has_rating),
            'has_spoiler_warning': float(has_spoiler_warning),
            'technical_detail_ratio': sum(term_coverage[f'{cat}_terms_ratio'] 
                                       for cat in ['technical', 'acting', 'narrative'])
        } 