import pandas as pd
import numpy as np
import gzip
import json
import requests
import time
from pathlib import Path
import logging
from typing import List, Tuple, Dict
import os

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class IMDbDataCollector:
    """Handles collection and processing of IMDb dataset reviews"""
    
    def __init__(self, data_dir: str = 'data'):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        
        # IMDb dataset URLs
        self.urls = {
            'reviews': 'https://datasets.imdbws.com/title.ratings.tsv.gz',
            'metadata': 'https://datasets.imdbws.com/title.basics.tsv.gz'
        }
        
        # Rate limiting parameters
        self.request_delay = 1.0  # seconds between requests
        self.last_request_time = 0
        
    def _rate_limited_request(self, url: str) -> requests.Response:
        """Make a rate-limited request"""
        current_time = time.time()
        time_since_last_request = current_time - self.last_request_time
        
        if time_since_last_request < self.request_delay:
            time.sleep(self.request_delay - time_since_last_request)
        
        try:
            response = requests.get(url, stream=True, timeout=30)
            response.raise_for_status()
            self.last_request_time = time.time()
            return response
        except requests.exceptions.RequestException as e:
            logger.error(f"Error downloading from {url}: {str(e)}")
            raise
    
    def download_dataset(self, dataset_type: str) -> Path:
        """Download and extract IMDb dataset"""
        if dataset_type not in self.urls:
            raise ValueError(f"Unknown dataset type: {dataset_type}")
        
        url = self.urls[dataset_type]
        output_file = self.data_dir / f"{dataset_type}.tsv"
        
        if output_file.exists():
            logger.info(f"Dataset {dataset_type} already exists")
            return output_file
        
        try:
            logger.info(f"Downloading {dataset_type} dataset...")
            response = self._rate_limited_request(url)
            
            # Save and extract the gzipped file
            gz_file = output_file.with_suffix('.tsv.gz')
            with open(gz_file, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            # Extract the gzipped file
            with gzip.open(gz_file, 'rb') as f_in:
                with open(output_file, 'wb') as f_out:
                    f_out.write(f_in.read())
            
            # Clean up
            gz_file.unlink()
            return output_file
            
        except Exception as e:
            logger.error(f"Error processing dataset {dataset_type}: {str(e)}")
            raise
    
    def load_reviews(self, min_votes: int = 100) -> pd.DataFrame:
        """Load and preprocess IMDb reviews"""
        try:
            # Download datasets
            reviews_file = self.download_dataset('reviews')
            metadata_file = self.download_dataset('metadata')
            
            # Load reviews with proper dtype specification
            reviews_df = pd.read_csv(reviews_file, sep='\t', low_memory=False)
            metadata_df = pd.read_csv(metadata_file, sep='\t', low_memory=False)
            
            # Merge datasets
            df = reviews_df.merge(metadata_df, on='tconst')
            
            # Filter and clean data
            df = df[df['numVotes'] >= min_votes]  # Filter by minimum votes
            df = df[df['titleType'] == 'movie']   # Keep only movies
            
            return df
            
        except Exception as e:
            logger.error(f"Error loading reviews: {str(e)}")
            raise
    
    def prepare_training_data(self, num_samples: int = 10000) -> Tuple[List[str], List[int]]:
        """Prepare training data for the classifier"""
        try:
            df = self.load_reviews()
            logger.info(f"Loaded {len(df)} total movies")
            
            # Define criteria for genuine vs fake reviews
            genuine_positive = df[
                (df['averageRating'] >= 7.5) & 
                (df['numVotes'] > df['numVotes'].quantile(0.75))
            ]
            genuine_negative = df[
                (df['averageRating'] <= 4.5) & 
                (df['numVotes'] > df['numVotes'].quantile(0.75))
            ]
            potential_fake = df[
                ((df['averageRating'] >= 9.0) | (df['averageRating'] <= 2.0)) &
                (df['numVotes'] < df['numVotes'].quantile(0.25))
            ]
            
            logger.info(f"Found {len(genuine_positive)} genuine positive reviews")
            logger.info(f"Found {len(genuine_negative)} genuine negative reviews")
            logger.info(f"Found {len(potential_fake)} potential fake reviews")
            
            # Calculate maximum possible samples while maintaining balance
            max_genuine_per_class = min(len(genuine_positive), len(genuine_negative)) // 2
            max_fake = len(potential_fake)
            max_total_samples = min(num_samples, 2 * min(max_genuine_per_class, max_fake))
            
            samples_per_class = max_total_samples // 2
            samples_per_genuine = samples_per_class // 2
            
            logger.info(f"Using {samples_per_class * 2} total samples")
            
            # Sample with replacement if necessary
            genuine = pd.concat([
                genuine_positive.sample(n=samples_per_genuine, replace=True, random_state=42),
                genuine_negative.sample(n=samples_per_genuine, replace=True, random_state=43)
            ])
            fake = potential_fake.sample(n=samples_per_class, replace=True, random_state=44)
            
            # Combine and shuffle
            all_data = pd.concat([genuine, fake]).sample(frac=1, random_state=42)
            
            # Prepare labels (0 for genuine, 1 for fake)
            reviews = all_data['primaryTitle'].tolist()
            labels = [0] * len(genuine) + [1] * len(fake)
            
            return reviews, labels
            
        except Exception as e:
            logger.error(f"Error preparing training data: {str(e)}")
            raise

def get_training_data(num_samples: int = 10000) -> Tuple[List[str], List[int]]:
    """Convenience function to get training data"""
    collector = IMDbDataCollector()
    return collector.prepare_training_data(num_samples) 