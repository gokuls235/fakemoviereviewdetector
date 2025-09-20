import schedule
import time
import logging
from datetime import datetime
from pathlib import Path
import joblib
from typing import Optional
import json

from .text_classifier import ReviewClassifier, train_model

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('model_training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ModelTrainer:
    def __init__(self, model_dir: str = 'models'):
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(exist_ok=True)
        
        # Initialize metrics history
        self.metrics_file = self.model_dir / 'training_metrics.json'
        self.metrics_history = self._load_metrics_history()
        
        # Ensure we have initial metrics
        if not self.metrics_history['models']:
            logger.info("No existing models found. Training initial model...")
            self.train_new_model()
    
    def _load_metrics_history(self) -> dict:
        """Load training metrics history"""
        if self.metrics_file.exists():
            try:
                with open(self.metrics_file, 'r') as f:
                    return json.load(f)
            except json.JSONDecodeError:
                logger.warning("Error reading metrics file. Creating new metrics history.")
                return {'models': []}
        return {'models': []}
    
    def _save_metrics_history(self):
        """Save training metrics history"""
        try:
            with open(self.metrics_file, 'w') as f:
                json.dump(self.metrics_history, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving metrics history: {str(e)}")
    
    def train_new_model(self, num_samples: int = 10000) -> Optional[str]:
        """Train a new model and save it if it performs better"""
        try:
            logger.info("Starting new model training...")
            
            # Train new model
            classifier, metrics = train_model()
            
            # Generate model filename with timestamp
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            model_file = self.model_dir / f'review_classifier_{timestamp}.joblib'
            
            # Save the model
            classifier.save_model(str(model_file))
            
            # Update metrics history
            training_record = {
                'timestamp': timestamp,
                'model_file': str(model_file),
                'metrics': metrics,
                'num_samples': num_samples
            }
            self.metrics_history['models'].append(training_record)
            self._save_metrics_history()
            
            logger.info(f"New model trained and saved: {model_file}")
            logger.info(f"Metrics: {metrics}")
            
            return str(model_file)
            
        except Exception as e:
            logger.error(f"Error training new model: {str(e)}")
            return None
    
    def get_best_model(self) -> Optional[str]:
        """Get the path to the best performing model"""
        if not self.metrics_history['models']:
            return None
            
        try:
            # Sort by F1 score
            sorted_models = sorted(
                self.metrics_history['models'],
                key=lambda x: x['metrics']['f1_score'],
                reverse=True
            )
            
            best_model_file = sorted_models[0]['model_file']
            
            # Verify the file exists
            if not Path(best_model_file).exists():
                logger.warning(f"Best model file not found: {best_model_file}")
                return None
                
            return best_model_file
            
        except Exception as e:
            logger.error(f"Error getting best model: {str(e)}")
            return None
    
    def cleanup_old_models(self, keep_best_n: int = 3):
        """Remove old models, keeping the best N"""
        if len(self.metrics_history['models']) <= keep_best_n:
            return
        
        try:
            # Sort models by performance
            sorted_models = sorted(
                self.metrics_history['models'],
                key=lambda x: x['metrics']['f1_score'],
                reverse=True
            )
            
            # Keep the best N models
            models_to_keep = set(m['model_file'] for m in sorted_models[:keep_best_n])
            
            # Remove old models
            for model in sorted_models[keep_best_n:]:
                try:
                    model_path = Path(model['model_file'])
                    if model_path.exists() and str(model_path) not in models_to_keep:
                        model_path.unlink()
                        logger.info(f"Removed old model: {model_path}")
                except Exception as e:
                    logger.error(f"Error removing old model {model['model_file']}: {str(e)}")
            
            # Update metrics history
            self.metrics_history['models'] = sorted_models[:keep_best_n]
            self._save_metrics_history()
            
        except Exception as e:
            logger.error(f"Error during cleanup: {str(e)}")

def schedule_training(trainer: ModelTrainer, interval_hours: int = 24):
    """Schedule regular model training"""
    def training_job():
        logger.info("Starting scheduled model training...")
        new_model = trainer.train_new_model()
        if new_model:
            trainer.cleanup_old_models()
        logger.info("Scheduled training completed.")
    
    # Schedule training job
    schedule.every(interval_hours).hours.do(training_job)
    
    while True:
        schedule.run_pending()
        time.sleep(60)  # Check every minute

if __name__ == '__main__':
    trainer = ModelTrainer()
    
    # Start scheduled training
    schedule_training(trainer) 