import sqlite3
import os
import datetime
from werkzeug.security import generate_password_hash, check_password_hash

# Database file path
DB_PATH = 'movie_review_detector.db'

def init_db():
    """Initialize the database with required tables"""
    if not os.path.exists(DB_PATH):
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # Create users table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id TEXT PRIMARY KEY,
            username TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        ''')
        
        # Create analysis_history table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS analysis_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT NOT NULL,
            movie_title TEXT,
            review_text TEXT NOT NULL,
            prediction TEXT NOT NULL,
            confidence REAL NOT NULL,
            features TEXT NOT NULL,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
        ''')
        
        conn.commit()
        conn.close()
        print(f"Database initialized at {DB_PATH}")

def get_user(user_id):
    """Get user by ID"""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    cursor.execute('SELECT * FROM users WHERE id = ?', (user_id,))
    user = cursor.fetchone()
    
    conn.close()
    return user

def get_user_by_username(username):
    """Get user by username"""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    cursor.execute('SELECT * FROM users WHERE username = ?', (username,))
    user = cursor.fetchone()
    
    conn.close()
    return user

def create_user(username, password):
    """Create a new user"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Check if username already exists
    cursor.execute('SELECT id FROM users WHERE username = ?', (username,))
    if cursor.fetchone():
        conn.close()
        return False, "Username already exists"
    
    # Create new user
    user_id = username  # Using username as ID for simplicity
    hashed_password = generate_password_hash(password)
    
    cursor.execute(
        'INSERT INTO users (id, username, password) VALUES (?, ?, ?)',
        (user_id, username, hashed_password)
    )
    
    conn.commit()
    conn.close()
    return True, user_id

def verify_user(username, password):
    """Verify user credentials"""
    user = get_user_by_username(username)
    
    if user and check_password_hash(user['password'], password):
        return True, user
    return False, None

def save_analysis(user_id, movie_title, review_text, prediction):
    """Save analysis to user history"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Extract data from prediction
    pred_type = prediction['prediction']
    confidence = prediction['confidence']
    features = prediction['features']
    
    # Convert features to JSON string
    features_json = str(features)
    
    cursor.execute(
        '''INSERT INTO analysis_history 
           (user_id, movie_title, review_text, prediction, confidence, features)
           VALUES (?, ?, ?, ?, ?, ?)''',
        (user_id, movie_title, review_text, pred_type, confidence, features_json)
    )
    
    conn.commit()
    conn.close()
    return True

def get_user_analyses(user_id, limit=100):
    """Get user's analysis history"""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    cursor.execute(
        '''SELECT * FROM analysis_history 
           WHERE user_id = ? 
           ORDER BY timestamp DESC 
           LIMIT ?''',
        (user_id, limit)
    )
    
    analyses = cursor.fetchall()
    conn.close()
    return analyses

def get_user_stats(user_id):
    """Get user statistics"""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    # Get total reviews analyzed
    cursor.execute('SELECT COUNT(*) FROM analysis_history WHERE user_id = ?', (user_id,))
    reviews_analyzed = cursor.fetchone()[0]
    
    # Get fake reviews detected
    cursor.execute(
        'SELECT COUNT(*) FROM analysis_history WHERE user_id = ? AND prediction = "fake"',
        (user_id,)
    )
    fake_detected = cursor.fetchone()[0]
    
    conn.close()
    
    # Calculate fake percentage
    fake_percentage = (fake_detected / reviews_analyzed * 100) if reviews_analyzed > 0 else 0
    
    # Mock accuracy (in a real app, you'd calculate this based on user feedback)
    user_accuracy = 85
    
    return {
        'reviews_analyzed': reviews_analyzed,
        'fake_detected': round(fake_percentage, 1),
        'accuracy': user_accuracy
    } 