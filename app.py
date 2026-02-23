from flask import Flask, render_template, request, jsonify, redirect, url_for, flash
import pandas as pd
import numpy as np
import os  # Removed duplicate import
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
import bcrypt
import json
import re  # Moved import to top
import google.generativeai as genai

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-change-this-in-production'

# Configure Google Gemini API
api_key = os.getenv('GOOGLE_API_KEY')
if not api_key:
    print("WARNING: GOOGLE_API_KEY environment variable not set!")
    print("Prediction feature will use fallback mode.")
else:
    genai.configure(api_key=api_key)

# Initialize Flask-Login
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

# User class
class User(UserMixin):
    def __init__(self, id, username, email, password_hash):
        self.id = id
        self.username = username
        self.email = email
        self.password_hash = password_hash

# User management functions
def load_users():
    users_file = 'data/users.json'
    if os.path.exists(users_file):
        with open(users_file, 'r') as f:
            return json.load(f)
    return {}

def save_users(users):
    users_file = 'data/users.json'
    os.makedirs('data', exist_ok=True)
    with open(users_file, 'w') as f:
        json.dump(users, f)

def hash_password(password):
    return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')

def check_password(password, hashed):
    return bcrypt.checkpw(password.encode('utf-8'), hashed.encode('utf-8'))

@login_manager.user_loader
def load_user(user_id):
    users = load_users()
    if user_id in users:
        user_data = users[user_id]
        return User(user_id, user_data['username'], user_data['email'], user_data['password_hash'])
    return None

# Preprocess data (similar to the reference project)
def preprocess_data(df):
    # Fill NaN with empty strings
    df.fillna('', inplace=True)
    
    # Combine text fields
    df['text'] = df['title'] + ' ' + df['location'] + ' ' + df['company_profile'] + ' ' + df['description'] + ' ' + df['requirements'] + ' ' + df['benefits'] + ' ' + df['required_experience'] + ' ' + df['required_education'] + ' ' + df['industry'] + ' ' + df['function']
    
    # Calculate character count
    df['character_count'] = df['text'].apply(len)
    
    # Simple ratio (placeholder)
    df['ratio'] = 0  # Would need location processing
    
    return df

# Load data
data_path = 'data/fake_job_postings.csv'
df = pd.DataFrame()  # Initialize as empty DataFrame

if os.path.exists(data_path):
    try:
        df = pd.read_csv(data_path)
        print(f"✅ Loaded {len(df)} records from {data_path}")
        df = preprocess_data(df)
        print("✅ Data preprocessing completed")
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        df = pd.DataFrame()
else:
    print(f"⚠️ Data file not found: {data_path}")
    print("The application will run with limited functionality.")
    df = pd.DataFrame()

# Routes
@app.route('/')
def index():
    # Dashboard metrics
    if df.empty or 'fraudulent' not in df.columns:
        total_jobs = 0
        fake_jobs = 0
        real_jobs = 0
        accuracy = 0.0
        recent_jobs = []
        industries = ['Technology', 'Healthcare', 'Finance', 'Education', 'Other']
        industry_values = [120, 80, 60, 40, 20]
    else:
        total_jobs = len(df)
        fake_jobs = int(df['fraudulent'].sum())
        real_jobs = total_jobs - fake_jobs
        accuracy = 0.97  # Placeholder
        
        # Sample recent jobs
        if 'job_id' in df.columns:
            recent_jobs = df.tail(10)[['job_id', 'title', 'location', 'company_profile', 'fraudulent']].rename(columns={'company_profile': 'company'}).to_dict('records')
        else:
            recent_jobs = []
        
        # Top industries
        if 'industry' in df.columns:
            industry_counts = df['industry'].value_counts().head(5).to_dict()
            industries = list(industry_counts.keys())
            industry_values = list(industry_counts.values())
        else:
            industries = ['Technology', 'Healthcare', 'Finance', 'Education', 'Other']
            industry_values = [120, 80, 60, 40, 20]
    
    return render_template('index.html', 
                         total_jobs=total_jobs,
                         fake_jobs=fake_jobs,
                         real_jobs=real_jobs,
                         accuracy=accuracy,
                         recent_jobs=recent_jobs,
                         industries=industries,
                         industry_values=industry_values)

@app.route('/exploration')
def exploration():
    return render_template('exploration.html')

@app.route('/nlp_analysis')
def nlp_analysis():
    return render_template('nlp_analysis.html')

@app.route('/model_training')
def model_training():
    return render_template('model_training.html')

@app.route('/prediction')
def prediction():
    return render_template('prediction.html')

@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        # Combine text fields
        text = f"""Job Title: {data.get('title', '')}
Location: {data.get('location', '')}
Company: {data.get('company', '')}
Description: {data.get('description', '')}
Requirements: {data.get('requirements', '')}
Industry: {data.get('industry', '')}
Function: {data.get('function', '')}"""
        
        # Check if API key is configured
        if not os.getenv('GOOGLE_API_KEY'):
            # Return varied demo results based on input
            import random
            # Use title length as simple heuristic for demo
            title = data.get('title', '')
            if len(title) > 10:
                fraudulent = random.choice([0, 1])
            else:
                fraudulent = 1 if random.random() > 0.5 else 0
            
            probability = 0.7 if fraudulent else 0.3
            return jsonify({
                'fraudulent': fraudulent,
                'probability': probability,
                'demo_mode': True
            })
        
        # Use Gemini API for prediction
        model = genai.GenerativeModel('gemini-pro')
        prompt = f"""Analyze the following job posting and determine if it is likely fake or real. 
Consider factors like:
- Unrealistic salary promises
- Poor grammar and spelling
- Suspicious company details
- Lack of specific requirements
- Too good to be true offers

Respond with only 'fake' or 'real' and a confidence score between 0 and 1.
Example format: 'fake 0.85' or 'real 0.92'

Job Posting:
{text}"""
        
        response = model.generate_content(prompt)
        result = response.text.strip().lower()
        
        # Extract classification and probability
        if 'fake' in result:
            fraudulent = 1
            probability = 0.8  # Default
        elif 'real' in result:
            fraudulent = 0
            probability = 0.2  # Default
        else:
            # Fallback - use simple heuristics
            fraudulent = 1 if 'urgent' in text.lower() or 'immediate' in text.lower() else 0
            probability = 0.6
        
        # Try to extract probability from response
        prob_match = re.search(r'(\d+\.\d+)', result)
        if prob_match:
            probability = float(prob_match.group(1))
            # Ensure probability makes sense with classification
            if fraudulent == 1 and probability < 0.5:
                probability = 1 - probability
            elif fraudulent == 0 and probability > 0.5:
                probability = 1 - probability
        
        return jsonify({
            'fraudulent': fraudulent,
            'probability': probability
        })
        
    except Exception as e:
        print(f"Prediction error: {str(e)}")
        # Return a varied demo result instead of always fake
        import random
        return jsonify({
            'fraudulent': random.choice([0, 1]),
            'probability': round(random.uniform(0.3, 0.9), 2),
            'error': str(e),
            'demo_mode': True
        })

# Authentication routes
@app.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('index'))
    
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        
        print(f"Login attempt - Username: {username}")  # Debug print
        
        users = load_users()
        print(f"Users loaded: {users}")  # Debug print
        
        # Find user by username
        user_data = None
        user_id = None
        for uid, udata in users.items():
            print(f"Checking user: {udata['username']}")  # Debug print
            if udata['username'] == username:
                user_data = udata
                user_id = uid
                print(f"User found: {user_data}")  # Debug print
                break
        
        if user_data:
            print(f"Password check: {check_password(password, user_data['password_hash'])}")  # Debug print
        
        if user_data and check_password(password, user_data['password_hash']):
            user = User(user_id, user_data['username'], user_data['email'], user_data['password_hash'])
            login_user(user)
            flash('Logged in successfully!', 'success')
            print("Login successful!")  # Debug print
            next_page = request.args.get('next')
            return redirect(next_page) if next_page else redirect(url_for('index'))
        else:
            flash('Invalid username or password', 'error')
            print("Login failed!")  # Debug print
    
    return render_template('login.html')
@app.route('/register', methods=['GET', 'POST'])
def register():
    if current_user.is_authenticated:
        return redirect(url_for('index'))
    
    if request.method == 'POST':
        username = request.form.get('username')
        email = request.form.get('email')
        password = request.form.get('password')
        confirm_password = request.form.get('confirm_password')
        
        if password != confirm_password:
            flash('Passwords do not match', 'error')
            return render_template('register.html')
        
        if len(password) < 6:
            flash('Password must be at least 6 characters long', 'error')
            return render_template('register.html')
        
        users = load_users()
        
        # Check if username or email already exists
        for uid, udata in users.items():
            if udata['username'] == username:
                flash('Username already exists', 'error')
                return render_template('register.html')
            if udata['email'] == email:
                flash('Email already exists', 'error')
                return render_template('register.html')
        
        # Create new user
        user_id = str(len(users) + 1)
        users[user_id] = {
            'username': username,
            'email': email,
            'password_hash': hash_password(password)
        }
        save_users(users)
        
        # Auto login after registration
        user = User(user_id, username, email, users[user_id]['password_hash'])
        login_user(user)
        
        flash('Account created successfully! You are now logged in.', 'success')
        return redirect(url_for('index'))
    
    return render_template('register.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    flash('Logged out successfully!', 'success')
    return redirect(url_for('login'))

if __name__ == '__main__':
    app.run(debug=True)