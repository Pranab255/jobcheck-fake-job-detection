from flask import Flask, render_template, request, jsonify, redirect, url_for, flash
import pandas as pd
import numpy as np
import os
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
import bcrypt
import json
import re
import google.generativeai as genai
from functools import wraps  # नया import
from datetime import datetime  # नया import

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-change-this-in-production'

# Admin required decorator
def admin_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not current_user.is_authenticated or not current_user.is_admin:
            flash('Access denied. Admin privileges required.', 'error')
            return redirect(url_for('index'))
        return f(*args, **kwargs)
    return decorated_function

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

# User class - is_admin field के साथ
class User(UserMixin):
    def __init__(self, id, username, email, password_hash, is_admin=False):
        self.id = id
        self.username = username
        self.email = email
        self.password_hash = password_hash
        self.is_admin = is_admin

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
        json.dump(users, f, indent=2)

def hash_password(password):
    return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')

def check_password(password, hashed):
    return bcrypt.checkpw(password.encode('utf-8'), hashed.encode('utf-8'))

@login_manager.user_loader
def load_user(user_id):
    users = load_users()
    if user_id in users:
        user_data = users[user_id]
        return User(
            user_id, 
            user_data['username'], 
            user_data['email'], 
            user_data['password_hash'],
            user_data.get('is_admin', False)
        )
    return None

# Preprocess data
def preprocess_data(df):
    df.fillna('', inplace=True)
    df['text'] = df['title'] + ' ' + df['location'] + ' ' + df['company_profile'] + ' ' + df['description'] + ' ' + df['requirements'] + ' ' + df['benefits'] + ' ' + df['required_experience'] + ' ' + df['required_education'] + ' ' + df['industry'] + ' ' + df['function']
    df['character_count'] = df['text'].apply(len)
    df['ratio'] = 0
    return df

# Load data
data_path = 'data/fake_job_postings.csv'
df = pd.DataFrame()

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
    df = pd.DataFrame()

# Routes
@app.route('/')
def index():
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
        accuracy = 0.97
        
        if 'job_id' in df.columns:
            recent_jobs = df.tail(10)[['job_id', 'title', 'location', 'company_profile', 'fraudulent']].rename(columns={'company_profile': 'company'}).to_dict('records')
        else:
            recent_jobs = []
        
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
        text = f"""Job Title: {data.get('title', '')}
Location: {data.get('location', '')}
Company: {data.get('company', '')}
Description: {data.get('description', '')}
Requirements: {data.get('requirements', '')}
Industry: {data.get('industry', '')}
Function: {data.get('function', '')}"""
        
        if not os.getenv('GOOGLE_API_KEY'):
            import random
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
        
        if 'fake' in result:
            fraudulent = 1
            probability = 0.8
        elif 'real' in result:
            fraudulent = 0
            probability = 0.2
        else:
            fraudulent = 1 if 'urgent' in text.lower() or 'immediate' in text.lower() else 0
            probability = 0.6
        
        prob_match = re.search(r'(\d+\.\d+)', result)
        if prob_match:
            probability = float(prob_match.group(1))
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
        
        users = load_users()
        
        user_data = None
        user_id = None
        for uid, udata in users.items():
            if udata['username'] == username:
                user_data = udata
                user_id = uid
                break
        
        if user_data and check_password(password, user_data['password_hash']):
            user = User(user_id, user_data['username'], user_data['email'], user_data['password_hash'], user_data.get('is_admin', False))
            login_user(user)
            flash('Logged in successfully!', 'success')
            next_page = request.args.get('next')
            return redirect(next_page) if next_page else redirect(url_for('index'))
        else:
            flash('Invalid username or password', 'error')
    
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
        
        for uid, udata in users.items():
            if udata['username'] == username:
                flash('Username already exists', 'error')
                return render_template('register.html')
            if udata['email'] == email:
                flash('Email already exists', 'error')
                return render_template('register.html')
        
        user_id = str(len(users) + 1)
        users[user_id] = {
            'username': username,
            'email': email,
            'password_hash': hash_password(password),
            'is_admin': False,
            'created_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        save_users(users)
        
        user = User(user_id, username, email, users[user_id]['password_hash'], False)
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

# ==================== ADMIN ROUTES ====================

@app.route('/admin')
@login_required
@admin_required
def admin_dashboard():
    users = load_users()
    
    # Statistics
    total_users = len(users)
    total_admins = sum(1 for u in users.values() if u.get('is_admin', False))
    total_predictions = 1250  # Placeholder - आप अपना actual data डाल सकते हैं
    fake_detected = 866
    
    # Recent users
    recent_users = []
    for uid, udata in list(users.items())[-5:]:
        user_info = {'id': uid, **udata}
        recent_users.append(user_info)
    
    # Activity log (placeholder)
    activities = [
        {'user': 'admin', 'action': 'Logged in', 'time': '2 minutes ago'},
        {'user': 'john_doe', 'action': 'Analyzed job posting', 'time': '15 minutes ago'},
        {'user': 'jane_smith', 'action': 'Registered new account', 'time': '1 hour ago'},
        {'user': 'admin', 'action': 'Updated system settings', 'time': '3 hours ago'},
        {'user': 'mike_wilson', 'action': 'Detected fake job', 'time': '5 hours ago'}
    ]
    
    return render_template('admin/dashboard.html',
                         total_users=total_users,
                         total_admins=total_admins,
                         total_predictions=total_predictions,
                         fake_detected=fake_detected,
                         recent_users=recent_users,
                         activities=activities)

@app.route('/admin/users')
@login_required
@admin_required
def admin_users():
    users = load_users()
    users_list = [{'id': uid, **udata} for uid, udata in users.items()]
    return render_template('admin/users.html', users=users_list)

@app.route('/admin/user/<user_id>')
@login_required
@admin_required
def admin_user_detail(user_id):
    users = load_users()
    if user_id not in users:
        flash('User not found', 'error')
        return redirect(url_for('admin_users'))
    
    user_data = users[user_id]
    return render_template('admin/user_detail.html', user=user_data, user_id=user_id)

@app.route('/admin/user/<user_id>/toggle-admin', methods=['POST'])
@login_required
@admin_required
def toggle_admin(user_id):
    users = load_users()
    
    if user_id not in users:
        return jsonify({'success': False, 'error': 'User not found'})
    
    # Don't allow toggling your own admin status
    if user_id == current_user.id:
        return jsonify({'success': False, 'error': 'Cannot change your own admin status'})
    
    users[user_id]['is_admin'] = not users[user_id].get('is_admin', False)
    save_users(users)
    
    return jsonify({
        'success': True,
        'is_admin': users[user_id]['is_admin']
    })

@app.route('/admin/user/<user_id>/delete', methods=['POST'])
@login_required
@admin_required
def delete_user(user_id):
    users = load_users()
    
    if user_id not in users:
        return jsonify({'success': False, 'error': 'User not found'})
    
    # Don't allow deleting yourself
    if user_id == current_user.id:
        return jsonify({'success': False, 'error': 'Cannot delete your own account'})
    
    del users[user_id]
    save_users(users)
    
    return jsonify({'success': True})

@app.route('/admin/analytics')
@login_required
@admin_required
def admin_analytics():
    return render_template('admin/analytics.html')

@app.route('/admin/settings')
@login_required
@admin_required
def admin_settings():
    return render_template('admin/settings.html')

@app.route('/admin/api-key', methods=['POST'])
@login_required
@admin_required
def update_api_key():
    new_api_key = request.json.get('api_key')
    os.environ['GOOGLE_API_KEY'] = new_api_key
    genai.configure(api_key=new_api_key)
    return jsonify({'success': True, 'message': 'API key updated successfully'})

@app.route('/admin/system-status')
@login_required
@admin_required
def system_status():
    import psutil
    import platform
    
    status = {
        'python_version': platform.python_version(),
        'os': platform.system(),
        'cpu_usage': psutil.cpu_percent(),
        'memory_usage': psutil.virtual_memory().percent,
        'disk_usage': psutil.disk_usage('/').percent,
        'data_file_exists': os.path.exists('data/fake_job_postings.csv'),
        'total_users': len(load_users()),
        'google_api_configured': bool(os.getenv('GOOGLE_API_KEY'))
    }
    
    return jsonify(status)

@app.route('/admin/update-settings', methods=['POST'])
@login_required
@admin_required
def update_settings():
    settings = request.json
    # Save settings to a config file
    with open('data/admin_settings.json', 'w') as f:
        json.dump(settings, f, indent=2)
    return jsonify({'success': True})

if __name__ == '__main__':
    app.run(debug=True)