import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os

# Load data
data_path = 'data/fake_job_postings.csv'
df = pd.read_csv(data_path)

# Preprocess
df.fillna('', inplace=True)
df['text'] = df['title'] + ' ' + df['location'] + ' ' + df['company_profile'] + ' ' + df['description'] + ' ' + df['requirements'] + ' ' + df['benefits'] + ' ' + df['required_experience'] + ' ' + df['required_education'] + ' ' + df['industry'] + ' ' + df['function']

# Features and target
X = df['text']
y = df['fraudulent']

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Vectorize
vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Train model
model = LogisticRegression(random_state=42)
model.fit(X_train_vec, y_train)

# Evaluate
y_pred = model.predict(X_test_vec)
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.4f}')
print(classification_report(y_test, y_pred))

# Save model and vectorizer
os.makedirs('models', exist_ok=True)
joblib.dump(model, 'models/fake_job_model.pkl')
joblib.dump(vectorizer, 'models/tfidf_vectorizer.pkl')

print('Model and vectorizer saved.')