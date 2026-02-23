# JobCheck - Fake Job Post Detection System

A comprehensive web-based application for detecting fake job postings using Natural Language Processing (NLP) and Machine Learning techniques.

## Features

- **5-Page Dashboard Interface**: Complete web application with dashboard, data exploration, NLP analysis, model training, and real-time prediction pages
- **Advanced ML Pipeline**: Utilizes multiple algorithms including Logistic Regression, Random Forest, XGBoost, and SGD Classifier
- **Real-time Prediction**: Analyze new job postings instantly
- **Interactive Visualizations**: Plotly-powered charts and graphs
- **Batch Processing**: Upload CSV files for bulk analysis
- **Responsive Design**: Mobile-friendly Bootstrap 5 interface

## Dataset

The application uses the "Real or Fake Fake Jobposting Prediction" dataset from Kaggle, containing 17,880 job postings with 18 features including:
- Job title, company, location
- Job description and requirements
- Employment type, industry, function
- Binary features (has_company_logo, has_questions, telecommuting)
- Target variable: fraudulent (0 = real, 1 = fake)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/jobcheck-fake-job-detection.git
cd jobcheck-fake-job-detection
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Run the application:
```bash
python app.py
```

5. Open your browser and navigate to `http://localhost:5000`

## Project Structure

```
jobcheck-fake-job-detection/
├── app.py                 # Main Flask application
├── requirements.txt       # Python dependencies
├── Dockerfile            # Docker configuration
├── .env.example          # Environment variables template
├── README.md             # Project documentation
├── data/
│   ├── fake_job_postings.csv    # Original dataset
│   └── processed_data.csv       # Processed data
├── models/
│   ├── model.pkl               # Trained model
│   ├── vectorizer.pkl          # TF-IDF vectorizer
│   └── model_metrics.json      # Model performance metrics
├── notebooks/
│   ├── 01_eda.ipynb           # Exploratory Data Analysis
│   ├── 02_feature_engineering.ipynb
│   └── 03_model_training.ipynb
├── src/
│   ├── __init__.py
│   ├── data_preprocessing.py
│   ├── feature_extraction.py
│   ├── model_training.py
│   └── utils.py
├── static/
│   ├── css/
│   │   └── style.css
│   ├── js/
│   │   └── main.js
│   └── images/
├── templates/
│   ├── base.html
│   ├── index.html          # Dashboard
│   ├── exploration.html    # Data Exploration
│   ├── nlp_analysis.html   # NLP Analysis
│   ├── model_training.html # Model Training
│   └── prediction.html     # Real-time Prediction
└── tests/
    └── test_app.py
```

## Usage

### Web Interface

1. **Dashboard**: Overview of key metrics and visualizations
2. **Data Exploration**: Dataset statistics, distributions, and correlations
3. **NLP Analysis**: Text preprocessing, TF-IDF, sentiment analysis
4. **Model Training**: Model comparison, performance metrics, feature importance
5. **Prediction**: Real-time analysis of new job postings

### API Usage

```python
import requests

# Real-time prediction
data = {
    "title": "Software Engineer",
    "company": "Tech Corp",
    "description": "We are looking for a skilled software engineer...",
    "location": "New York, NY",
    # ... other fields
}

response = requests.post("http://localhost:5000/api/predict", json=data)
result = response.json()
```

## Model Performance

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| Logistic Regression | 0.965 | 0.720 | 0.650 | 0.683 |
| Random Forest | 0.972 | 0.780 | 0.720 | 0.749 |
| XGBoost | 0.975 | 0.800 | 0.750 | 0.774 |
| **SGD Classifier** | **0.974** | **0.790** | **0.785** | **0.788** |

## Deployment

### Local Development
```bash
python app.py
```

### Docker
```bash
docker build -t jobcheck .
docker run -p 5000:5000 jobcheck
```

### Cloud Deployment
The application can be deployed to Heroku, AWS, or Azure with the provided Dockerfile and requirements.txt.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Dataset from Kaggle: [Real or Fake Fake Jobposting Prediction](https://www.kaggle.com/shivamb/real-or-fake-fake-jobposting-prediction)
- Inspired by the work of Anshupriya Srivastava
- Built with Flask, scikit-learn, and Plotly