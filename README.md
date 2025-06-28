# House Price Prediction - AutoML with MLflow

A machine learning application that predicts house prices using 3 AutoML models (XGBoost, LightGBM, Random Forest) with MLflow tracking and a Streamlit web interface.

## Features
- 🤖 3 AutoML models with hyperparameter tuning
- 📊 MLflow experiment tracking
- 🌐 Interactive Streamlit web app
- 🔄 Single and batch predictions
- 📈 Model comparison dashboard
- 🐳 Docker containerized deployment

## Quick Start

### Local Development
```bash
# Clone repository
git clone <your-repo-url>
cd house-price-prediction

# Install dependencies
pip install -r requirements.txt

# Train models
jupyter nbconvert --execute --to notebook --inplace house_price_automl_mlflow.ipynb

# Run Streamlit app
streamlit run app.py
```

### Docker Deployment
```bash
# Build and run
docker build -t house-price-app .
docker run -p 8501:8501 house-price-app
```

## EC2 Deployment
See deployment steps below for complete EC2 setup instructions.

## Project Structure
```
├── app.py                           # Streamlit web application
├── house_price_automl_mlflow.ipynb  # Model training notebook
├── requirements.txt                 # Python dependencies
├── Dockerfile                       # Docker configuration
├── House Price Prediction Dataset.csv # Dataset
└── models/                          # Trained models (generated)
```

## Models
1. **XGBoost** - Gradient boosting with GridSearchCV
2. **LightGBM** - Fast gradient boosting with RandomizedSearchCV
3. **Random Forest** - Ensemble method with GridSearchCV
