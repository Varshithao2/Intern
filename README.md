# Credit Card Fraud Detection System

A comprehensive machine learning system for detecting fraudulent credit card transactions using FastAPI, Streamlit, and AWS services.

## Project Structure

```
credit-card-fraud/
│
├── backend/                      # Backend API (FastAPI)
│   ├── main.py                    # API entry point
│   ├── inference.py               # Load model & make predictions
│   ├── requirements.txt           # Backend dependencies
│   ├── utils/                     # Helper functions (S3 download, preprocessing, etc.)
│   └── config.py                  # AWS & app configuration
│
├── frontend/                      # Streamlit frontend
│   ├── app.py                     # Streamlit UI with Cognito login & API calls
│   ├── requirements.txt           # Frontend dependencies
│   └── config.toml                # Streamlit secrets/config
│
├── model/                         # Local model files (before upload to S3)
│   └── model.joblib
│
├── data/                          # Sample or synthetic datasets
│   └── transactions.csv
│
├── notebooks/                     # Jupyter/Colab notebooks
│   └── training.ipynb             # Original model training
│
├── scripts/                       # Utility scripts
│   ├── train.py                   # Train and save model
│   └── upload_model_s3.py         # Upload trained model to AWS S3
│
├── README.md                      # Project documentation
└── .gitignore                     # Git ignore rules
```

## Features

- **Machine Learning Model**: Fraud detection using Random Forest, Logistic Regression, and XGBoost
- **FastAPI Backend**: RESTful API for model inference
- **Streamlit Frontend**: Interactive web interface with AWS Cognito authentication
- **AWS Integration**: Model storage in S3, authentication via Cognito
- **Data Processing**: SMOTE for handling class imbalance, feature engineering
- **Model Evaluation**: Comprehensive metrics including ROC-AUC, precision, recall, F1-score

## Getting Started

1. **Train the Model**: Use the Jupyter notebook in `notebooks/training.ipynb` to train the fraud detection model
2. **Set up Backend**: Configure AWS credentials and run the FastAPI server
3. **Set up Frontend**: Configure Streamlit with AWS Cognito and run the web interface
4. **Deploy**: Upload model to S3 and deploy services to AWS

## Technology Stack

- **Machine Learning**: scikit-learn, XGBoost, imbalanced-learn
- **Backend**: FastAPI, uvicorn
- **Frontend**: Streamlit
- **Cloud Services**: AWS S3, AWS Cognito
- **Data Processing**: pandas, numpy
- **Visualization**: matplotlib, seaborn

## Installation

1. Clone the repository
2. Install backend dependencies: `pip install -r backend/requirements.txt`
3. Install frontend dependencies: `pip install -r frontend/requirements.txt`
4. Configure AWS credentials and settings
5. Run the training notebook to create the model
6. Start the backend and frontend services

## Usage

1. Access the Streamlit web interface
2. Login using AWS Cognito
3. Input transaction details
4. Get fraud prediction with confidence score
5. View prediction results and explanations
