import os
import joblib
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'model', 'model.joblib')

logger.info(f"Loading model from {MODEL_PATH}...")

try:
    model_pipeline = joblib.load(MODEL_PATH)
    logger.info("Model pipeline loaded successfully.")
except Exception as e:
    logger.error(f"Failed to load model: {e}")
    raise e

def time_to_minutes(t):
    """Converts a HH:MM time string to an integer representing minutes from midnight."""
    if isinstance(t, str):
        try:
            h, m = map(int, t.split(':'))
            return h * 60 + m
        except ValueError:
            # Handle cases where the time string is malformed
            return 0
    return 0

def preprocess_input(input_data: pd.DataFrame):
    """
    Preprocesses input data to match the format expected by the trained model.
    Applies the same transformations used during training.
    """
    logger.info("Preprocessing input data...")
    
    # Create a copy to avoid modifying the original data
    processed_data = input_data.copy()
    
    # Clean column names by removing leading/trailing whitespace
    processed_data.columns = processed_data.columns.str.strip()
    
    # Apply time conversion to 'Time' column if it exists
    if 'Time' in processed_data.columns:
        processed_data['Time'] = processed_data['Time'].fillna('00:00').apply(time_to_minutes)
    
    # Remove the target column if it exists (shouldn't be in prediction data)
    if 'Class' in processed_data.columns:
        processed_data = processed_data.drop('Class', axis=1)
        logger.warning("Removed 'Class' column from input data as it's the target variable.")
    
    logger.info("Input data preprocessing complete.")
    return processed_data

def make_prediction(input_data: pd.DataFrame):
    """
    Makes predictions using the loaded model pipeline.
    The input data will be preprocessed to match the model's expected format.
    
    Args:
        input_data (pd.DataFrame): Raw transaction data
        
    Returns:
        dict: Contains predictions, probabilities, and metadata
    """
    logger.info("make_prediction called.")

    if not isinstance(input_data, pd.DataFrame):
        logger.error("Input data is not a pandas DataFrame.")
        raise ValueError("Input data must be a pandas DataFrame")

    try:
        # Preprocess the input data
        processed_data = preprocess_input(input_data)
        
        # Make predictions using the pipeline
        predictions = model_pipeline.predict(processed_data)
        probabilities = model_pipeline.predict_proba(processed_data)
        
        logger.info(f"Prediction completed for {len(predictions)} samples.")
        
        return {
            "predictions": predictions.tolist(),
            "probabilities": probabilities.tolist(),
            "prediction_labels": ["Legitimate" if pred == 0 else "Fraud" for pred in predictions],
            "fraud_probabilities": [prob[1] for prob in probabilities],  # Probability of fraud (class 1)
            "num_samples": len(predictions)
        }
    except Exception as e:
        logger.error(f"Error during prediction: {e}")
        raise

def predict_single_transaction(transaction_dict: dict):
    """
    Convenience function to predict on a single transaction.
    
    Args:
        transaction_dict (dict): Dictionary containing transaction features
        
    Returns:
        dict: Prediction result for the single transaction
    """
    logger.info("predict_single_transaction called.")
    
    # Convert dictionary to DataFrame
    transaction_df = pd.DataFrame([transaction_dict])
    
    # Make prediction
    result = make_prediction(transaction_df)
    
    # Return single transaction result
    return {
        "prediction": result["predictions"][0],
        "prediction_label": result["prediction_labels"][0],
        "fraud_probability": result["fraud_probabilities"][0],
        "confidence": max(result["probabilities"][0])  # Highest probability as confidence
    }