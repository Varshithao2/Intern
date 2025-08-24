import joblib
import pandas as pd

# Path to your saved model
MODEL_PATH = r"C:\323103382057\casestudy\credit-card-fraud\model\model.joblib"

def load_model():
    """Load the trained model pipeline from disk."""
    try:
        model_pipeline = joblib.load(MODEL_PATH)
        print("✅ Model loaded successfully.")
        return model_pipeline
    except FileNotFoundError:
        raise RuntimeError(f"❌ Model file not found at {MODEL_PATH}. Please train and save the model first.")

def get_sample_input():
    """Return a sample transaction for testing."""
    sample_data = {
        'Time': [600],  # Example: 10 hours (in minutes)
        'V1': [-1.3598071336738],
        'V2': [-0.0727811733098497],
        'V3': [2.53634673796914],
        'V4': [1.37815522427443],
        'V5': [-0.338320769942518],
        'V6': [0.462387777762292],
        'V7': [0.239598554061257],
        'V8': [0.0986979012610507],
        'V9': [0.363786969611213],
        'V10': [0.0907941719789316],
        'V11': [-0.551599533260813],
        'V12': [-0.617800855762348],
        'V13': [-0.991389847235408],
        'V14': [-0.311169353699879],
        'V15': [1.46817697209427],
        'V16': [-0.470400525259478],
        'V17': [0.207971241929242],
        'V18': [0.0257905801985591],
        'V19': [0.403992960255733],
        'V20': [0.251412098239705],
        'V21': [-0.018306777944153],
        'V22': [0.277837575558899],
        'V23': [-0.110473910188767],
        'V24': [0.0669280749146731],
        'V25': [0.128539358273528],
        'V26': [-0.189114843888824],
        'V27': [0.133558376740387],
        'V28': [-0.0210530534538215],
        'Amount': [149.62]
    }
    return pd.DataFrame(sample_data)

def main():
    # Load model
    model_pipeline = load_model()

    # Sample transaction
    input_df = get_sample_input()
    print("\n📝 Sample Input Data:")
    print(input_df.head())

    # Prediction
    prediction = model_pipeline.predict(input_df)[0]
    prediction_proba = model_pipeline.predict_proba(input_df)[0]

    # Results
    print("\n🔮 Prediction Results:")
    print(f"Predicted Class: {prediction} (0 = Legit, 1 = Fraud)")
    print(f"Probabilities → Legit: {prediction_proba[0]:.4f}, Fraud: {prediction_proba[1]:.4f}")

if __name__ == "__main__":
    main()
