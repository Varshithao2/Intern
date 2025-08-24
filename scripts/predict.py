import joblib
import pandas as pd
import sys
import os

# --- Configuration ---
MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "model", "model.joblib")

def load_model():
    """Load the trained model pipeline."""
    try:
        model_pipeline = joblib.load(MODEL_PATH)
        print("✅ Model loaded successfully.")
        return model_pipeline
    except FileNotFoundError:
        raise RuntimeError(f"❌ Model file not found at {MODEL_PATH}. Please train the model first.")

def predict_single(model_pipeline, input_dict):
    """Make prediction for a single transaction dictionary."""
    input_df = pd.DataFrame([input_dict])
    prediction = model_pipeline.predict(input_df)[0]
    proba = model_pipeline.predict_proba(input_df)[0]
    return prediction, proba

def predict_batch(model_pipeline, csv_path):
    """Make predictions for a batch of transactions from CSV."""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    df = pd.read_csv(csv_path)
    preds = model_pipeline.predict(df)
    probas = model_pipeline.predict_proba(df)
    df["Predicted_Class"] = preds
    df["Prob_Legit"] = probas[:, 0]
    df["Prob_Fraud"] = probas[:, 1]
    return df

def main():
    model_pipeline = load_model()

    # If a CSV is passed: batch prediction
    if len(sys.argv) > 1:
        csv_path = sys.argv[1]
        print(f"\n📂 Running batch predictions on: {csv_path}")
        results = predict_batch(model_pipeline, csv_path)
        print(results.head())
        out_path = "predictions.csv"
        results.to_csv(out_path, index=False)
        print(f"\n✅ Predictions saved to {out_path}")
    else:
        # Example single transaction (adjust features based on your dataset)
        sample_data = {
            "Time": 600,
            "V1": -1.3598071336738,
            "V2": -0.0727811733098497,
            "V3": 2.53634673796914,
            "V4": 1.37815522427443,
            "V5": -0.338320769942518,
            "V6": 0.462387777762292,
            "V7": 0.239598554061257,
            "V8": 0.0986979012610507,
            "V9": 0.363786969611213,
            "V10": 0.0907941719789316,
            "V11": -0.551599533260813,
            "V12": -0.617800855762348,
            "V13": -0.991389847235408,
            "V14": -0.311169353699879,
            "V15": 1.46817697209427,
            "V16": -0.470400525259478,
            "V17": 0.207971241929242,
            "V18": 0.0257905801985591,
            "V19": 0.403992960255733,
            "V20": 0.251412098239705,
            "V21": -0.018306777944153,
            "V22": 0.277837575558899,
            "V23": -0.110473910188767,
            "V24": 0.0669280749146731,
            "V25": 0.128539358273528,
            "V26": -0.189114843888824,
            "V27": 0.133558376740387,
            "V28": -0.0210530534538215,
            "Amount": 149.62,
        }

        prediction, proba = predict_single(model_pipeline, sample_data)
        print("\n🔮 Prediction Result (Single Input):")
        print(f"Predicted Class: {prediction} (0 = Legit, 1 = Fraud)")
        print(f"Probabilities → Legit: {proba[0]:.4f}, Fraud: {proba[1]:.4f}")

if __name__ == "__main__":
    main()
