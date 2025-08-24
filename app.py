import streamlit as st
import pandas as pd
import json
import io
import plotly.express as px
import plotly.graph_objects as go
import joblib
import os
import logging
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

st.set_page_config(
    page_title="Credit Card Fraud Detection", 
    page_icon="💳", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        text-align: center;
        color: #1f77b4;
        font-size: 3rem;
        margin-bottom: 2rem;
    }
    .fraud-alert {
        background-color: #ffebee;
        border-left: 5px solid #f44336;
        padding: 10px;
        margin: 10px 0;
    }
    .safe-alert {
        background-color: #e8f5e8;
        border-left: 5px solid #4caf50;
        padding: 10px;
        margin: 10px 0;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #dee2e6;
    }
</style>
""", unsafe_allow_html=True)

# Model loading and inference functions
def create_demo_model():
    """Create a demo model when the trained model is not available."""
    logger.info("Creating demo model as fallback...")
    
    # Create a simple pipeline without SMOTE for demo to avoid neighbor issues
    demo_pipeline = Pipeline([
        ('classifier', RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced'))
    ])
    
    # Create some dummy training data for the demo
    np.random.seed(42)
    n_samples = 10000  # Larger sample size to ensure proper training
    
    # Generate fake transaction data that matches expected format
    demo_data = {
        'Time': np.random.randint(0, 1440, n_samples),  # Minutes in a day
        'Amount': np.random.exponential(100, n_samples),
    }
    
    # Add V1-V28 features (simulated PCA features)
    for i in range(1, 29):
        demo_data[f'V{i}'] = np.random.normal(0, 1, n_samples)
    
    # Create target with imbalance but ensure minimum samples for each class
    fraud_count = max(500, int(n_samples * 0.01))  # At least 500 fraud cases (1%)
    fraud_indices = np.random.choice(n_samples, size=fraud_count, replace=False)
    target = np.zeros(n_samples)
    target[fraud_indices] = 1
    
    # Create more realistic fraud patterns in the training data
    X_demo = pd.DataFrame(demo_data)
    
    # Make fraudulent transactions have different patterns
    for idx in fraud_indices:
        # Fraud transactions tend to have:
        # - Higher amounts (sometimes)
        if np.random.random() > 0.7:  # 30% chance of high amount fraud
            X_demo.loc[idx, 'Amount'] = np.random.uniform(1000, 5000)
        
        # - Different time patterns (late night/early morning)
        if np.random.random() > 0.6:  # 40% chance of unusual time
            X_demo.loc[idx, 'Time'] = np.random.choice([
                np.random.randint(0, 120),      # Late night (00:00-02:00)
                np.random.randint(1320, 1440)  # Very late (22:00-24:00)
            ])
        
        # - Unusual V features (simulate suspicious patterns)
        for v_col in [f'V{i}' for i in range(1, 29)]:
            if np.random.random() > 0.8:  # 20% chance to modify each V feature
                X_demo.loc[idx, v_col] = np.random.normal(0, 3)  # More extreme values
    
    y_demo = target
    
    logger.info(f"Demo data created: {len(X_demo)} samples, {sum(y_demo)} fraud cases")
    
    # Train the demo model
    demo_pipeline.fit(X_demo, y_demo)
    
    logger.info("Demo model created and trained successfully!")
    return demo_pipeline

@st.cache_resource
def load_model():
    """Load the trained model pipeline with caching, or create demo model as fallback."""
    try:
        # Try to find the model file in various locations
        possible_paths = [
            # Local development paths (app.py is now in root)
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "model", "model.joblib"),
            # Streamlit Cloud paths
            "/mount/src/intern/casestudy/credit-card-fraud/model/model.joblib",
            "/mount/src/intern/model/model.joblib", 
            "model/model.joblib",
            "./model/model.joblib",
            # Alternative paths
            os.path.join(os.getcwd(), "model", "model.joblib"),
            os.path.join(os.path.dirname(__file__), "model", "model.joblib"),
        ]
        
        model_path = None
        for path in possible_paths:
            abs_path = os.path.abspath(path)
            logger.info(f"Trying model path: {abs_path}")
            if os.path.exists(abs_path):
                model_path = abs_path
                break
        
        if model_path is not None:
            logger.info(f"Loading model from: {model_path}")
            model_pipeline = joblib.load(model_path)
            logger.info(f"Model loaded successfully from: {model_path}")
            return model_pipeline, None
        else:
            # Model file not found, create demo model
            logger.warning("Trained model not found, creating demo model...")
            demo_model = create_demo_model()
            return demo_model, "Using demo model (trained model not found)"
            
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        try:
            # Fallback to demo model
            logger.info("Creating demo model as fallback due to error...")
            demo_model = create_demo_model()
            return demo_model, f"Using demo model due to error: {str(e)}"
        except Exception as demo_error:
            error_msg = f"Failed to load model and create demo: {str(demo_error)}"
            logger.error(error_msg)
            return None, error_msg

def time_to_minutes(t):
    """Converts a HH:MM time string to an integer representing minutes from midnight."""
    if isinstance(t, str):
        try:
            h, m = map(int, t.split(':'))
            return h * 60 + m
        except ValueError:
            return 0
    return 0

def preprocess_input(input_data: pd.DataFrame):
    """
    Preprocesses input data to match the format expected by the trained model.
    """
    logger.info("Preprocessing input data...")
    
    processed_data = input_data.copy()
    processed_data.columns = processed_data.columns.str.strip()
    
    # Apply time conversion to 'Time' column if it exists
    if 'Time' in processed_data.columns:
        processed_data['Time'] = processed_data['Time'].fillna('00:00').apply(time_to_minutes)
    
    # Remove the target column if it exists
    if 'Class' in processed_data.columns:
        processed_data = processed_data.drop('Class', axis=1)
        st.warning("Removed 'Class' column from input data as it's the target variable.")
    
    # Ensure we have the expected columns for the demo model
    expected_cols = ['Time', 'Amount'] + [f'V{i}' for i in range(1, 29)]
    
    # Add missing columns with default values
    for col in expected_cols:
        if col not in processed_data.columns:
            if col == 'Time':
                processed_data[col] = 0  # Default time to midnight
            elif col == 'Amount':
                processed_data[col] = 100.0  # Default amount
            else:  # V1-V28 features
                processed_data[col] = 0.0  # Default PCA features to zero
    
    # Reorder columns to match expected order
    processed_data = processed_data[expected_cols]
    
    logger.info("Input data preprocessing complete.")
    return processed_data

def make_prediction(model_pipeline, input_data: pd.DataFrame):
    """
    Makes predictions using the loaded model pipeline.
    """
    try:
        processed_data = preprocess_input(input_data)
        predictions = model_pipeline.predict(processed_data)
        probabilities = model_pipeline.predict_proba(processed_data)
        
        # For demo model, add some rule-based adjustments to make it more realistic
        if hasattr(model_pipeline, 'named_steps') and 'classifier' in str(type(model_pipeline)):
            # This is likely our demo model, add some manual fraud detection rules
            for i in range(len(processed_data)):
                # Rule 1: Very high amounts are more likely to be fraud
                if processed_data.iloc[i]['Amount'] > 1000:
                    probabilities[i][1] = min(1.0, probabilities[i][1] + 0.3)
                    probabilities[i][0] = 1.0 - probabilities[i][1]
                
                # Rule 2: Late night transactions (after 11 PM or before 5 AM)
                time_val = processed_data.iloc[i]['Time']
                if time_val > 1380 or time_val < 300:  # After 23:00 or before 05:00
                    probabilities[i][1] = min(1.0, probabilities[i][1] + 0.2)
                    probabilities[i][0] = 1.0 - probabilities[i][1]
                
                # Rule 3: Extreme V feature values
                v_features = [processed_data.iloc[i][f'V{j}'] for j in range(1, 29)]
                if any(abs(v) > 3 for v in v_features):
                    probabilities[i][1] = min(1.0, probabilities[i][1] + 0.15)
                    probabilities[i][0] = 1.0 - probabilities[i][1]
                
                # Update predictions based on probability
                if probabilities[i][1] > 0.5:
                    predictions[i] = 1
                else:
                    predictions[i] = 0
        
        return {
            "predictions": predictions.tolist(),
            "probabilities": probabilities.tolist(),
            "prediction_labels": ["Legitimate" if pred == 0 else "Fraudulent" for pred in predictions],
            "fraud_probabilities": [prob[1] for prob in probabilities],
            "num_samples": len(predictions)
        }
    except Exception as e:
        st.error(f"Error during prediction: {str(e)}")
        return None

def predict_single_transaction(model_pipeline, transaction_dict: dict):
    """
    Convenience function to predict on a single transaction.
    """
    transaction_df = pd.DataFrame([transaction_dict])
    result = make_prediction(model_pipeline, transaction_df)
    
    if result:
        return {
            "prediction": result["predictions"][0],
            "prediction_label": result["prediction_labels"][0],
            "fraud_probability": result["fraud_probabilities"][0],
            "confidence": max(result["probabilities"][0])
        }
    return None

# Load model at startup
model_pipeline, model_error = load_model()

# Header
st.markdown('<div class="main-header">💳 Credit Card Fraud Detection</div>', unsafe_allow_html=True)
st.markdown("**Detect fraudulent transactions using advanced machine learning**")

# Model status indicator
if model_pipeline is None:
    st.error(f"❌ Model Loading Error: {model_error}")
    st.stop()
elif model_error and "demo model" in model_error.lower():
    st.warning(f"⚠️ {model_error}")
    st.info("🔬 **Demo Mode**: The app is running with a demonstration model. Predictions are for testing purposes only.")
else:
    st.success("✅ Production model loaded successfully!")

# Sidebar for navigation
st.sidebar.title("🧭 Navigation")
app_mode = st.sidebar.selectbox("Choose Mode", 
    ["🏠 Home", "📊 Batch Prediction", "🔍 Single Transaction", "📊 Model Info"])

# Model status in sidebar
if model_pipeline is None:
    st.sidebar.error("❌ Model Failed")
elif model_error and "demo model" in model_error.lower():
    st.sidebar.warning("🔬 Demo Mode")
else:
    st.sidebar.success("✅ Model Ready")

# App info in sidebar
st.sidebar.markdown("---")
st.sidebar.markdown("### ℹ️ App Info")
st.sidebar.markdown("**Version:** 1.0.0")
st.sidebar.markdown("**ML Model:** Random Forest")
st.sidebar.markdown("**Features:** Real-time + Batch")

if app_mode == "🏠 Home":
    st.markdown("---")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### 📊 Batch Prediction")
        st.markdown("Upload CSV, JSON, or Excel files with multiple transactions for batch analysis.")
        
    with col2:
        st.markdown("### 🔍 Single Transaction")
        st.markdown("Analyze individual transactions in real-time with detailed risk assessment.")
        
    with col3:
        st.markdown("### 📊 Model Info")
        st.markdown("View model architecture, features, and performance metrics.")
    
    st.markdown("---")
    st.markdown("### 🚀 Features")
    st.markdown("""
    - **Real-time Fraud Detection**: Instant analysis of transaction patterns
    - **Batch Processing**: Handle multiple transactions simultaneously
    - **Confidence Scoring**: Get probability scores for risk assessment
    - **Multiple File Formats**: Support for CSV, JSON, and Excel files
    - **Interactive Visualizations**: Charts and graphs for better insights
    - **Integrated ML Pipeline**: No external API dependencies
    """)

elif app_mode == "📊 Batch Prediction":
    st.header("📊 Batch Prediction")
    st.markdown("Upload a file containing multiple transactions for batch fraud detection.")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        file_type = st.selectbox("Select file type:", ["CSV", "JSON", "Excel"])
        
        if file_type == "CSV":
            uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
        elif file_type == "JSON":
            uploaded_file = st.file_uploader("Choose a JSON file", type="json")
        else:
            uploaded_file = st.file_uploader("Choose an Excel file", type=["xls", "xlsx"])
    
    with col2:
        st.markdown("### 📋 Expected Columns")
        st.markdown("""
        - **Time**: Transaction time (HH:MM)
        - **Amount**: Transaction amount
        - **V1-V28**: PCA features
        - **Additional**: Any other features
        """)

    if uploaded_file is not None:
        st.write(f"📁 **File:** {uploaded_file.name}")

        try:
            # Parse file based on type
            if file_type == "CSV":
                df_input = pd.read_csv(uploaded_file)
            elif file_type == "JSON":
                data = json.load(uploaded_file)
                df_input = pd.DataFrame(data)
            else:
                df_input = pd.read_excel(uploaded_file)
            
            st.write(f"📊 **Data shape:** {df_input.shape}")
            
            with st.spinner('🔄 Processing transactions...'):
                prediction_result = make_prediction(model_pipeline, df_input)
            
            if prediction_result:
                st.success("✅ Batch prediction completed!")
                
                # Display summary statistics
                col1, col2, col3, col4 = st.columns(4)
                
                num_samples = prediction_result["num_samples"]
                predictions = prediction_result["predictions"]
                fraud_count = sum(predictions)
                legitimate_count = num_samples - fraud_count
                
                with col1:
                    st.metric("Total Transactions", num_samples)
                with col2:
                    st.metric("Legitimate", legitimate_count, delta=f"{legitimate_count/num_samples*100:.1f}%")
                with col3:
                    st.metric("Fraudulent", fraud_count, delta=f"{fraud_count/num_samples*100:.1f}%")
                with col4:
                    avg_fraud_prob = sum(prediction_result["fraud_probabilities"]) / num_samples if num_samples > 0 else 0
                    st.metric("Avg Risk Score", f"{avg_fraud_prob:.3f}")

                # Visualization
                if num_samples > 0:
                    fig_pie = px.pie(
                        values=[legitimate_count, fraud_count],
                        names=['Legitimate', 'Fraudulent'],
                        title="Transaction Distribution",
                        color_discrete_map={'Legitimate': '#4CAF50', 'Fraudulent': '#F44336'}
                    )
                    st.plotly_chart(fig_pie, use_container_width=True)
                
                # Display results table
                df_results = df_input.copy()
                df_results["Prediction"] = prediction_result["prediction_labels"]
                df_results["Risk_Score"] = prediction_result["fraud_probabilities"]
                df_results["Confidence"] = [max(prob) for prob in prediction_result["probabilities"]]
                
                st.markdown("### 📋 Detailed Results")
                st.dataframe(df_results, use_container_width=True)
                
                # Download button
                csv_results = df_results.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "📥 Download Results (CSV)", 
                    csv_results, 
                    "fraud_detection_results.csv", 
                    "text/csv"
                )

        except Exception as e:
            st.error(f"❌ Error processing file: {str(e)}")

elif app_mode == "🔍 Single Transaction":
    st.header("🔍 Single Transaction Analysis")
    st.markdown("Analyze a single transaction for fraud detection with detailed insights.")
    
    with st.form("transaction_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            time_input = st.time_input("Transaction Time", value=None)
            amount = st.number_input("Amount ($)", min_value=0.0, value=100.0, step=0.01)
            
            st.markdown("**PCA Features (V1-V14):**")
            v_features_1 = {}
            for i in range(1, 15):
                v_features_1[f'V{i}'] = st.number_input(f"V{i}", value=0.0, step=0.01, key=f"v{i}_1")
        
        with col2:
            st.markdown("**Additional Features (V15-V28):**")
            v_features_2 = {}
            for i in range(15, 29):
                v_features_2[f'V{i}'] = st.number_input(f"V{i}", value=0.0, step=0.01, key=f"v{i}_2")
        
        submitted = st.form_submit_button("🔍 Analyze Transaction", use_container_width=True)
        
        if submitted:
            transaction_data = {
                "Time": time_input.strftime("%H:%M") if time_input else "00:00",
                "Amount": amount,
                **v_features_1,
                **v_features_2
            }
            
            with st.spinner('🔄 Analyzing transaction...'):
                result = predict_single_transaction(model_pipeline, transaction_data)
            
            if result:
                prediction = result["prediction_label"]
                fraud_prob = result["fraud_probability"]
                confidence = result["confidence"]
                
                if prediction == "Fraudulent":
                    st.markdown(f"""
                    <div class="fraud-alert">
                        <h3>🚨 FRAUD DETECTED</h3>
                        <p><strong>Risk Score:</strong> {fraud_prob:.3f}</p>
                        <p><strong>Confidence:</strong> {confidence:.3f}</p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="safe-alert">
                        <h3>✅ LEGITIMATE TRANSACTION</h3>
                        <p><strong>Risk Score:</strong> {fraud_prob:.3f}</p>
                        <p><strong>Confidence:</strong> {confidence:.3f}</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Risk gauge chart
                fig_gauge = go.Figure(go.Indicator(
                    mode = "gauge+number+delta",
                    value = fraud_prob,
                    domain = {'x': [0, 1], 'y': [0, 1]},
                    title = {'text': "Fraud Risk Score"},
                    delta = {'reference': 0.5},
                    gauge = {
                        'axis': {'range': [None, 1]},
                        'bar': {'color': "darkblue"},
                        'steps': [
                            {'range': [0, 0.3], 'color': "lightgreen"},
                            {'range': [0.3, 0.7], 'color': "yellow"},
                            {'range': [0.7, 1], 'color': "red"}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': 0.9
                        }
                    }
                ))
                fig_gauge.update_layout(height=400)
                st.plotly_chart(fig_gauge, use_container_width=True)

elif app_mode == "📊 Model Info":
    st.header("📊 Model Information")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🏗️ Model Architecture")
        st.markdown("""
        **Pipeline Components:**
        1. **Preprocessor**: OneHotEncoder for categorical features
        2. **SMOTE**: Synthetic minority oversampling 
        3. **Classifier**: Random Forest (100 estimators)
        
        **Key Features:**
        - Handles class imbalance with SMOTE
        - Processes categorical and numerical features
        - Provides probability scores for risk assessment
        """)
        
        st.markdown("### 📋 Expected Features")
        st.markdown("""
        - **Time**: Transaction time (converted to minutes)
        - **Amount**: Transaction amount in dollars
        - **V1-V28**: PCA-transformed features
        - **location**: Transaction location (optional)
        - **merchant**: Merchant information (optional)
        """)
    
    with col2:
        st.markdown("### 📊 Model Performance")
        st.markdown("""
        **Evaluation Metrics:**
        - Classification Report (Precision, Recall, F1-Score)
        - Confusion Matrix Analysis
        - Log Loss for probability assessment
        
        **Data Processing:**
        - Stratified train/test split (80/20)
        - SMOTE balancing for training data
        - Time feature engineering
        """)
        
        if st.button("🔄 Reload Model"):
            st.cache_resource.clear()
            st.rerun()

# Footer
st.markdown("---")
st.markdown("**🚀 Integrated Streamlit ML Application** | **Status:** 🟢 Model Loaded")
