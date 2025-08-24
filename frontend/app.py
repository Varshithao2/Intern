import streamlit as st
import requests
import pandas as pd
import json
import io
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(
    page_title="Credit Card Fraud Detection", 
    page_icon="💳", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# API Configuration
API_URL = "http://127.0.0.1:8000"

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
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<div class="main-header">💳 Credit Card Fraud Detection</div>', unsafe_allow_html=True)
st.markdown("**Detect fraudulent transactions using advanced machine learning**")

# Sidebar for navigation
st.sidebar.title("🧭 Navigation")
app_mode = st.sidebar.selectbox("Choose Mode", 
    ["🏠 Home", "📊 Batch Prediction", "🔍 Single Transaction", "🏥 Health Check"])

# Health check function
def check_api_health():
    try:
        response = requests.get(f"{API_URL}/health", timeout=5)
        return response.status_code == 200, response.json() if response.status_code == 200 else None
    except:
        return False, None

# Display API status in sidebar
health_status, health_data = check_api_health()
if health_status:
    st.sidebar.success("✅ API Online")
else:
    st.sidebar.error("❌ API Offline")

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
        st.markdown("### 🏥 Health Check")
        st.markdown("Monitor API status and system health metrics.")
    
    st.markdown("---")
    st.markdown("### 🚀 Features")
    st.markdown("""
    - **Real-time Fraud Detection**: Instant analysis of transaction patterns
    - **Batch Processing**: Handle multiple transactions simultaneously
    - **Confidence Scoring**: Get probability scores for risk assessment
    - **Multiple File Formats**: Support for CSV, JSON, and Excel files
    - **Interactive Visualizations**: Charts and graphs for better insights
    """)

elif app_mode == "📊 Batch Prediction":
    st.header("📊 Batch Prediction")
    st.markdown("Upload a file containing multiple transactions for batch fraud detection.")
    
    # File upload section
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
            # Prepare file content
            file_content = uploaded_file.getvalue()
            
            if file_type == "CSV":
                mime_type = 'text/csv'
            elif file_type == "JSON":
                mime_type = 'application/json'
            else:
                mime_type = 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'

            # Send to API
            files = {'file': (uploaded_file.name, file_content, mime_type)}
            
            with st.spinner('🔄 Processing your file...'):
                response = requests.post(f"{API_URL}/predict", files=files, timeout=120)

            if response.status_code == 200:
                resp_json = response.json()
                prediction_data = resp_json.get("prediction", {})
                
                st.success("✅ Batch prediction completed!")
                
                # Display summary statistics
                col1, col2, col3, col4 = st.columns(4)
                
                num_samples = prediction_data.get("num_samples", 0)
                predictions = prediction_data.get("predictions", [])
                fraud_count = sum(predictions)
                legitimate_count = num_samples - fraud_count
                
                with col1:
                    st.metric("Total Transactions", num_samples)
                with col2:
                    st.metric("Legitimate", legitimate_count, delta=f"{legitimate_count/num_samples*100:.1f}%")
                with col3:
                    st.metric("Fraudulent", fraud_count, delta=f"{fraud_count/num_samples*100:.1f}%")
                with col4:
                    avg_fraud_prob = sum(prediction_data.get("fraud_probabilities", [])) / num_samples if num_samples > 0 else 0
                    st.metric("Avg Risk Score", f"{avg_fraud_prob:.3f}")

                # Visualization
                if num_samples > 0:
                    # Pie chart
                    fig_pie = px.pie(
                        values=[legitimate_count, fraud_count],
                        names=['Legitimate', 'Fraudulent'],
                        title="Transaction Distribution",
                        color_discrete_map={'Legitimate': '#4CAF50', 'Fraudulent': '#F44336'}
                    )
                    st.plotly_chart(fig_pie, use_container_width=True)
                
                # Display results table
                if file_type == "CSV":
                    df_input = pd.read_csv(io.StringIO(file_content.decode('utf-8')))
                elif file_type == "JSON":
                    data = json.loads(file_content)
                    df_input = pd.DataFrame(data)
                else:
                    df_input = pd.read_excel(io.BytesIO(file_content))
                
                # Create results dataframe
                df_results = df_input.copy()
                df_results["Prediction"] = ["Fraudulent" if p == 1 else "Legitimate" for p in predictions]
                df_results["Risk_Score"] = prediction_data.get("fraud_probabilities", [])
                df_results["Confidence"] = [max(prob) for prob in prediction_data.get("probabilities", [])]
                
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
                
            else:
                st.error(f"❌ API Error ({response.status_code}): {response.text}")

        except Exception as e:
            st.error(f"❌ Error processing file: {str(e)}")

elif app_mode == "🔍 Single Transaction":
    st.header("🔍 Single Transaction Analysis")
    st.markdown("Analyze a single transaction for fraud detection with detailed insights.")
    
    # Input form
    with st.form("transaction_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            time_input = st.time_input("Transaction Time", value=None)
            amount = st.number_input("Amount ($)", min_value=0.0, value=100.0, step=0.01)
            
            # V1-V14 features
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
            # Prepare transaction data
            transaction_data = {
                "Time": time_input.strftime("%H:%M") if time_input else "00:00",
                "Amount": amount,
                **v_features_1,
                **v_features_2
            }
            
            try:
                with st.spinner('🔄 Analyzing transaction...'):
                    response = requests.post(
                        f"{API_URL}/predict/single",
                        json={"transaction": transaction_data},
                        timeout=30
                    )
                
                if response.status_code == 200:
                    result = response.json()["result"]
                    
                    # Display result
                    prediction = result["prediction_label"]
                    fraud_prob = result["fraud_probability"]
                    confidence = result["confidence"]
                    
                    if prediction == "Fraud":
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
                    
                else:
                    st.error(f"❌ API Error ({response.status_code}): {response.text}")
                    
            except Exception as e:
                st.error(f"❌ Error analyzing transaction: {str(e)}")

elif app_mode == "🏥 Health Check":
    st.header("🏥 System Health Check")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### API Status")
        if health_status:
            st.success("✅ API is online and responding")
            if health_data:
                st.json(health_data)
        else:
            st.error("❌ API is offline or not responding")
    
    with col2:
        st.markdown("### Connection Details")
        st.markdown(f"**API URL:** `{API_URL}`")
        
        if st.button("🔄 Refresh Status"):
            st.rerun()

# Footer
st.markdown("---")
st.markdown(f"**API Endpoint:** {API_URL} | **Status:** {'🟢 Online' if health_status else '🔴 Offline'}")