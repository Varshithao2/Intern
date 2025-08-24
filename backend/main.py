from fastapi import FastAPI, File, UploadFile, HTTPException
import pandas as pd
import json
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
import logging
from pydantic import BaseModel
from typing import Dict, Any, List
import io
import numpy as np

from backend.inference import make_prediction, predict_single_transaction

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Credit Card Fraud Detection API", version="1.0.0")
logger.info("Starting FastAPI app...")

class PredictionResponse(BaseModel):
    message: str
    prediction: Dict[str, Any]

class SingleTransactionRequest(BaseModel):
    transaction: Dict[str, Any]

class SingleTransactionResponse(BaseModel):
    message: str
    result: Dict[str, Any]

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/", response_class=HTMLResponse)
async def read_root():
    logger.info("Root endpoint '/' called")
    return """
    <h1>Welcome to the Credit Card Fraud Detection API</h1>
    <h2>Available Endpoints:</h2>
    <ul>
        <li><strong>POST /predict</strong> - Upload a file (CSV, JSON, Excel) with transaction data for batch prediction</li>
        <li><strong>POST /predict/single</strong> - Send a single transaction for prediction</li>
        <li><strong>GET /health</strong> - Check API health status</li>
    </ul>
    """

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    logger.info("Health check endpoint called")
    return {"status": "healthy", "message": "Credit Card Fraud Detection API is running"}

@app.post("/predict", response_model=PredictionResponse)
async def predict_batch(file: UploadFile = File(...)):
    """
    Batch prediction endpoint for file uploads.
    Accepts CSV, JSON, or Excel files with transaction data.
    """
    logger.info(f"'/predict' endpoint called with file: {file.filename}")
    filename = file.filename.lower()

    try:
        contents = await file.read()
        logger.info(f"File {file.filename} read successfully, size: {len(contents)} bytes")

        # Parse file based on extension
        if filename.endswith(".csv"):
            df = pd.read_csv(io.BytesIO(contents))
            logger.info("CSV file parsed to DataFrame")
        elif filename.endswith(".json"):
            data = json.loads(contents)
            df = pd.DataFrame(data)
            logger.info("JSON file parsed to DataFrame")
        elif filename.endswith((".xls", ".xlsx")):
            df = pd.read_excel(io.BytesIO(contents))
            logger.info("Excel file parsed to DataFrame")
        else:
            logger.warning("Unsupported file type")
            raise HTTPException(status_code=400, detail="Unsupported file type. Please use CSV, JSON, or Excel files.")

        logger.info(f"DataFrame shape: {df.shape}")
        
        # Make prediction using the inference module (preprocessing is handled internally)
        prediction = make_prediction(df)
        logger.info("Batch prediction completed successfully")

        return PredictionResponse(
            message=f"File processed successfully. Predicted {prediction['num_samples']} transactions.", 
            prediction=prediction
        )

    except Exception as e:
        logger.error(f"Error processing file: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")

@app.post("/predict/single", response_model=SingleTransactionResponse)
async def predict_single(request: SingleTransactionRequest):
    """
    Single transaction prediction endpoint.
    Accepts a JSON object with transaction features.
    """
    logger.info("'/predict/single' endpoint called")

    try:
        # Make prediction for single transaction
        result = predict_single_transaction(request.transaction)
        logger.info("Single transaction prediction completed successfully")

        return SingleTransactionResponse(
            message="Single transaction processed successfully",
            result=result
        )

    except Exception as e:
        logger.error(f"Error processing single transaction: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing transaction: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)