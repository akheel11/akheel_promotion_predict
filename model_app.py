from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import joblib
import logging
import numpy as np
from typing import List

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

class Input(BaseModel):
    employee_id: int
    department: str
    region: str
    education: str
    gender: str
    recruitment_channel: str
    no_of_trainings: int
    age: int
    previous_year_rating: float
    length_of_service: int
    KPIs_met_gt_80_prct: int
    awards_won: int
    avg_training_score: int

    class Config:
        from_attributes = True

class Output(BaseModel):
    is_promoted: int

@app.get("/")
async def root():
    return {"message": "Employee Promotion Prediction API"}

@app.post("/predict", response_model=Output)
def predict(data: Input) -> Output:
    try:
        # Convert input data to DataFrame
        input_dict = data.dict()
        X_input = pd.DataFrame([input_dict])
        
        logger.info(f"Input data shape: {X_input.shape}")
        logger.info(f"Input data columns: {X_input.columns}")
        
        # Load model
        try:
            model = joblib.load('jobchg_pipeline_model.pkl')
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise HTTPException(status_code=500, detail="Error loading the model")

        # Make prediction
        try:
            prediction = model.predict(X_input)
            prediction_value = int(prediction[0])
            logger.info(f"Prediction successful: {prediction_value}")
        except Exception as e:
            logger.error(f"Error during prediction: {str(e)}")
            raise HTTPException(status_code=500, detail="Error making prediction")

        return Output(is_promoted=prediction_value)

    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Add an exception handler for validation errors
@app.exception_handler(Exception)
async def validation_exception_handler(request, exc):
    return {
        "status_code": 500,
        "detail": str(exc)
    }