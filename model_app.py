from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import pandas as pd
import joblib
import logging
import numpy as np
from typing import List

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = FastAPI()

# Fix the Input class definition with Field aliases
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
    KPIs_met_gt_80: int = Field(..., alias="KPIs_met >80%")
    awards_won: int = Field(..., alias="awards_won?")
    avg_training_score: int

    class Config:
        from_attributes = True
        populate_by_name = True
        allow_population_by_field_name = True

class Output(BaseModel):
    is_promoted: int

@app.get("/")
async def root():
    return {"message": "Employee Promotion Prediction API"}

@app.post("/predict", response_model=Output)
def predict(data: Input) -> Output:
    try:
        # Convert input data to DataFrame
        input_dict = data.model_dump(by_alias=True)
        X_input = pd.DataFrame([input_dict])
        
        logger.info(f"Input data shape: {X_input.shape}")
        logger.info(f"Input data columns: {X_input.columns}")
        
        # Load model
        try:
            model = joblib.load('jobchg_pipeline_model.pkl')
            logger.info("Model loaded successfully")
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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
