from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import numpy as np
from keras.models import load_model

app = FastAPI()

# Load the Keras model
model = load_model("best_model.keras")

class PredictionInput(BaseModel):
    utc_timestamp: float
    acc_x: float
    acc_y: float
    acc_z: float
    acc_mag: float
    eda: float
    hr: float
    temp: float

class PredictionOutput(BaseModel):
    prediction: int

@app.post("/predict", response_model=PredictionOutput)
async def predict(data: PredictionInput):
    # Convert input data to numpy array
    input_data = np.array([[data.utc_timestamp, data.acc_x, data.acc_y, data.acc_z, data.acc_mag, data.eda, data.hr, data.temp]])

    # Make prediction using the loaded Keras model
    prediction = model.predict(input_data)[0][0]

    # Apply threshold and return the result
    result = 1 if prediction > 0.5 else 0

    return {"prediction": result}
