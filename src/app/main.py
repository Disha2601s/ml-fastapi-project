from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import Literal
import pandas as pd
from src.app import model as m

app = FastAPI(title='ML Model API')

class PredictRequest(BaseModel):
    feat_0: float = Field(..., example=0.1)
    feat_1: float = Field(..., example=0.0)
    feat_2: float = Field(..., example=0.5)
    feat_3: float = Field(..., example=-0.1)
    feat_4: float = Field(..., example=1.0)
    education: Literal['Primary', 'Secondary', 'Bachelors', 'Masters'] = Field(..., example='Bachelors')

class PredictResponse(BaseModel):
    prediction: int

@app.get('/')
def root():
    return {'message': 'Welcome to the model API!'}

@app.post('/predict', response_model=PredictResponse)
def predict(req: PredictRequest):
    df = pd.DataFrame([req.dict()])
    preds = m.inference(df)
    return {'prediction': int(preds[0])}
