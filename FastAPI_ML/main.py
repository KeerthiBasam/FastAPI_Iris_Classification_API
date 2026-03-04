from fastapi import FastAPI
from pydantic import BaseModel, Field
import numpy as np
import pickle

# Load model once when the app starts (best practice)
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

app = FastAPI(title="Iris Classifier API")

# Request schema
class IrisRequest(BaseModel):
    sepal_length: float = Field(..., example=5.1)
    sepal_width: float = Field(..., example=3.5)
    petal_length: float = Field(..., example=1.4)
    petal_width: float = Field(..., example=0.2)

# Response schema (optional but nice)
class IrisResponse(BaseModel):
    prediction_index: int
    prediction_label: str
    probabilities: list[float]

target_names = ["setosa", "versicolor", "virginica"]

@app.get("/")
def home():
    return {"message": "API is running. Use POST /predict"}

@app.post("/predict", response_model=IrisResponse)
def predict(req: IrisRequest):
    X = np.array([[req.sepal_length, req.sepal_width, req.petal_length, req.petal_width]])

    pred = int(model.predict(X)[0])
    probs = model.predict_proba(X)[0].tolist()

    return IrisResponse(
        prediction_index=pred,
        prediction_label=target_names[pred],
        probabilities=probs
    )