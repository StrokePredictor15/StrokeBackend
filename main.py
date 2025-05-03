from fastapi import FastAPI
from api import Predict, auth

app = FastAPI()

app.include_router(auth.router, prefix="/auth")
app.include_router(Predict.router, prefix="/model")

@app.get("/")
def home():
    return {"message": "Stroke Prediction API running"}
