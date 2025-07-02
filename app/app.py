from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import app.data_loader
import train_model

app = FastAPI()

class PredictRequest(BaseModel):
    ticker: str

@app.get("/test")
def test():
    return {"status": "OK"}

@app.post("/predict")
async def get_prediction(request: PredictRequest):
    try:
        data = data_loader.load_moex_data(request.ticker, request.start_date, request.end_date)
        train_model(data)

    except Exception as e:
        raise HTTPException(status_code=500, detail="Internal server error")