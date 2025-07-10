from fastapi import APIRouter, HTTPException
from app.data.data_manager import DataManager
from app.schemas.schemas import PredictRequest
from app.models.models_handler import ModelsHandler
from app.train_model import StockPredictor
import traceback

router = APIRouter()



@router.get("/test")
def test():
    return {"status": "OK"}

@router.post("/predict")
async def get_prediction(request: PredictRequest):
    try:
        data_manager = DataManager(ticker=request.ticker,
                                   start_date=request.start_date,
                                   end_date=request.end_date)

        data = data_manager.get_ticker_dataframe()

        model_handler = ModelsHandler()
        prediction = model_handler.predict(data)

        return {"ticker": request.ticker, "direction": prediction}

    except Exception as e:
        detail = f"{str(e)}\n{traceback.format_exc()}"
        raise HTTPException(status_code=500, detail=detail)
