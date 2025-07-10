from pydantic import BaseModel
from typing import Optional

class PredictRequest(BaseModel):
    """
    Класс описывающий схему входных данных
    """
    ticker: str
    start_date: Optional[str] = None
    end_date: Optional[str] = None