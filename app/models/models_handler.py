import mlflow.pyfunc
import pandas as pd

class ModelsHandler:
    """
    Класс для работы с развернутыми моделями в MLFlow
    """

    def __init__(self, model_uri: str = "models:/stock_prophet/latest"):
        self.model = mlflow.pyfunc.load_model(model_uri)

    def predict(self, df: pd.DataFrame) -> str:
        df = df.rename(columns={"TRADEDATE": "ds", "CLOSE": "y"})
        
        future = self.model.make_future_dataframe(periods=1)
        forecast = self.model.predict(future)

        last_close = df['y'].iloc[-1]
        next_price = forecast['yhat'].iloc[-1]

        if abs(next_price - last_close) < 0.005 * last_close:
            return "NEUTRAL"
        elif next_price > last_close:
            return "UP"
        else:
            return "DOWN"