from typing import Optional
import pandas as pd
from .data_loader import DataLoader
from .data_handler import DataHandler

class DataManager:

    def __init__(self, ticker: str, start_date: Optional[str] = None, end_date: Optional[str] = None):
        self.loader = DataLoader(ticker=ticker,
                                 start_date=start_date,
                                 end_date=end_date)
        
    def get_ticker_dataframe(self) -> pd.DataFrame:
        maex_data = self.loader.load_moex_data()
        df_ticker = DataHandler.prepeare_data(ticker_data=maex_data)

        return df_ticker
