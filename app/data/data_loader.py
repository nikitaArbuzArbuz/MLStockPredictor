from typing import Optional
from datetime import datetime, timedelta
from requests import get, exceptions

from app.config.settings import Settings

class DataLoader:
    """
    Класс, который содержит все функции выгрузки и проверки выгруженных данных
    """

    def __init__(self, ticker: str, start_date: Optional[str] = None, end_date: Optional[str] = None):
        self.ticker = ticker
        self.start_date = start_date
        self.end_date = end_date
        self._check_dates()

    def _check_dates(self) -> None:
        """
        Функция проверки переданных временных границ, в случае их отсутсвия заменяются текущей датой и час назад от текущей
        """
        date_format = "%Y-%m-%dT%H:%M:%S"
        timeframe = 1

        if self.start_date is None:
            self.start_date = (datetime.now() - timedelta(hours=timeframe)).strftime(date_format)
        if self.end_date is None:
            self.end_date = datetime.now().strftime(date_format)

    def load_moex_data(self) -> Optional[str]:
        params = {"from": self.start_date, 
                  "till": self.end_date}
        settings = Settings()
        headers = {
            "Accept": "text/csv"
        }
        try:
            response = get(
                settings.MOEX_URL.format(ticker=self.ticker),
                params=params,
                headers=headers,
                timeout=10
            )
        except exceptions.ConnectTimeout:
                print("[ERROR] Истекло время ожидания подключения к MOEX")
                raise
        try:
            response.raise_for_status()
        except exceptions.HTTPError as e: 
            print(f"HTTP ошибка произошла: {e}")
            raise exceptions.HTTPError
        else:
            return response.text
            


