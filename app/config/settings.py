from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    MOEX_URL: str = "https://iss.moex.com/iss/history/engines/stock/markets/shares/securities/{ticker}.csv"