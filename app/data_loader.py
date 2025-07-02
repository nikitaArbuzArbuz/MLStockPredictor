import pandas as pd
from datetime import datetime

def load_moex_data(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    url = f"https://iss.moex.com/iss/history/engines/stock/markets/shares/securities/{ticker}.csv"
    params = {"from": start_date, "till": end_date}
    response = requests.get(url, params=params)
    response.raise_for_status()
    df = pd.read_csv(StringIO(response.text))
    df['TRADEDATE'] = pd.to_datetime(df['TRADEDATE'])
    return df[['TRADEDATE', 'CLOSE']]