import pandas as pd
from io import StringIO

class DataHandler:
    """
    Класс, который содержит все функции обработки и преобразования данных
    """
    @classmethod
    def prepeare_data(cls, ticker_data: str) -> pd.DataFrame:
        lines = ticker_data.strip().splitlines()

        try:
            header_index = next(i for i, line in enumerate(lines) if 'TRADEDATE' in line)
        except StopIteration:
            raise ValueError("Заголовок с 'TRADEDATE' не найден в данных")

        data_lines = lines[header_index:]

        clean_lines = []
        for line in data_lines:
            line_strip = line.strip().lower()
            if line_strip == '' or 'history.cursor' in line_strip:
                break
            clean_lines.append(line)

        clean_data = '\n'.join(clean_lines)

        df = pd.read_csv(StringIO(clean_data), sep=';')

        if 'id' in df.columns:
            df = df.drop(columns=['id'])

        if 'TRADEDATE' in df.columns:
            df['TRADEDATE'] = pd.to_datetime(df['TRADEDATE'])
        else:
            raise ValueError("Колонка 'TRADEDATE' отсутствует в DataFrame")

        df = df.reset_index(drop=True)

        return df