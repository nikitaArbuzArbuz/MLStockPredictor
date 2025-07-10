# from prophet import Prophet
# import mlflow
# import mlflow.pyfunc
# import pandas as pd
# import os
# from datetime import datetime, timedelta
# from app.data.data_manager import DataManager  # Replace with your actual import path
# import joblib  # Added import


# class ProphetWrapper(mlflow.pyfunc.PythonModel):
#     def load_context(self, context):
#         import joblib
#         self.model = joblib.load(context.artifacts["model_path"])

#     def predict(self, context, model_input):
#         future = self.model.make_future_dataframe(periods=1)
#         return self.model.predict(future)


# def fetch_and_save_data(ticker: str, filename: str):
#     """Загружает исторические данные с MOEX и сохраняет в CSV"""
#     end_date = datetime.now().strftime("%Y-%m-%d")
#     start_date = (datetime.now() - timedelta(days=180)).strftime("%Y-%m-%d")

#     print(f"[INFO] Загружаем данные по {ticker} с {start_date} по {end_date}...")

#     data_manager = DataManager(ticker=ticker, start_date=start_date, end_date=end_date)
#     df = data_manager.get_ticker_dataframe()
#     df.to_csv(filename, index=False)

#     print(f"[INFO] Данные сохранены в {filename}")
#     return df


# def train_and_log():
#     ticker = "SBER"
#     data_path = "data/sber_history.csv"
#     os.makedirs("data", exist_ok=True)

#     # 1. Загрузка данных
#     df = fetch_and_save_data(ticker=ticker, filename=data_path)
#     df = df[["TRADEDATE", "CLOSE"]].rename(columns={"TRADEDATE": "ds", "CLOSE": "y"})
#     df = df.dropna()

#     # 2. Настройка MLflow
#     mlflow.set_tracking_uri("http://localhost:5000")  # Or your remote tracking server
#     mlflow.set_experiment("StockForecasting")

#     # 3. Обучение и логирование
#     with mlflow.start_run():
#         # Создаем и обучаем модель
#         model = Prophet()
#         model.fit(df)

#         # Ручное логирование параметров
#         mlflow.log_params({
#             "growth": model.growth,
#             "seasonality_mode": model.seasonality_mode,
#             "changepoint_prior_scale": model.changepoint_prior_scale
#         })

#         # Сохранение модели
#         model_path = "prophet_model"
#         os.makedirs(model_path, exist_ok=True)

#         # Use joblib to save the model
#         model_filename = os.path.join(model_path, "model.joblib")
#         joblib.dump(model, model_filename)  # Save the Prophet model with joblib



#         # Логирование модели
#         mlflow.pyfunc.log_model(
#             artifact_path="model",
#             python_model=ProphetWrapper(),
#             artifacts={"model_path": model_filename},
#             registered_model_name="StockModel"
#         )

#         # Добавление тегов
#         mlflow.set_tag("model_type", "Prophet")
#         mlflow.set_tag("ticker", ticker)

#         print(f"[INFO] Модель зарегистрирована: {mlflow.active_run().info.run_id}")

#     # 4. Переводим модель в Production
#     client = mlflow.tracking.MlflowClient()
#     latest_version = client.get_latest_versions("StockModel", stages=["None"])[0].version
#     client.transition_model_version_stage(
#         name="StockModel",
#         version=latest_version,
#         stage="Production"
#     )
#     print("[INFO] Модель переведена в Production")


# if __name__ == "__main__":
#     train_and_log()



import pandas as pd
import numpy as np
import mlflow
import mlflow.pyfunc
import pickle
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import logging
import os
from io import StringIO  # Import StringIO
import requests  # Import requests
from prophet import Prophet  # Import Prophet
from prophet.diagnostics import cross_validation, performance_metrics

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

MOEX_URL = "https://iss.moex.com/iss/history/engines/stock/markets/shares/boards/TQBR/securities/SBER.csv?iss.a"
MODEL_NAME = "stock_prophet"
MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")
os.environ["MLFLOW_S3_ENDPOINT_URL"] = os.environ.get("MLFLOW_S3_ENDPOINT_URL", "http://localhost:9000")
os.environ["AWS_ACCESS_KEY_ID"] = os.environ.get("AWS_ACCESS_KEY_ID", "minio")
os.environ["AWS_SECRET_ACCESS_KEY"] = os.environ.get("AWS_SECRET_ACCESS_KEY", "minio123")


class StockPredictor(mlflow.pyfunc.PythonModel):
    def __init__(self, model_path):
        self.model_path = model_path
        self.model = None

    def load_context(self, context):
        logging.info("Loading model from artifacts...")
        self.model = mlflow.pyfunc.load_model(context.artifacts[self.model_path])

    def predict(self, context, model_input):
        logging.info("Starting prediction...")
        try:
            if 'ds' not in model_input.columns:
                raise ValueError("Column 'ds' (datetime) not found in the input data.")

            predictions = self.model.predict(model_input)
            logging.info("Prediction complete.")
            return predictions[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]

        except Exception as e:
            logging.error(f"Error during prediction: {e}")
            raise


class DataHandler:
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

        df = df.rename(columns={'TRADEDATE': 'ds', 'CLOSE': 'y'})
        df = df[['ds', 'y']] 
        df = df.reset_index(drop=True)

        return df


def train_prophet(data):
    logging.info("Starting Prophet model training...")
    try:
        model = Prophet(daily_seasonality=False, weekly_seasonality=True, yearly_seasonality=True) 
        model.fit(data)
        logging.info("Prophet model training complete.")
        return model
    except Exception as e:
        logging.error(f"Error during Prophet training: {e}")
        raise


def main():
    logging.info("Starting main function...")

    logging.info(f"Loading data from MOEX URL: {MOEX_URL}...")
    try:
        response = requests.get(MOEX_URL)
        response.raise_for_status()
        moex_data = response.text
        df = DataHandler.prepeare_data(moex_data)
        logging.info("Data loaded and prepared successfully.")
    except requests.exceptions.RequestException as e:
        logging.error(f"Error fetching data from MOEX: {e}")
        return
    except ValueError as e:
        logging.error(f"Error preparing data: {e}")
        return

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment("stock_prediction")

    with mlflow.start_run() as run:
        logging.info(f"MLflow run started with run_id: {run.info.run_id}")

        logging.info("Training model...")
        model = train_prophet(df)

        mlflow.log_param("moex_url", MOEX_URL)
        mlflow.log_param("model_type", "Prophet")

        logging.info("Logging model to MLflow...")
        model_path = "prophet_model"
        mlflow.pyfunc.log_model(
            python_model=StockPredictor(model_path=model_path),
            artifact_path=model_path,
            code_paths=["."] 
        )

        logging.info("Registering model to MLflow...")
        model_uri = f"runs:/{run.info.run_id}/{model_path}"
        try:
            mv = mlflow.register_model(model_uri, MODEL_NAME)
            logging.info(f"Model registered successfully. Name: {mv.name}, Version: {mv.version}")
            model_version = mv.version  
        except Exception as e:
            logging.error(f"Error registering model: {e}")
            return

        logging.info("Verifying model logging and deploying...")
        try:
            loaded_model = mlflow.pyfunc.load_model(model_uri)
            logging.info("Model loaded successfully from MLflow.")
            future = loaded_model.predict(pd.DataFrame({'ds': pd.to_datetime(['2024-01-01', '2024-01-02'])}))
            print(future[['ds', 'yhat', 'yhat_lower', 'yhat_upper']])


            import subprocess
            deploy_command = [
                "mlflow", "models", "serve",
                "-m", f"models:/{MODEL_NAME}/{model_version}",
                "-p", "5001",
                "--host", "0.0.0.0"
            ]
            logging.info(f"Deploying model with command: {' '.join(deploy_command)}")
            subprocess.Popen(deploy_command)
            logging.info("Model deployment started in background.")


        except Exception as e:
            logging.error(f"Error verifying or deploying model: {e}")
            return

    logging.info("Main function complete.")

if __name__ == "__main__":
    main()
