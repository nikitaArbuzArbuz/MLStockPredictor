from prophet import Prophet
import mlflow
import mlflow.pyfunc
import os
from dotenv import load_dotenv

load_dotenv()

class ProphetModel(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        import joblib
        self.model = joblib.load(context.artifacts['model'])

    def predict(self, context, model_input):
        return self.model.predict(model_input)

def train_model(df):
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
    mlflow.set_experiment("stock-forecast")

    with mlflow.start_run() as run:
        model = Prophet()
        model.fit(df)

        import joblib, tempfile
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "model.pkl")
            joblib.dump(model, path)
            mlflow.register_model(f"runs:/{run.info.run_id}/model", "prophet_stock_model")