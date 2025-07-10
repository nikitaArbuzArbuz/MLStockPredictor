from fastapi import FastAPI

from app.api.handlers import router


app = FastAPI(
    title="MLStockPredictor",
    access_log=True
)

app.include_router(router)