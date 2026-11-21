"""FastAPI backend that serves the trained flight-delay model.

Run from the repo root:  uvicorn api.main:app --reload
Docs (auto-generated):   http://127.0.0.1:8000/docs

The model is loaded once at startup from pipeline/artifacts/model.joblib (override with the
MODEL_PATH environment variable). Callers send raw flight details; the same feature
engineering used for training (preprocessing/preprocess.py: engineer_features) is applied
before predicting, so there is no train/serve skew.
"""
import os
import sys
from contextlib import asynccontextmanager
from datetime import date
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field, field_validator, model_validator

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from preprocessing.preprocess import engineer_features  # noqa: E402

MODEL_PATH = Path(os.environ.get("MODEL_PATH", ROOT / "pipeline" / "artifacts" / "model.joblib"))


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the model, reference data and feature list once at startup."""
    if not MODEL_PATH.exists():
        raise RuntimeError(f"model not found at {MODEL_PATH}; run `python pipeline/pipeline.py` first")
    cfg = yaml.safe_load((ROOT / "pipeline" / "config.yaml").read_text())["features"]
    app.state.model = joblib.load(MODEL_PATH)
    app.state.columns = cfg["onehot"] + cfg["target_encode"] + cfg["numeric"]
    app.state.airports = pd.read_csv(ROOT / "data" / "airports.csv")
    app.state.airlines = pd.read_csv(ROOT / "data" / "airlines.csv").set_index("airline_code")["airline"].to_dict()
    yield


app = FastAPI(title="Airline Delay Prediction API", version="1.0.0", lifespan=lifespan)


class FlightRequest(BaseModel):
    airline_code: str = Field(description="Two-character IATA airline code, e.g. 'DL'", examples=["DL"])
    flight_date: date = Field(description="Scheduled date of the flight (YYYY-MM-DD)", examples=["2026-07-04"])
    origin: str = Field(description="Origin airport IATA code", examples=["ATL"])
    dest: str = Field(description="Destination airport IATA code", examples=["JFK"])
    sched_dep_time: int = Field(ge=0, le=2359, description="Scheduled departure time as local HHMM, e.g. 1745", examples=[1745])
    distance: float | None = Field(default=None, gt=0, description="Route distance in miles; estimated from airport coordinates if omitted")

    @field_validator("airline_code", "origin", "dest")
    @classmethod
    def uppercase(cls, v: str) -> str:
        return v.strip().upper()

    @field_validator("sched_dep_time")
    @classmethod
    def valid_clock_time(cls, v: int) -> int:
        if v % 100 > 59:
            raise ValueError("sched_dep_time must be HHMM with minutes 00-59")
        return v

    @model_validator(mode="after")
    def different_airports(self):
        if self.origin == self.dest:
            raise ValueError("origin and dest must be different airports")
        return self


class PredictionResponse(BaseModel):
    predicted_delay_minutes: float = Field(description="Expected delay in minutes (worse of departure/arrival delay, floored at 0)")
    predicted_log_delay_minutes: float = Field(description="Raw model output: log1p(delay minutes)")
    airline: str


@app.get("/health")
def health(request: Request) -> dict:
    return {"status": "ok", "model_loaded": request.app.state.model is not None}


@app.post("/predict", response_model=PredictionResponse)
def predict(flight: FlightRequest, request: Request) -> PredictionResponse:
    state = request.app.state
    if flight.airline_code not in state.airlines:
        raise HTTPException(422, f"unknown airline_code '{flight.airline_code}'; known: {sorted(state.airlines)}")
    known = set(state.airports["iata_code"])
    for field, code in (("origin", flight.origin), ("dest", flight.dest)):
        if code not in known:
            raise HTTPException(422, f"unknown {field} airport '{code}'")

    row = {
        "flight_date": pd.Timestamp(flight.flight_date),
        "airline_code": flight.airline_code,
        "origin": flight.origin,
        "dest": flight.dest,
        "sched_dep_time": flight.sched_dep_time,
    }
    if flight.distance is not None:
        row["distance"] = flight.distance
    features = engineer_features(pd.DataFrame([row]), state.airports)

    log_pred = float(state.model.predict(features[state.columns])[0])
    minutes = float(max(np.expm1(log_pred), 0.0))
    return PredictionResponse(
        predicted_delay_minutes=round(minutes, 1),
        predicted_log_delay_minutes=round(log_pred, 4),
        airline=state.airlines[flight.airline_code],
    )
