"""Streamlit front end for the airline delay prediction API.

Run locally from the repo root:  streamlit run frontend/app.py
Deploy on Streamlit Community Cloud with main file path `frontend/app.py`.

The API URL defaults to the Cloud Run service; override it with an `API_URL` entry in
Streamlit secrets (.streamlit/secrets.toml locally, or the app's Secrets settings in the cloud).
"""
from datetime import datetime
from pathlib import Path

import pandas as pd
import requests
import streamlit as st

DEFAULT_API_URL = "https://airline-delay-api-170279949235.us-east1.run.app"
HERE = Path(__file__).parent


def api_url() -> str:
    try:
        return st.secrets["API_URL"].rstrip("/")
    except (KeyError, FileNotFoundError):
        return DEFAULT_API_URL


@st.cache_data
def load_options() -> tuple[dict[str, str], dict[str, str]]:
    airports = pd.read_csv(HERE / "airports.csv")
    airlines = pd.read_csv(HERE / "airlines.csv")
    airport_labels = {r.iata_code: f"{r.iata_code} - {r.airport_name} ({r.city})" for r in airports.itertuples()}
    airline_labels = dict(zip(airlines["airline_code"], airlines["airline"]))
    return airport_labels, airline_labels


st.set_page_config(page_title="Flight Delay Predictor", page_icon="✈️")
st.title("✈️ Flight Delay Predictor")
st.write("Pick a flight and get the model's predicted delay in minutes.")

airports, airlines = load_options()
codes = list(airports)

with st.form("flight"):
    airline = st.selectbox("Airline", list(airlines), format_func=lambda c: airlines[c], index=list(airlines).index("DL") if "DL" in airlines else 0)
    origin = st.selectbox("Origin airport", codes, format_func=airports.get, index=codes.index("ATL") if "ATL" in codes else 0)
    dest = st.selectbox("Destination airport", codes, format_func=airports.get, index=codes.index("JFK") if "JFK" in codes else 1)
    col1, col2 = st.columns(2)
    flight_date = col1.date_input("Flight date")
    dep_time = col2.time_input("Scheduled departure time (local)", value=datetime.strptime("17:45", "%H:%M").time(), step=300)
    submitted = st.form_submit_button("Predict delay", type="primary")

if submitted:
    if origin == dest:
        st.error("Origin and destination must be different airports.")
    else:
        payload = {
            "airline_code": airline,
            "flight_date": flight_date.isoformat(),
            "origin": origin,
            "dest": dest,
            "sched_dep_time": dep_time.hour * 100 + dep_time.minute,
        }
        try:
            with st.spinner("Asking the model (the first request after idle can take a few seconds)..."):
                resp = requests.post(f"{api_url()}/predict", json=payload, timeout=60)
        except requests.RequestException as e:
            st.error(f"Could not reach the prediction service: {e}")
        else:
            if resp.ok:
                minutes = resp.json()["predicted_delay_minutes"]
                st.metric("Predicted delay", f"{minutes:.0f} min" if minutes >= 1 else "< 1 min")
                st.caption(
                    "An estimate of the typical delay for flights like this one, based on airline, date, "
                    "time and airports. Most flights have little or no delay, so predictions are usually small."
                )
            else:
                st.error(f"The service returned an error ({resp.status_code}): {resp.text[:300]}")
