# Front End (Streamlit)

A form that calls the deployed API. Part of the project described in the [root README](../README.md); it calls the [API](../api/README.md).

## Reproduce this module
Run from the repo root.
```
pip install -r frontend/requirements.txt
streamlit run frontend/app.py
```
To deploy it, see the Community Cloud steps below.

## How it works
`frontend/app.py` is a one-page form that calls the deployed API (`POST /predict`). The user picks an airline, origin airport, destination airport, flight date and scheduled departure time, and the page shows the predicted delay in minutes. It also validates that origin and destination differ and shows the API's error message if a call fails.

It is self-contained: `frontend/airports.csv` and `frontend/airlines.csv` hold the dropdown options (copied from `data/`), and `frontend/requirements.txt` lists its dependencies.

Run locally from the repo root:
```
pip install -r frontend/requirements.txt
streamlit run frontend/app.py
```

The API URL defaults to the Cloud Run service (`https://airline-delay-api-170279949235.us-east1.run.app`). To point at a different backend (for example `http://localhost:8080`), set `API_URL` in `.streamlit/secrets.toml` locally (git-ignored) or in the app's **Secrets** settings on Streamlit Community Cloud:
```
API_URL = "https://your-service-url"
```

**Deploying on Streamlit Community Cloud:** push the repo to GitHub, then at https://share.streamlit.io choose **Create app**, select the repo and branch, set the main file path to `frontend/app.py`, and deploy. Streamlit installs `frontend/requirements.txt` automatically.
