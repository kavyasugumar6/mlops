# HealthPredict Stroke MLOps

End-to-end example showing MLflow experiment tracking, a Streamlit inference UI, Dockerization, and Railway-ready deployment for stroke-risk prediction.

## Quickstart
1) Create a virtual environment and install dependencies:
```
pip install -r requirements.txt
```
2) Train and log with MLflow (writes artifacts/model.pkl and artifacts/reference_data.csv):
```
python train.py
```
3) Inspect experiments locally:
```
mlflow ui --backend-store-uri mlruns
```
4) Launch the Streamlit dashboard:
```
streamlit run app.py
```

## Project Structure
- `train.py` – Generates synthetic stroke-like data, trains a RandomForest pipeline, logs to MLflow, and saves deployable artifacts.
- `app.py` – Streamlit UI for real-time scoring, showing probability and reference distributions.
- `requirements.txt` – Pinned runtime dependencies.
- `Dockerfile` – Containerizes the app for local use or Railway.
- `Procfile` – Railway-friendly entrypoint for Streamlit (`web` process on `$PORT`).
- `railway.json` – Tells Railway to use the Docker builder.
- `artifacts/` – Output folder for the trained pipeline and reference data (created by `train.py`).
- `mlruns/` – Local MLflow tracking store (created after you run training and MLflow UI).

## Docker
Build and run locally:
```
docker build -t healthpredict .
docker run -p 8501:8501 healthpredict
```

## Railway Deployment (summary)
- Create a new Railway project, add this repo, and choose Docker deployment.
- Set the service port to 8501 and expose to the web.
- Ensure `python train.py` has been run so `artifacts/model.pkl` exists before building the image.

## Notes
- Data is synthetic to keep the example self-contained; swap `generate_synthetic_stroke_data` with a real dataset loader as needed.
- MLflow tracking URI defaults to the local `mlruns` folder; configure `MLFLOW_TRACKING_URI` for remote tracking servers.
