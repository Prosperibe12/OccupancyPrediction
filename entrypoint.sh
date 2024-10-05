#!/usr/bin/env sh
export MLFLOW_TRACKING_URI=http://localhost:5000
mlflow models serve -m "models:/MLOccupancy/9"
python app.py