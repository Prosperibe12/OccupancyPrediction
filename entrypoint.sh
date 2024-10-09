#!/bin/bash
set -e

# Use full path to the virtual environment's binaries
echo "MLflow server started"
/opt/venv/bin/mlflow server --host 0.0.0.0 --port 6000 &

# Run the pipeline
/opt/venv/bin/python3 run_pipeline.py

# Start the Flask app
/opt/venv/bin/python3 app.py