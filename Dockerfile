FROM python:3.10-slim

# Accept the MLflow Run ID at build time so the image is
# pinned to a specific, audited model version.
ARG RUN_ID=unknown
ENV RUN_ID=${RUN_ID}

WORKDIR /app

# Install runtime dependencies
RUN pip install --no-cache-dir mlflow scikit-learn

# Entry point: simulate downloading the model artifact for
# the given Run ID. Replace 'echo' with a real mlflow
# artifacts download command when a live server is available.
CMD ["sh", "-c", "echo \"Downloading model artifacts for Run ID: ${RUN_ID}\" && echo \"Model ready.\""]
