import os
from pathlib import Path

import dagshub
import mlflow
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


THRESHOLD_READY = 0.85


def main() -> None:
    # Initialise DagsHub — sets the MLflow tracking URI and
    # injects credentials from env vars automatically in CI.
    dagshub.init(
        repo_owner="Cludoy",
        repo_name="MLOPs-Assignment6",
        mlflow=True,
    )

    experiment_name = os.getenv(
        "MLFLOW_EXPERIMENT_NAME", "assignment6_pipeline"
    )
    mlflow.set_experiment(experiment_name)

    X, y = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    with mlflow.start_run() as run:
        run_id = run.info.run_id

        model = LogisticRegression(max_iter=200, random_state=42)
        model.fit(X_train, y_train)

        preds = model.predict(X_test)
        accuracy = accuracy_score(y_test, preds)

        # Optional override for demo runs (set via workflow_dispatch input)
        forced_accuracy = os.getenv("FORCE_ACCURACY")
        if forced_accuracy is not None and forced_accuracy.strip():
            accuracy = float(forced_accuracy)

        mlflow.log_param("model", "LogisticRegression")
        mlflow.log_param("threshold", THRESHOLD_READY)
        mlflow.log_metric("accuracy", accuracy)
        mlflow.sklearn.log_model(model, artifact_path="model")

        # Write Run ID — consumed by the deploy job via artifact hand-off
        Path("model_info.txt").write_text(run_id, encoding="utf-8")

        print(f"Run ID  : {run_id}")
        print(f"Accuracy: {accuracy:.4f}")


if __name__ == "__main__":
    main()
