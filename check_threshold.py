import os
import sys
from pathlib import Path

import mlflow


THRESHOLD = 0.85


def main() -> int:
    run_id_path = Path("model_info.txt")

    if not run_id_path.exists():
        print("ERROR: model_info.txt not found. Did the validate job upload it?")
        return 1

    run_id = run_id_path.read_text(encoding="utf-8").strip()
    if not run_id:
        print("ERROR: model_info.txt is empty.")
        return 1

    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "./mlruns")
    mlflow.set_tracking_uri(tracking_uri)

    run = mlflow.get_run(run_id)
    accuracy = run.data.metrics.get("accuracy")

    if accuracy is None:
        print(f"ERROR: No 'accuracy' metric found for run {run_id}.")
        return 1

    print(f"Run ID   : {run_id}")
    print(f"Accuracy : {accuracy:.4f}")
    print(f"Threshold: {THRESHOLD:.2f}")

    if accuracy < THRESHOLD:
        print(f"❌ Accuracy {accuracy:.4f} is BELOW threshold {THRESHOLD}.")
        print("   Deployment blocked.")
        return 1

    print(f"✅ Accuracy {accuracy:.4f} meets threshold. Deployment approved.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
