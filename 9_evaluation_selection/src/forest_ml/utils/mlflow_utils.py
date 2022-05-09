import mlflow
from typing import Any

def create_mlflow_experiment_by_name(name: str) -> Any:
    """Create mlflow experiment by specified name.
    Returns experiment ID (existing or created)."""
    if str is None:
        raise Exception("Experiment name is empty!")

    experiment = mlflow.get_experiment_by_name(name)
    if experiment is not None:
        return experiment.experiment_id

    experiment_id = mlflow.create_experiment(name)
    print(f"created experiment ID: {experiment_id}")
    return experiment_id
