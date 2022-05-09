import mlflow


def create_mlflow_experiment_by_name(name: str) -> str:
    if str is None:
        raise Exception("Experiment name is empty!")

    experiment = mlflow.get_experiment_by_name(name)
    if experiment is not None:
        return experiment.experiment_id

    experiment_id = mlflow.create_experiment(name)
    print(f"created experiment ID: {experiment_id}")
    return experiment_id
