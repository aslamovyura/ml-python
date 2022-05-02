from pathlib import Path
from joblib import dump

import click
from sklearn.metrics import accuracy_score

# from .data import get_dataset
# from .pipeline import create_pipeline
import data as d
import pipeline as p



def train_cl(
    dataset_path: Path,
    save_model_path: Path,
    random_state: int,
    test_split_ratio: float,
    use_scaler: bool,
    max_iter: int,
    logreg_c: float,
# ) -> None:
) -> float:
    features_train, features_val, target_train, target_val = d.get_dataset(
        dataset_path,
        random_state,
        test_split_ratio,
    )
    pipeline = p.create_pipeline(use_scaler, max_iter, logreg_c, random_state)
    pipeline.fit(features_train, target_train)
    accuracy = accuracy_score(target_val, pipeline.predict(features_val))
    click.echo(f"Accuracy: {accuracy}.")
    dump(pipeline, save_model_path)
    click.echo(f"Model is saved to {save_model_path}.")
    return accuracy
