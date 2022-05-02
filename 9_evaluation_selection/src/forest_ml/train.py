import click
from pathlib import Path
from joblib import dump
from sklearn.metrics import accuracy_score

import data as dt
import pipeline as p


@click.command()
@click.option(
    "-d",
    "--dataset-path",
    default="../../data/train.csv",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    show_default=True,
)
@click.option(
    "-s",
    "--save-model-path",
    default="../../data/model.joblib",
    type=click.Path(dir_okay=False, writable=True, path_type=Path),
    show_default=True,
)
@click.option(
    "--random-state",
    default=42,
    type=int,
    show_default=True,
)
@click.option(
    "--test-split-ratio",
    default=0.2,
    type=click.FloatRange(0, 1, min_open=True, max_open=True),
    show_default=True,
)
@click.option(
    "--use-scaler",
    default=True,
    type=bool,
    show_default=True,
)
@click.option(
    "--max-iter",
    default=100,
    type=int,
    show_default=True,
)
@click.option(
    "--logreg-c",
    default=1.0,
    type=float,
    show_default=True,
)
def train_cl(
    dataset_path: Path,
    save_model_path: Path,
    random_state: int,
    test_split_ratio: float,
    use_scaler: bool,
    max_iter: int,
    logreg_c: float,
) -> None:
    features_train, features_val, target_train, target_val = dt.get_dataset(
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


if __name__ == '__main__':
    train_cl()