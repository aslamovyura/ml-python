import click
from pathlib import Path
from joblib import dump
from sklearn.model_selection import cross_validate

import numpy as np
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
    "--cv",
    default=5,
    type=int,
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
    default=1000,
    type=int,
    show_default=True,
)
@click.option(
    "--logreg-c",
    default=1.0,
    type=float,
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
    cv: int,
    use_scaler: bool,
    max_iter: int,
    logreg_c: float,
) -> None:
    features, target = dt.get_dataset(dataset_path)
    pipeline = p.create_pipeline(use_scaler, max_iter, logreg_c, random_state)

    scoring = ['f1_macro', 'precision_macro', 'recall_macro', 'r2']
    scores = cross_validate(pipeline, features, target, scoring=scoring, cv=cv, return_train_score=True)

    precision = np.mean(scores['test_precision_macro'])
    recall = np.mean(scores['test_recall_macro'])
    f1 = np.mean(scores['test_f1_macro'])
    test_r2 = np.mean(scores['test_r2'])
    train_r2 = np.mean(scores['train_r2'])

    click.echo("---")
    click.echo(f"Precision: {precision}.")
    click.echo(f"Recall: {recall}.")
    click.echo(f"F1-score: {f1}.")
    click.echo(f"R2 error (train): {train_r2}.")
    click.echo(f"R2 error (test): {test_r2}.")
    click.echo("---")

    dump(pipeline, save_model_path)
    click.echo(f"Model is saved to {save_model_path}.")


if __name__ == '__main__':
    train_cl()