import click
from pathlib import Path
from joblib import dump
from sklearn.model_selection import cross_validate

import mlflow
import mlflow.sklearn

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
    "--use-pca",
    default=False,
    type=bool,
    show_default=True,
)
@click.option(
    "--n-estimators",
    default=100,
    type=int,
    show_default=True,
)
@click.option(
    "--max-depth",
    default=10,
    type=int,
    show_default=True,
)
@click.option(
    "--criterion",
    default='gini',
    type=str,
    show_default=True,
)
def train(
    dataset_path: Path,
    save_model_path: Path,
    random_state: int,
    cv: int,
    use_scaler: bool,
    use_pca: bool,
    n_estimators: int,
    max_depth: int,
    criterion: str
) -> None:
    features, target = dt.get_dataset(dataset_path)

    with mlflow.start_run(experiment_id=2):
        pipeline = p.create_RFC_pipeline(use_scaler, use_pca, n_estimators, max_depth, criterion, random_state)

        scoring = ['accuracy', 'f1_macro', 'precision_macro', 'recall_macro']
        scores = cross_validate(pipeline, features, target, scoring=scoring, cv=cv, return_train_score=True)

        accuracy = np.mean(scores['test_accuracy'])
        precision = np.mean(scores['test_precision_macro'])
        recall = np.mean(scores['test_recall_macro'])
        f1 = np.mean(scores['test_f1_macro'])

        click.echo("---")
        click.echo(f"Accuracy: {accuracy}.")
        click.echo(f"Precision: {precision}.")
        click.echo(f"Recall: {recall}.")
        click.echo(f"F1-score: {f1}.")
        click.echo("---")

        mlflow.log_param("use_scaler", use_scaler)
        mlflow.log_param("use_pca", use_pca)
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_param("criterion", criterion)
        mlflow.log_metric("accuracy", accuracy)

        dump(pipeline, save_model_path)
        click.echo(f"Model is saved to {save_model_path}.")


if __name__ == '__main__':
    train()