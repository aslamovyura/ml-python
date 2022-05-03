import click
from pathlib import Path
from joblib import dump
from sklearn.model_selection import cross_validate, RandomizedSearchCV, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import TruncatedSVD

import mlflow
import mlflow.sklearn

import numpy as np
from .data import get_dataset
from .pipeline import create_logistic_pipeline


@click.command()
@click.option(
    "-d",
    "--dataset-path",
    default="./data/train.csv",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    show_default=True,
)
@click.option(
    "-s",
    "--save-model-path",
    default="./data/model_logreg.joblib",
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
    "--use-dim-reducer",
    default=False,
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
def train(
    dataset_path: Path,
    save_model_path: Path,
    random_state: int,
    cv: int,
    use_scaler: bool,
    use_dim_reducer: bool,
    max_iter: int,
    logreg_c: float
) -> None:
    features, target = get_dataset(dataset_path)

    with mlflow.start_run(experiment_id=1):
        pipeline = create_logistic_pipeline(use_scaler, use_dim_reducer, max_iter, logreg_c, random_state)

        scoring = ['accuracy', 'f1_macro', 'precision_macro', 'recall_macro', 'r2']
        scores = cross_validate(pipeline, features, target, scoring=scoring, cv=cv, return_train_score=True)

        accuracy = np.mean(scores['test_accuracy'])
        precision = np.mean(scores['test_precision_macro'])
        recall = np.mean(scores['test_recall_macro'])
        f1 = np.mean(scores['test_f1_macro'])
        test_r2 = np.mean(scores['test_r2'])
        train_r2 = np.mean(scores['train_r2'])

        click.echo("---")
        click.echo(f"Accuracy: {accuracy}.")
        click.echo(f"Precision: {precision}.")
        click.echo(f"Recall: {recall}.")
        click.echo(f"F1-score: {f1}.")
        click.echo(f"R2 error (train): {train_r2}.")
        click.echo(f"R2 error (test): {test_r2}.")
        click.echo("---")

        mlflow.log_param("use_scaler", use_scaler)
        mlflow.log_param("use_dim_reducer", use_dim_reducer)
        mlflow.log_param("max_iter", max_iter)
        mlflow.log_param("logreg_c", logreg_c)
        mlflow.log_metric("accuracy", accuracy)

        dump(pipeline, save_model_path)
        click.echo(f"Model is saved to {save_model_path}.")


@click.command()
@click.option(
    "-d",
    "--dataset-path",
    default="./data/train.csv",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    show_default=True,
)
@click.option(
    "-s",
    "--save-model-path",
    default="./data/model_logreg.joblib",
    type=click.Path(dir_okay=False, writable=True, path_type=Path),
    show_default=True,
)
def train_with_opt_hyperparameters(
        dataset_path: Path,
        save_model_path: Path
) -> None:

    features, target = get_dataset(dataset_path)

    param_space = {
        "scaler": ["passthrough", StandardScaler()],
        "reduce_dim": ["passthrough", TruncatedSVD(10), TruncatedSVD(20)],
        "classifier__max_iter": range(100, 5000),
        "classifier__C": [0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000]
    }

    pipe = create_logistic_pipeline()
    accuracy_nested = []
    model_nested = []
    # N_TRIALS = 20
    # N_TRIALS = 10
    # N_TRIALS = 5
    N_TRIALS = 2
    for i in range(N_TRIALS):
        with mlflow.start_run(experiment_id=1):
            # For each trial, we use cross-validation splits on independently
            # randomly shuffled data by passing distinct values to the random_state
            # parameter.
            inner_cv = KFold(n_splits=5, shuffle=True, random_state=i)
            outer_cv = KFold(n_splits=3, shuffle=True, random_state=i)

            # Non_nested parameter search and scoring
            model = RandomizedSearchCV(pipe,
                                       param_distributions=param_space,
                                       # n_iter=20,
                                       # n_iter=10,
                                       n_iter=2,
                                       scoring='accuracy',
                                       cv=inner_cv,
                                       n_jobs=-1)
            model.fit(features, target)

            # Nested CV with parameter optimization
            params = model.best_params_
            print(params)

            scoring = ['accuracy', 'f1_macro', 'precision_macro', 'recall_macro', 'r2']
            scores = cross_validate(model, features, target, scoring=scoring, cv=outer_cv, return_train_score=True)

            accuracy = np.mean(scores['test_accuracy'])
            precision = np.mean(scores['test_precision_macro'])
            recall = np.mean(scores['test_recall_macro'])
            f1 = np.mean(scores['test_f1_macro'])
            test_r2 = np.mean(scores['test_r2'])
            train_r2 = np.mean(scores['train_r2'])

            click.echo("---")
            click.echo(f"Accuracy: {accuracy}.")
            click.echo(f"Precision: {precision}.")
            click.echo(f"Recall: {recall}.")
            click.echo(f"F1-score: {f1}.")
            click.echo(f"R2 error (train): {train_r2}.")
            click.echo(f"R2 error (test): {test_r2}.")
            click.echo("---")

            mlflow.log_param("scaler", params['scaler'])
            mlflow.log_param("dim_reducer", params['reduce_dim'])
            mlflow.log_param("max_iter", params['classifier__max_iter'])
            mlflow.log_param("logreg_C", params['classifier__C'])
            mlflow.log_metric("accuracy", accuracy)
            mlflow.log_metric("precision", precision)
            mlflow.log_metric("recall", recall)
            mlflow.log_metric("F1", f1)
            mlflow.log_metric("R2 error", test_r2)

            accuracy_nested.append(accuracy)
            model_nested.append(model)

    print(accuracy_nested)
    index = np.argmax(accuracy_nested)
    model_opt = model_nested[index]

    dump(model_opt, save_model_path)
    click.echo(f"Model is saved to {save_model_path}.")



if __name__ == '__main__':
    # train()
    train_with_opt_hyperparameters()