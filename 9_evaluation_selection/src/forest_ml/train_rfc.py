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
from .pipeline import create_rfc_pipeline
from .utils.mlflow_utils import create_mlflow_experiment_by_name

EXPERIMENT_NAME = 'RandomForestClassifier'

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
    default="./data/model_rfc.joblib",
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
    use_dim_reducer: bool,
    n_estimators: int,
    max_depth: int,
    criterion: str
) -> None:
    features, target = get_dataset(dataset_path)

    experiment_id = create_mlflow_experiment_by_name(EXPERIMENT_NAME)
    with mlflow.start_run(experiment_id=experiment_id):
        pipeline = create_rfc_pipeline(use_scaler, use_dim_reducer, n_estimators, max_depth, criterion, random_state)

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

        mlflow.log_param("scaler", 'StandardScaler()' if use_scaler else 'passthrough')
        mlflow.log_param("dim_reducer", 'TruncatedSVD(10)' if use_dim_reducer else 'passthrough')
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_param("criterion", criterion)
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
    default="./data/model_rfc.joblib",
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
        "classifier__n_estimators": range(10, 126),
        "classifier__max_depth": range(5, 50),
        "classifier__criterion": ['entropy', 'gini']
    }

    experiment_id = create_mlflow_experiment_by_name(EXPERIMENT_NAME)
    pipe = create_rfc_pipeline()
    accuracy_nested = []
    model_nested = []
    # N_TRIALS = 20
    N_TRIALS = 10
    # N_TRIALS = 2
    for i in range(N_TRIALS):
        # with mlflow.start_run(experiment_id=2):
        with mlflow.start_run(experiment_id=experiment_id):
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

            scoring = ['accuracy', 'f1_macro', 'precision_macro', 'recall_macro']
            # pipeline = p.create_rfc_pipeline()
            # pipeline = create_rfc_pipeline()

            scores = cross_validate(model, features, target, scoring=scoring, cv=outer_cv, return_train_score=True)

            accuracy = np.mean(scores['test_accuracy'])
            # print(accuracy)
            precision = np.mean(scores['test_precision_macro'])
            recall = np.mean(scores['test_recall_macro'])
            f1 = np.mean(scores['test_f1_macro'])

            click.echo("---")
            click.echo(f"Accuracy: {accuracy}.")
            click.echo(f"Precision: {precision}.")
            click.echo(f"Recall: {recall}.")
            click.echo(f"F1-score: {f1}.")
            click.echo("---")

            mlflow.log_param("scaler", params['scaler'])
            mlflow.log_param("dim_reducer", params['reduce_dim'])
            mlflow.log_param("n_estimators", params['classifier__n_estimators'])
            mlflow.log_param("max_depth", params['classifier__max_depth'])
            mlflow.log_param("criterion", params['classifier__criterion'])
            mlflow.log_metric("accuracy", accuracy)
            mlflow.log_metric("precision", precision)
            mlflow.log_metric("recall", recall)
            mlflow.log_metric("F1", f1)

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