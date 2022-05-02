from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import TruncatedSVD, PCA
from sklearn.manifold import TSNE
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_val_score, KFold, cross_validate
import click
from pathlib import Path

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
def train_rfc_with_opt_hyperparameters(dataset_path: Path, save_model_path: Path) -> None:

    features, target = dt.get_dataset(dataset_path)

    param_space = {
        "scaler": ["passthrough", StandardScaler()],
        "reduce_dim": ["passthrough", TruncatedSVD(10), TruncatedSVD(20)],
        "classifier__n_estimators": range(10, 126),
        "classifier__max_depth": range(5, 50),
        "classifier__criterion": ['entropy', 'gini']
    }

    with mlflow.start_run(experiment_id=2):
        pipe = p.create_RFC_pipeline()
        accuracy_nested = []
        model_nested = []
        # N_TRIALS = 20
        N_TRIALS = 2
        for i in range(N_TRIALS):
            # For each trial, we use cross-validation splits on independently
            # randomly shuffled data by passing distinct values to the random_state
            # parameter.
            inner_cv = KFold(n_splits=5, shuffle=True, random_state=i)
            outer_cv = KFold(n_splits=3, shuffle=True, random_state=i)

            # Non_nested parameter search and scoring
            model = RandomizedSearchCV(pipe,
                                       param_distributions=param_space,
                                       # n_iter=20,
                                       n_iter=2,
                                       scoring='accuracy',
                                       cv=inner_cv,
                                       n_jobs=-1)
            model.fit(features, target)

            # Nested CV with parameter optimization
            params = model.best_params_
            print(params)

            scoring = ['accuracy', 'f1_macro', 'precision_macro', 'recall_macro']
            pipeline = p.create_RFC_pipeline()

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
            mlflow.log_param("use_pca", params['reduce_dim'])
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
    train_rfc_with_opt_hyperparameters()