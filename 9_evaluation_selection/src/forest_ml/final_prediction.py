import click
from pathlib import Path
import pandas as pd

from .data import get_test_features, get_dataset
from .pipeline import create_rfc_pipeline


@click.command()
@click.option(
    "-d",
    "--train-dataset-path",
    default="./data/train.csv",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    show_default=True,
)
@click.option(
    "-t",
    "--test-dataset-path",
    default="./data/test.csv",
    type=click.Path(dir_okay=False, writable=True, path_type=Path),
    show_default=True,
)
def train_and_test(
    train_dataset_path: Path,
    test_dataset_path: Path,
) -> None:
    click.echo("loading train and test data...")
    (train_features, train_labels) = get_dataset(train_dataset_path)
    train_features.drop(columns=['Id'], inplace=True)

    test_features = get_test_features(test_dataset_path)
    test_ids = test_features['Id']
    test_features.drop(columns=['Id'], inplace=True)
    click.echo("success!")

    pipeline = create_rfc_pipeline(
        use_scaler = False,
        use_dim_reducer = True,
        n_estimators = 115,
        max_depth = 8,
        criterion = 'entropy',
        random_state = 42,
    )

    click.echo("fitting...")
    pipeline.fit(train_features, train_labels)
    click.echo("success!")

    click.echo("prediction...")
    test_labels = pipeline.predict(test_features)
    click.echo("success!")

    click.echo("saving result to csv...")
    d = {'Id': test_ids, 'Cover_Type': test_labels}
    submission_data = pd.DataFrame(data=d)
    submission_data = submission_data.set_index('Id')
    submission_data.to_csv('submission.csv')
    click.echo("success!")


