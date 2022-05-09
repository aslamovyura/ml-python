import pandas as pd
from pandasgui import show
from pandas_profiling import ProfileReport
import logging
from pathlib import Path
import click


@click.command()
@click.option(
    "-d",
    "--dataset-path",
    default="./data/train.csv",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    show_default=True,
)
def dataset_profiling(dataset_path: Path) -> None:
    if dataset_path is None:
        raise Exception("Dataset path is empty!")

    dataset = pd.read_csv(dataset_path)
    profile = ProfileReport(dataset, title="Pandas Profiling Report", explorative=True)

    filename = "profiling_report.html"
    profile.to_file(filename)

    click.echo("---")
    click.echo(
        f"Profiling report successfully saved to file `{filename}` in the root directory!"
    )
    click.echo("---")


@click.command()
@click.option(
    "-d",
    "--dataset-path",
    default="./data/train.csv",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    show_default=True,
)
def visualize_dataset(dataset_path: Path) -> None:
    if dataset_path is None:
        raise Exception("Dataset path is empty!")

    dataset = pd.read_csv(dataset_path)
    gui = show(dataset)


if __name__ == "__main__":
    dataset_profiling()
