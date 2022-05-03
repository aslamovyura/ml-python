from click.testing import CliRunner
import pytest
import click

from src.forest_ml.train_rfc import train_with_opt_hyperparameters

@pytest.fixture
def runner() -> CliRunner:
    """Fixture providing click runner."""
    return CliRunner()

def test_error_for_invalid_dataset_path(
    runner: CliRunner
) -> None:
    """It fails when test `dataset path` is empty."""
    result = runner.invoke(train_with_opt_hyperparameters, ["-d", "",] )
    assert result.exit_code == 2





