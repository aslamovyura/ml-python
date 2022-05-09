from click.testing import CliRunner
import pytest

from src.forest_ml.train_logistic import train


@pytest.fixture
def runner() -> CliRunner:
    """Fixture providing click runner."""
    return CliRunner()


def test_error_for_invalid_dataset_path(runner: CliRunner) -> None:
    """It fails when test `dataset path` is empty."""
    result = runner.invoke(train, ["-d", ""])
    assert result.exit_code == 2


def test_error_for_invalid_save_model_path(runner: CliRunner) -> None:
    """It fails when test `save model path` is empty."""
    result = runner.invoke(train, ["-d", ""])
    assert result.exit_code == 2


def test_error_for_invalid_cv(runner: CliRunner) -> None:
    """It fails when test `cv` is lower than 1."""
    result = runner.invoke(train, ["--cv", -5])
    assert result.exit_code == 1
