from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import RandomForestClassifier


def create_logistic_pipeline(
    use_scaler: bool = True,
    use_dim_reducer: bool = True,
    max_iter: int = 1000,
    logreg_C: float = 1.0,
    random_state: int = 42,
) -> Pipeline:
    pipeline_steps = []
    if use_scaler:
        pipeline_steps.append(("scaler", StandardScaler()))

    if use_dim_reducer:
        pipeline_steps.append(("reduce_dim", TruncatedSVD(n_components=10)))

    pipeline_steps.append(
        (
            "classifier",
            LogisticRegression(
                random_state=random_state, max_iter=max_iter, C=logreg_C
            ),
        )
    )
    return Pipeline(steps=pipeline_steps)


def create_rfc_pipeline(
    use_scaler: bool = True,
    use_dim_reducer: bool = True,
    n_estimators: int = 100,
    max_depth: int = 10,
    criterion: str = "gini",
    random_state: int = 42,
) -> Pipeline:
    pipeline_steps = []
    if use_scaler:
        pipeline_steps.append(("scaler", StandardScaler()))

    if use_dim_reducer:
        pipeline_steps.append(("reduce_dim", TruncatedSVD(n_components=10)))

    pipeline_steps.append(
        (
            "classifier",
            RandomForestClassifier(
                n_estimators=n_estimators,
                criterion=criterion,
                max_depth=max_depth,
                random_state=random_state,
            ),
        )
    )
    return Pipeline(steps=pipeline_steps)
