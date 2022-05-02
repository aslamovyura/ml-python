from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import TruncatedSVD, PCA
from sklearn.manifold import TSNE
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier


def create_logistic_pipeline(
    use_scaler: bool, use_pca: bool, max_iter: int, logreg_C: float, random_state: int
) -> Pipeline:
    pipeline_steps = []
    if use_scaler:
        pipeline_steps.append(("scaler", StandardScaler()))

    if use_pca:
        pipeline_steps.append(("pca", PCA(5)))

    pipeline_steps.append(
        (
            "classifier",
            LogisticRegression(
                random_state=random_state, max_iter=max_iter, C=logreg_C
            ),
        )
    )
    return Pipeline(steps=pipeline_steps)


def create_RFC_pipeline(
        use_scaler: bool,
        use_pca: bool,
        n_estimators: int,
        max_depth: int,
        criterion: str,
        random_state: int

) -> Pipeline:
    pipeline_steps = []
    if use_scaler:
        pipeline_steps.append(("scaler", StandardScaler()))

    if use_pca:
        pipeline_steps.append(("pca", PCA(5)))

    pipeline_steps.append(
        (
            "classifier",
            RandomForestClassifier(n_estimators=n_estimators,
                                   criterion=criterion,
                                   max_depth=max_depth,
                                   random_state=random_state),
        )
    )
    return Pipeline(steps=pipeline_steps)