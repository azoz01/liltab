import numpy as np

from sklearn.compose import make_column_selector, make_column_transformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder


def get_preprocessing_pipeline() -> Pipeline:
    """
    Returns preprocessing pipeline composed with:
        1. Simple imputer with mean/most_frequent strategy depending on data type.
        2. StandardScaler/One-hot encoding depending on data type
    Returns:
        Pipeline: Pipeline with preprocessing steps.
    """
    cat_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("one-hot", OneHotEncoder(sparse_output=False, handle_unknown="ignore")),
        ]
    ).set_output(transform="pandas")

    num_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="mean")),
            ("scaler", StandardScaler()),
        ]
    ).set_output(transform="pandas")

    pipeline = Pipeline(
        [
            (
                "transformers",
                make_column_transformer(
                    (
                        cat_pipeline,
                        make_column_selector(dtype_include=("object", "category")),
                    ),
                    (num_pipeline, make_column_selector(dtype_include=np.number)),
                ),
            )
        ]
    ).set_output(transform="pandas")

    return pipeline
