from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def get_preprocessing_pipeline() -> Pipeline:
    """
    Returns preprocessing pipeline composed with:
        1. Simple imputer with mean strategy
        2. StandardScaler
    Returns:
        Pipeline: Pipeline with preprocessing steps.
    """
    return make_pipeline(SimpleImputer(strategy="mean"), StandardScaler())
