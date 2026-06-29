"""Tools for reproducible wearable freezing-of-gait research baselines."""

from parkinson_wearable_biomarkers.data import (
    DataSchema,
    DataValidationError,
    SensorDataset,
    load_csv,
)
from parkinson_wearable_biomarkers.features import (
    FeatureDataset,
    FeatureExtractionError,
    extract_features,
)
from parkinson_wearable_biomarkers.preprocessing import (
    WindowedDataset,
    WindowingError,
    create_windows,
)
from parkinson_wearable_biomarkers.validation import (
    SubjectFold,
    SubjectLeakageError,
    SubjectSplitError,
    assert_no_subject_leakage,
    group_k_fold_splits,
    leave_one_subject_out_splits,
)

__all__ = [
    "DataSchema",
    "DataValidationError",
    "FeatureDataset",
    "FeatureExtractionError",
    "SensorDataset",
    "SubjectFold",
    "SubjectLeakageError",
    "SubjectSplitError",
    "WindowedDataset",
    "WindowingError",
    "assert_no_subject_leakage",
    "create_windows",
    "extract_features",
    "group_k_fold_splits",
    "leave_one_subject_out_splits",
    "load_csv",
]
