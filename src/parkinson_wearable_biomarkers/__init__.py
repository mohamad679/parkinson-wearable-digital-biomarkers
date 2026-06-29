"""Tools for reproducible wearable freezing-of-gait research baselines."""

from parkinson_wearable_biomarkers.calibration import (
    CalibrationBin,
    CalibrationInputError,
    brier_score,
    calibration_curve_data,
    expected_calibration_error,
)
from parkinson_wearable_biomarkers.data import (
    DataSchema,
    DataValidationError,
    SensorDataset,
    load_csv,
)
from parkinson_wearable_biomarkers.evaluate import (
    ConfusionMatrix,
    MetricInputError,
    ThresholdMetrics,
    auprc,
    auroc,
    confusion_matrix,
    f1_score,
    sensitivity,
    specificity,
    threshold_analysis,
)
from parkinson_wearable_biomarkers.features import (
    FeatureDataset,
    FeatureExtractionError,
    extract_features,
)
from parkinson_wearable_biomarkers.models import (
    FoldPredictions,
    ModelInputError,
    ModelTrainingError,
    TrainedBaselineModel,
    fit_logistic_regression,
    fit_predict_fold,
    fit_random_forest,
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
    "CalibrationBin",
    "CalibrationInputError",
    "ConfusionMatrix",
    "DataSchema",
    "DataValidationError",
    "FeatureDataset",
    "FeatureExtractionError",
    "FoldPredictions",
    "MetricInputError",
    "ModelInputError",
    "ModelTrainingError",
    "SensorDataset",
    "SubjectFold",
    "SubjectLeakageError",
    "SubjectSplitError",
    "ThresholdMetrics",
    "TrainedBaselineModel",
    "WindowedDataset",
    "WindowingError",
    "assert_no_subject_leakage",
    "auprc",
    "auroc",
    "brier_score",
    "calibration_curve_data",
    "confusion_matrix",
    "create_windows",
    "expected_calibration_error",
    "extract_features",
    "f1_score",
    "fit_logistic_regression",
    "fit_predict_fold",
    "fit_random_forest",
    "group_k_fold_splits",
    "leave_one_subject_out_splits",
    "load_csv",
    "sensitivity",
    "specificity",
    "threshold_analysis",
]
