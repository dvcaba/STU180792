from models.model_base import BaseClassifierModel  # Import the base classification model wrapper
from sklearn.svm import SVC  # Import Support Vector Classifier from scikit-learn

class SVMClassifierModel(BaseClassifierModel):
    def __init__(
        self,
        df,
        test_size: float = 0.2,
        shuffle: bool = False,
        use_sliding_window: bool = False,
        n_splits: int = 5,
        **kwargs,
    ):
        """
        Initialize an SVM classifier that inherits from BaseClassifierModel.

        Parameters:
        - df: Input DataFrame with features and target.
        - test_size: Fraction of data to use for testing (default: 0.2).
        - shuffle: Whether to shuffle data before splitting (default: False).
        - use_sliding_window: Whether to use time-series cross-validation (default: False).
        - n_splits: Number of cross-validation splits (default: 5).
        - **kwargs: Additional parameters to pass to the SVC model.

        The default parameter 'class_weight="balanced"' is applied unless overridden in kwargs.
        """
        # Default parameter to handle class imbalance
        default_params = {"class_weight": "balanced"}
        # Allow overriding default parameters with user-specified kwargs
        default_params.update(kwargs)

        # Create the SVC model with probability=True to enable probability outputs
        model = SVC(probability=True, **default_params)

        # Call the parent constructor to initialize the training pipeline
        super().__init__(
            model,
            df,
            test_size=test_size,
            shuffle=shuffle,
            use_sliding_window=use_sliding_window,
            n_splits=n_splits,
            **kwargs,
        )
