from models.model_base import BaseClassifierModel  # Import base model wrapper
from xgboost import XGBClassifier  # Import XGBoost classifier

class XGBoostClassifierModel(BaseClassifierModel):
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
        Initialize an XGBoost classifier wrapped in BaseClassifierModel.

        Parameters:
        - df: Input DataFrame containing features and 'Target' column.
        - test_size: Fraction of data to use for test split (default: 0.2).
        - shuffle: Whether to shuffle the dataset before splitting (default: False).
        - use_sliding_window: Whether to use time-series sliding window CV (default: False).
        - n_splits: Number of splits for time-series cross-validation (default: 5).
        - **kwargs: Additional keyword arguments for XGBClassifier.

        Automatically computes `scale_pos_weight` if not provided to handle class imbalance.
        """

        # Count positive and negative samples in the target column
        pos_count = (df["Target"] == 1).sum()
        neg_count = (df["Target"] == 0).sum()

        # Compute default scale_pos_weight for class imbalance
        default_weight = neg_count / pos_count if pos_count else 1

        # Allow user to override scale_pos_weight, otherwise use default
        scale_pos_weight = kwargs.pop("scale_pos_weight", default_weight)

        # Initialize the XGBClassifier with a log loss evaluation metric
        model = XGBClassifier(
            eval_metric="logloss",  # Avoid warning: must specify eval_metric manually
            scale_pos_weight=scale_pos_weight,  # Adjust for imbalance
            **kwargs  # Include any additional parameters passed
        )

        # Call the BaseClassifierModel constructor to complete setup
        super().__init__(
            model,
            df,
            test_size=test_size,
            shuffle=shuffle,
            use_sliding_window=use_sliding_window,
            n_splits=n_splits,
            **kwargs,
        )
