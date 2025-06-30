from models.model_base import BaseClassifierModel  # Import base model class for classification
from sklearn.linear_model import LogisticRegression  # Import logistic regression model from scikit-learn

class LogisticModel(BaseClassifierModel):
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
        Logistic regression model subclassing the generic BaseClassifierModel.

        Parameters:
        - df: Input DataFrame containing features and labels
        - test_size: Proportion of data to use for testing (default: 0.2)
        - shuffle: Whether to shuffle the data before splitting (default: False)
        - use_sliding_window: Whether to use sliding window CV for time series (default: False)
        - n_splits: Number of splits for cross-validation (default: 5)
        - **kwargs: Additional parameters passed to LogisticRegression
        """
        # Initialize a logistic regression model with balanced class weights to handle imbalance
        model = LogisticRegression(class_weight="balanced", **kwargs)

        # Call the constructor of the base class with all relevant arguments
        super().__init__(
            model,
            df,
            test_size=test_size,
            shuffle=shuffle,
            use_sliding_window=use_sliding_window,
            n_splits=n_splits,
            **kwargs,
        )
