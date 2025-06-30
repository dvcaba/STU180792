from models.model_base import BaseClassifierModel  # Import the custom base model class
from sklearn.ensemble import RandomForestClassifier  # Import the Random Forest classifier from scikit-learn

class RandomForestModel(BaseClassifierModel):
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
        Initialize a Random Forest classifier within the BaseClassifierModel framework.

        Parameters:
        - df: The input DataFrame containing features and target variable
        - test_size: Fraction of the dataset to use for testing (default: 0.2)
        - shuffle: Whether to shuffle data before splitting (default: False)
        - use_sliding_window: Whether to use time series sliding window CV (default: False)
        - n_splits: Number of splits for sliding window (default: 5)
        - **kwargs: Additional keyword arguments passed to RandomForestClassifier
        """
        # Create a RandomForestClassifier with balanced class weights to handle imbalanced datasets
        model = RandomForestClassifier(class_weight="balanced", **kwargs)

        # Call the constructor of the BaseClassifierModel to initialize the pipeline
        super().__init__(
            model,
            df,
            test_size=test_size,
            shuffle=shuffle,
            use_sliding_window=use_sliding_window,
            n_splits=n_splits,
            **kwargs,
        )
