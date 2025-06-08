
from models.model_base import BaseClassifierModel
from sklearn.linear_model import LogisticRegression

class LogisticModel(BaseClassifierModel):
    def __init__(self, df, test_size: float = 0.2, shuffle: bool = False, **kwargs):
        """Logistic regression model with parameter tracking."""
        model = LogisticRegression(class_weight="balanced", **kwargs)
        super().__init__(model, df, test_size=test_size, shuffle=shuffle, **kwargs)
