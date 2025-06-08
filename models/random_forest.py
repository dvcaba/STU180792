from models.model_base import BaseClassifierModel
from sklearn.ensemble import RandomForestClassifier

class RandomForestModel(BaseClassifierModel):
    def __init__(self, df, test_size: float = 0.2, shuffle: bool = False, **kwargs):
        """Random forest classifier with parameter tracking."""
        model = RandomForestClassifier(class_weight="balanced", **kwargs)
        super().__init__(model, df, test_size=test_size, shuffle=shuffle, **kwargs)
