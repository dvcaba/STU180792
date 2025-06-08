from models.model_base import BaseClassifierModel
from xgboost import XGBClassifier

class XGBoostClassifierModel(BaseClassifierModel):
    def __init__(self, df, test_size: float = 0.2, shuffle: bool = False, **kwargs):
        """XGBoost classifier with parameter tracking."""
        model = XGBClassifier(eval_metric="logloss", **kwargs)
        super().__init__(model, df, test_size=test_size, shuffle=shuffle, **kwargs)
