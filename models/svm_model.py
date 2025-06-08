from models.model_base import BaseClassifierModel
from sklearn.svm import SVC

class SVMClassifierModel(BaseClassifierModel):
    def __init__(self, df, test_size: float = 0.2, shuffle: bool = False, **kwargs):
        """SVM classifier with parameter tracking."""
        model = SVC(probability=True, **kwargs)
        super().__init__(model, df, test_size=test_size, shuffle=shuffle, **kwargs)
