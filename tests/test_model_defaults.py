import unittest  # Standard library for unit testing
import os  # For manipulating file paths
import sys  # To modify the system path for module imports
import pandas as pd  # For creating and manipulating the test DataFrame

# Add the parent directory to sys.path so that model modules can be imported correctly
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

# Import custom classifier models to be tested
from models.svm_model import SVMClassifierModel
from models.xgboost_model import XGBoostClassifierModel

class TestModelDefaults(unittest.TestCase):
    def setUp(self):
        """
        Setup method run before each test.
        Creates a small synthetic DataFrame for model instantiation.
        """
        self.df = pd.DataFrame({
            "Date": pd.date_range("2021-01-01", periods=10),  # 10 sequential dates
            "Ticker": ["AAPL"] * 10,                          # Constant ticker for simplicity
            "Feature": range(10),                             # Dummy feature values
            "Target": [0, 1] * 5,                              # Balanced binary target
        })

    def test_svm_default_class_weight(self):
        """
        Verify that the default class_weight for SVM is set to 'balanced'.
        """
        model = SVMClassifierModel(self.df)
        self.assertEqual(
            model.model.get_params().get("class_weight"), "balanced"
        )

    def test_xgboost_scale_pos_weight(self):
        """
        Check that XGBoost automatically calculates scale_pos_weight
        as the ratio of negative to positive samples in the target.
        """
        expected_weight = (
            (self.df["Target"] == 0).sum() / (self.df["Target"] == 1).sum()
        )
        model = XGBoostClassifierModel(self.df)
        self.assertAlmostEqual(
            model.model.get_params().get("scale_pos_weight"), expected_weight
        )

# Entry point for the test suite when run as a script
if __name__ == "__main__":
    unittest.main()
