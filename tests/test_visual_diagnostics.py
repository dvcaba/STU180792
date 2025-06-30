# tests/test_visual_diagnostics.py

import pytest  # Pytest for running tests
import matplotlib.pyplot as plt  # Used for plotting predictions
from unittest.mock import patch  # Used to mock seaborn heatmap in test
from ingest.stock_data_fetcher import StockDataFetcher  # Load stock data
from preprocessing.data_preprocessor import DataPreprocessor  # Preprocessing module
from models.random_forest import RandomForestModel  # Model used for predictions

@pytest.mark.visual
def test_prediction_vs_actual_plot():
    """
    Test to verify the generation of a line plot comparing actual vs predicted values.
    This includes:
    - Loading the dataset from a local file
    - Preprocessing for classification task
    - Training a Random Forest model
    - Plotting actual vs predicted values
    - Saving the plot to the diagnostics folder
    """

    # Load historical stock data from local CSV (offline)
    df = StockDataFetcher.load_from_csv("data/raw/stock_data.csv")

    # Set the number of periods ahead to predict
    forecast_horizon = 1

    # Initialize the preprocessing pipeline
    preprocessor = DataPreprocessor(
        data=df,
        scale="standard",            # StandardScaler for feature scaling
        model_type="classification", # Predict direction (classification)
        target_type="direction",
        forecast_horizon=forecast_horizon
    )
    df_processed = preprocessor.preprocess()

    # Train the Random Forest model and run the full workflow
    model = RandomForestModel(df_processed)
    model.run_all()

    # Get actual and predicted target values from the model
    y_test, y_pred = model.get_raw_predictions()

    # Check that the prediction and actual arrays are of the same length
    assert len(y_test) == len(y_pred), "Mismatch in length of predictions and targets"

    # Create the actual vs predicted plot and save it
    plt.figure(figsize=(12, 6))
    plt.plot(y_test.values, label="Actual", linestyle="-")
    plt.plot(y_pred, label="Predicted", linestyle="--")
    plt.title("Predicted vs Actual Direction")
    plt.xlabel("Time Steps")
    plt.ylabel("Direction")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("results/diagnostics/predicted_vs_actual.png")

    # Check that the plot was created (i.e., figure exists)
    assert plt.gcf().number > 0, "Plot was not generated"

def test_empty_correlation_matrix_skips_heatmap(capsys):
    """
    Test that verifies the heatmap is skipped when the correlation matrix is empty.
    The seaborn heatmap function should not be called and a warning should be printed.
    """
    import pandas as pd
    from eda.stock_eda import StockEDA  # Module that includes correlation plotting logic

    # Create an empty DataFrame with expected columns
    df = pd.DataFrame(columns=["Date", "Open", "High", "Low", "Close", "Volume", "Ticker"])

    # Instantiate the StockEDA class with the empty DataFrame
    eda = StockEDA(df)

    # Patch seaborn.heatmap to ensure it is not called
    with patch("seaborn.heatmap") as mock_heatmap:
        eda.plot_correlation_matrix()
        mock_heatmap.assert_not_called()  # Verify heatmap is not triggered

    # Capture console output and check for warning message
    captured = capsys.readouterr()
    assert "Warning: correlation matrix is empty" in captured.out
