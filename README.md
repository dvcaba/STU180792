# ML Pipeline

This project builds a classification pipeline for stock data. It downloads price information, engineers technical indicators, performs exploratory data analysis and preprocessing and finally trains several machine learning models.

## Running the pipeline

1. Install the requirements: `pip install -r requirements.txt`.
2. Execute the main script: `python main.py`.
3. Configure pipeline options in `config.yaml` under the `pipeline` section.
   This includes the tickers to download, the forecast horizon for target
   creation and all model hyperparameters.

The pipeline also includes an optional **random search** for hyperparameter
optimization. Use `RandomSearchWrapper` with a `param_distributions` dictionary
and set `n_iter` to the number of sampled configurations you want to try. All
parameter ranges for grid and random search are stored in `config.yaml`.

## Workflow overview

The training workflow is composed of two phases:

1. **Baseline evaluation** – After preprocessing the data a baseline `LogisticRegression` model is trained with default parameters. The resulting metrics are stored in `results/metrics/baseline_LogisticRegression_metrics.csv`.
2. **Hyperparameter tuning** – Grid search is run for each supported model (Logistic Regression, Random Forest, XGBoost and SVM) using the ranges defined in `config.yaml`. Metrics for these tuned models are saved in their respective files within `results/metrics`.

After grid search the script compares the best results and saves the top performing model in `results/best_model`.

## Random search

`RandomSearchWrapper` provides a quicker alternative to grid search. Pass a
`param_distributions` dictionary with ranges of values and specify `n_iter` to
control how many random combinations are evaluated. The wrapper saves its
results in the same directories used by grid search. These distributions are
defined alongside the grid search parameters in `config.yaml`.

## Running tests

Use `tests/test_version2.py` to execute individual test modules with pytest. The script exposes a function for each test file. For example, to run only the data ingestion tests:

```
python tests/test_version2.py run_data_ingestion
```

Omit the argument or provide an invalid name to see the list of available functions.
