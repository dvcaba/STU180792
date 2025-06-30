import pytest  # Pytest for running test suites programmatically
import sys     # Sys for accessing command-line arguments

# -------------------------------
# Individual test runners
# -------------------------------

def run_data_ingestion():
    """
    Run the data ingestion test suite.
    """
    return pytest.main(["tests/test_data_ingestion.py", "-v", "--tb=short"])

def run_feature_engineering():
    """
    Run the feature engineering test suite.
    """
    return pytest.main(["tests/test_feature_engineering.py", "-v", "--tb=short"])

def run_model_defaults():
    """
    Run tests that validate default parameters in model classes.
    """
    return pytest.main(["tests/test_model_defaults.py", "-v", "--tb=short"])

def run_pipeline_integration():
    """
    Run the full pipeline integration test.
    """
    return pytest.main(["tests/test_pipeline_integration.py", "-v", "--tb=short"])

def run_prediction_output():
    """
    Run tests that validate prediction output structure and integrity.
    """
    return pytest.main(["tests/test_prediction_output.py", "-v", "--tb=short"])

def run_stress_periods():
    """
    Run stress testing of the model on high-volatility periods.
    """
    return pytest.main(["tests/test_stress_periods.py", "-v", "--tb=short"])

def run_visual_diagnostics():
    """
    Run tests for visualizations (assumes presence of relevant visual diagnostics file).
    """
    return pytest.main(["tests/test_visual_diagnostics.py", "-v", "--tb=short"])

def run_all():
    """
    Run all test suites in the `tests/` directory.
    """
    return pytest.main(["tests", "-v", "--tb=short"])


# -------------------------------
# Export functions for external import
# This allows using `from test_version2 import run_pipeline_integration` etc.
# -------------------------------

__all__ = [
    name
    for name, obj in globals().items()
    if name.startswith("run_") and callable(obj)
]

# -------------------------------
# CLI usage
# If this file is run directly, it allows the user to execute:
# python test_version2.py run_pipeline_integration
# -------------------------------

if __name__ == "__main__":
    if len(sys.argv) > 1:
        func_name = sys.argv[1]  # Get the function name passed as argument
        func = globals().get(func_name)
        if callable(func):
            # Execute the selected function
            sys.exit(func())
        else:
            # If the function name is invalid, show help
            print(f"Unknown function: {func_name}")
            print("Available functions:")
            for name in sorted(n for n in globals() if n.startswith("run_")):
                print(f"  {name}")
            sys.exit(1)
    else:
        # If no argument is provided, show usage
        print("Usage: python tests/test_version2.py <function>")
        print("Available functions:")
        for name in sorted(n for n in globals() if n.startswith("run_")):
            print(f"  {name}")
        sys.exit(1)

# Optional default behavior: run all tests if this script is executed without arguments (can be commented if undesired)
run_all()
