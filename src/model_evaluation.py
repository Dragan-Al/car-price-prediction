from pathlib import Path
import joblib
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from data_cleaning import DEFAULT_DATA_PATH, clean_data
from data_preprocessing import TARGET_COLUMN, RANDOM_STATE, split_features_target

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "linear_regression_model.joblib"

def calculate_metrics(y_true, y_pred):
    # Return common regression metrics as a dictionary
    mse = mean_squared_error(y_true, y_pred)
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "MSE": mse,
        "RMSE": mse ** 0.5,
        "R2": r2_score(y_true, y_pred),
    }


def make_prediction_examples(y_true, y_pred, n=10):
    # Return a readable table of actual prices, predictions, and absolute errors
    examples = pd.DataFrame(
        {
            "Actual price (USD)": list(y_true),
            "Predicted price (USD)": list(y_pred),
        }
    )
    examples["Absolute error (USD)"] = (
        examples["Actual price (USD)"] - examples["Predicted price (USD)"]
    ).abs()
    return examples.head(n).round(2)


def evaluate_saved_model(
    data_path: Path | str = DEFAULT_DATA_PATH,
    model_path: Path | str = DEFAULT_MODEL_PATH,
    test_size: float = 0.2,
    random_state: int = RANDOM_STATE,
):
    # Evaluate a saved pipeline on the reproducible held-out test split
    raw_df = pd.read_csv(data_path)
    df = clean_data(raw_df)
    X, y = split_features_target(df)
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    model = joblib.load(model_path)
    predictions = model.predict(X_test)
    metrics = calculate_metrics(y_test, predictions)
    examples = make_prediction_examples(y_test, predictions)
    return metrics, examples


if __name__ == "__main__":
    metrics, examples = evaluate_saved_model()
    print("Evaluation on the held-out test set:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:,.4f}")
    print("\nExample predictions:")
    print(examples.to_string(index=False))
