from pathlib import Path
import joblib
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

from data_cleaning import DEFAULT_DATA_PATH, clean_data
from data_preprocessing import RANDOM_STATE, build_model_pipeline, split_features_target

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "linear_regression_model.joblib"


def train_baseline_model(
    data_path: Path | str = DEFAULT_DATA_PATH,
    model_path: Path | str = DEFAULT_MODEL_PATH,
    test_size: float = 0.2,
    random_state: int = RANDOM_STATE,
):
    """Train and save a Linear Regression baseline model."""
    cleaned_df = clean_data(pd.read_csv(data_path))
    X, y = split_features_target(cleaned_df)
    X_train, _, y_train, _ = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state
    )

    model = build_model_pipeline(X_train, LinearRegression(), reference_year=pd.Timestamp.today().year)
    model.fit(X_train, y_train)
   

    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)

    return model


if __name__ == "__main__":
    model = train_baseline_model()

    print("Linear Regression model trained successfully.")
    print(f"Saved model: {DEFAULT_MODEL_PATH}")