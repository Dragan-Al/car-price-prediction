from pathlib import Path
import joblib
import pandas as pd

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor

from .data_cleaning import DEFAULT_DATA_PATH, clean_data
from .data_preprocessing import RANDOM_STATE, build_model_pipeline, split_features_target
from .model_evaluation import calculate_metrics

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "car_price_model.joblib"
REPORT_PATH = PROJECT_ROOT / "reports" / "model_comparison.csv"

def get_models(random_state=RANDOM_STATE):
    """Return the regression estimators to compare."""
    return {
        "Linear Regression": LinearRegression(),
        "Decision Tree": DecisionTreeRegressor(
            max_depth=18, min_samples_leaf=10, random_state=random_state
        ),
        "Random Forest": RandomForestRegressor(
            n_estimators=30,
            max_depth=16,
            min_samples_leaf=8,
            max_features=0.5,
            n_jobs=-1,
            random_state=random_state,
        ),
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=40,
            learning_rate=0.05,
            max_depth=3,
            random_state=random_state,
        ),
    }


def compare_models(
    data_path: Path | str = DEFAULT_DATA_PATH,
    model_path: Path | str = MODEL_PATH,
    report_path: Path | str = REPORT_PATH,
    test_size: float = 0.2,
    random_state: int = RANDOM_STATE,
):
    # Train four pipelines, compare metrics, and save the lowest-MAE model
    raw_df = pd.read_csv(data_path)
    cleaned_df = clean_data(raw_df)
    X, y = split_features_target(cleaned_df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    reference_year = pd.Timestamp.today().year

    results = []
    fitted_models = {}
    predictions_by_model = {}

    for name, estimator in get_models(random_state).items():
        print(f"Training {name}...")
        pipeline = build_model_pipeline(
            X_train, estimator, reference_year=reference_year
        )
        pipeline.fit(X_train, y_train)
        predictions = pipeline.predict(X_test)
        model_metrics = calculate_metrics(y_test, predictions)
        results.append({"Model": name, **model_metrics})
        fitted_models[name] = pipeline
        predictions_by_model[name] = predictions

    results_df = pd.DataFrame(results).sort_values("MAE", ascending=True).reset_index(drop=True)
    best_name = results_df.loc[0, "Model"]
    best_model = fitted_models[best_name]

    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_model, model_path)

    report_path = Path(report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(report_path, index=False)

    predictions_df = pd.DataFrame(
        {
            "Actual price (USD)": y_test.to_numpy(),
            "Predicted price (USD)": predictions_by_model[best_name],
        }
    )
    predictions_df["Absolute error (USD)"] = (
        predictions_df["Actual price (USD)"] - predictions_df["Predicted price (USD)"]
    ).abs()

    return results_df, best_name, best_model, predictions_df


if __name__ == "__main__":
    results, best_name, _, predictions = compare_models()
    print("\nModel comparison (sorted by MAE):")
    print(results.to_string(index=False, float_format=lambda value: f"{value:,.3f}"))
    print(f"\nSelected model: {best_name}")
    print(f"Saved selected model to: {MODEL_PATH}")
    print(f"Saved comparison report to: {REPORT_PATH}")
    print("\nExample predictions:")
    print(predictions.head(10).round(2).to_string(index=False))
