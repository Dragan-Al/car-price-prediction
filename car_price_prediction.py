from src.model_evaluation import evaluate_saved_model
from src.data_cleaning import DEFAULT_DATA_PATH, clean_data
from src.model_training import train_baseline_model
from src.model_comparison import compare_models
import pandas as pd

def main():
    print("Step 1: Cleaning the raw set ")
    raw_df = pd.read_csv(DEFAULT_DATA_PATH)
    cleaned_df = clean_data(raw_df)
    print(f"Raw rows: {len(raw_df)}")
    print(f"Rows after cleaning: {len(cleaned_df)}")

    print("\nStep 2: Training the Linear Regression baseline")
    train_baseline_model()
    baseline_metrics, baseline_results = evaluate_saved_model()
    for metric, value in baseline_metrics.items():
        print(f"{metric}: {value:,.4f}")

    print("\nStep 3: comparing regression models")
    results, best_name, _, predictions = compare_models()
    print(results.to_string(index=False, float_format=lambda value: f"{value:,.3f}"))
    print(f"\nBest model by MAE: {best_name}")
    print("\nExample predictions:")
    print(predictions.head(10).round(2).to_string(index=False))
    print("\nWorkflow complete. See models/ and reports/ for saved results")

if __name__ == "__main__":
    main()
