from pathlib import Path
import pandas as pd
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "cars.csv"

COLUMN_RENAME_MAP = {
    "priceUSD": "price_usd",
    "mileage(kilometers)": "mileage_km",
    "volume(cm3)": "volume_cm3"
}

NUMERIC_COLUMNS = ["price_usd", "year", "mileage_km", "volume_cm3"]
FEATURE_NUMERIC_COLUMNS = ["year", "mileage_km", "volume_cm3"]
CATEGORICAL_COLUMNS = [
    "make",
    "model",
    "condition",
    "fuel_type",
    "color",
    "transmission",
    "drive_unit",
    "segment"
]

MISSING_TEXT_VALUES = {"", " ", "nan", "none", "null", "na", "n/a"}


def _standardize_columns_and_text(df: pd.DataFrame) -> pd.DataFrame:
    # Apply shared column-name and text normalization to a copy.
    cleaned = df.copy().rename(columns=COLUMN_RENAME_MAP)

    for column in cleaned.select_dtypes(include=["object", "string"]).columns:
        values = cleaned[column].astype("object").map(
            lambda value: value.strip() if isinstance(value, str) else value
        )

        values = values.map(
            lambda value: np.nan 
            if isinstance(value, str) and value.lower() in MISSING_TEXT_VALUES
            else value
        )

        cleaned[column] = values.map(
            lambda value: value.lower() if isinstance(value, str) else value
        )

    return cleaned


def _validate_numeric_features(df: pd.DataFrame) -> pd.DataFrame:
    # Convert numeric feature columns and mark impossible values as missing

    df = df.copy()

    for column in FEATURE_NUMERIC_COLUMNS:
        if column in df.columns:
            df[column] = pd.to_numeric(
                df[column],
                errors="coerce"
            )

    current_year = pd.Timestamp.today().year

    if "year" in df.columns:
        invalid_year = (
            (df["year"] < 1886)
            | (df["year"] > current_year)
        )
        df.loc[invalid_year, "year"] = np.nan

    if "mileage_km" in df.columns:
        df.loc[df["mileage_km"] < 0, "mileage_km"] = np.nan

    if "volume_cm3" in df.columns:
        df.loc[df["volume_cm3"] <= 0, "volume_cm3"] = np.nan

    return df

def clean_data (df: pd.DataFrame) -> pd.DataFrame:
    """Return a cleaned training dataset with a valid, positive target.
    Impossible numeric feature values become missing values so they can be
    imputed by the preprocessing pipeline. Complete duplicates and rows
    without a valid target are removed.
    """
    cleaned = _standardize_columns_and_text(df)

    for column in NUMERIC_COLUMNS:
        if column in cleaned.columns:
            cleaned[column] = pd.to_numeric(cleaned[column], errors="coerce")

    requared_columns = set (NUMERIC_COLUMNS + CATEGORICAL_COLUMNS)
    missing_columns = requared_columns.difference(cleaned.columns)
    if missing_columns:
        raise ValueError(f"Requared columns are missing: {sorted(missing_columns)}")

    rows_before = len(cleaned)
    cleaned = cleaned.drop_duplicates().copy()
    duplicates_removed = rows_before - len(cleaned)

    cleaned = cleaned.loc[cleaned["price_usd"].notna() & (cleaned["price_usd"] > 0)].copy()
    cleaned = _validate_numeric_features(cleaned)
    cleaned = cleaned.reset_index(drop=True)
    cleaned.attrs["duplicates_removed"] = duplicates_removed
    return cleaned

def prepare_prediction_features(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize new car records for prediction; the target is not required.
    Accepts either original dataset column names or already standardized names.
    Any supplied target column is removed before returning the feature matrix.
    """

    features = _standardize_columns_and_text(df)
    if "price_usd" in features.columns:
        features= features.drop(columns=["price_usd"])

    features = _validate_numeric_features(features)
    required_columns = set(FEATURE_NUMERIC_COLUMNS + CATEGORICAL_COLUMNS)
    missing_columns = required_columns.difference(features.columns)
    if missing_columns:
        raise ValueError(f"Requared input feature columns are missing: {sorted(missing_columns)}")

    return features[FEATURE_NUMERIC_COLUMNS + CATEGORICAL_COLUMNS].copy()

def load_and_clean_data(data_path: Path | str = DEFAULT_DATA_PATH) -> pd.DataFrame:
    # Load the raw CSV file and return the cleaned dataset
    raw_df = pd.read_csv(data_path)
    return clean_data(raw_df)

if __name__ == "__main__":
    raw_df = pd.read_csv(DEFAULT_DATA_PATH)
    cleaned_df = clean_data(raw_df)
    output_path = PROJECT_ROOT / "data" / "processed" / "cars_cleaned.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cleaned_df.to_csv(output_path, index=False)

    print(f"Raw rows: {len(raw_df)}")
    print(f"Cleaned rows: {len(cleaned_df)}")
    print(f"Duplicates raws removed: {cleaned_df.attrs.get('duplicates_removed', 'see_source')}")
    print(f"Saved cleaned dataset to: {output_path.relative_to(PROJECT_ROOT)}")


