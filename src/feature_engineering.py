import numpy as np
import pandas as pd

def add_features(df: pd.DataFrame, reference_year: int | None = None) -> pd.DataFrame:
    '''Return a copy with useful derived car features added.
    reference_year is fixed when a model is trained and saved, so the same
    reference year is used at prediction time.'''
     
    features = df.copy()
    if reference_year is None:
        reference_year = pd.Timestamp.today().year
    requared_columns = {"year", "mileage_km", "volume_cm3"}
    missind_columns = requared_columns.difference(features.columns)
    if missind_columns:
        raise ValueError(f"Requarid columns are missing: {sorted(missind_columns)}")

    year = pd.to_numeric(features["year"], errors="coerce")
    mileage = pd.to_numeric(features["mileage_km"], errors="coerce")
    volume = pd.to_numeric(features["volume_cm3"], errors="coerce")

    features["car_age"] = (reference_year - year).where(year <= reference_year)
    positive_age = features ["car_age"].where(features["car_age"] > 0)
    features["mileage_per_year"] = mileage / positive_age
    features["engine_volume_liters"] = volume / 1000
    features["is_newer_car"] = np.where(features["car_age"].notna(), (features["car_age"] <= 5).astype(float), np.nan)
    features["is_high_mileage"] = np.where(mileage.notna(), (mileage >= 200_000).astype(float), np.nan)

    return features
    