from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
from .feature_engineering import add_features
import pandas as pd

TARGET_COLUMN = "price_usd"
RANDOM_STATE = 42

def split_features_target(df):
    # Split a cleaned dataset into input features (X) and target (y).
    if TARGET_COLUMN not in df.columns:
        raise ValueError (f"Target column '{TARGET_COLUMN}' was not found")
    X = df.drop(columns=[TARGET_COLUMN]).copy()
    y = df[TARGET_COLUMN].copy()
    return X, y

def get_column_types(X):
    # Return explicit numeric and categorical feature column lists.
    numeric_features = X.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical_features = X.select_dtypes(include=["object", "string", "category"]).columns.tolist()
    return numeric_features, categorical_features

def build_preprocesor(X):
    # Create a ColumnTransformer with imputation, scaling, and one-hot encoding.
    numeric_features, categorical_features = get_column_types(X)
    numeric_pipeline = Pipeline(
        steps=[
           ("imputer", SimpleImputer(strategy = "median")),
           ("scaler", StandardScaler())
        ]
    )

    categorical_pipeline = Pipeline(
            steps=[
               ("imputer", SimpleImputer(strategy = "most_frequent")),
               ("onehot", OneHotEncoder(handle_unknown = "ignore"))
            ]
        )

    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_features),
            ("categorical", categorical_pipeline, categorical_features)
        ],
        remainder="drop"
    )

def build_model_pipeline(X_train, estimator, reference_year=None):
    # Build a full pipeline that also applies feature engineering to raw X.
    if reference_year is None:
        reference_year = pd.Timestamp.today().year

    X_train_engineered = add_features(X_train, reference_year=reference_year)
    preprocesor = build_preprocesor(X_train_engineered)
    feature_transformer = FunctionTransformer(
        func=add_features,
        kw_args={"reference_year": reference_year},
        validate=False
    )

    return Pipeline(
        steps=[
            ("feature_engineering", feature_transformer),
            ("preprocesor", preprocesor),
            ("regressor", estimator)
        ]
    )