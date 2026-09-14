"""Preprocessing utilities for the Personalized Diet Recommendation project."""
from pathlib import Path
import numpy as np
import pandas as pd

TARGET = "Recommended_Meal_Plan"
ID_COL = "Patient_ID"
LEAKAGE_COLS = ["Recommended_Calories", "Recommended_Protein",
                "Recommended_Carbs", "Recommended_Fats"]

def load_dataset(csv_path):
    return pd.read_csv(csv_path)

def normalize_categories(df, categorical_cols):
    out = df.copy()
    for col in categorical_cols:
        out[col] = out[col].fillna("Unknown/Not Reported").astype(str).str.strip()
    return out

def fit_imputer(df, numeric_cols, categorical_cols):
    params = {"numeric": {}, "categorical": {}}
    for col in numeric_cols:
        params["numeric"][col] = float(df[col].median())
    for col in categorical_cols:
        values = df[col].fillna("Unknown/Not Reported").astype(str).str.strip()
        params["categorical"][col] = sorted(values.unique().tolist())
    return params

def transform(df, params, numeric_cols, categorical_cols):
    out = pd.DataFrame(index=df.index)
    for col in numeric_cols:
        out[col] = pd.to_numeric(df[col], errors="coerce").fillna(params["numeric"][col])
    for col in categorical_cols:
        s = df[col].fillna("Unknown/Not Reported").astype(str).str.strip()
        for category in params["categorical"][col]:
            out[f"{col}_{category}"] = (s == category).astype(int)
    return out

def prepare_model_data(df):
    """Return modelling candidates after removing ID and target-derived outputs."""
    predictors = [c for c in df.columns if c not in [TARGET, ID_COL] + LEAKAGE_COLS]
    X = df[predictors].copy()
    y = df[TARGET].copy()
    return X, y
