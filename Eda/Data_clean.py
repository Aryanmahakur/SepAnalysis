
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer

# ============================================================
# 1. PROJECT CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from Datasets.Data1 import loadData

OUTPUT_FILE = PROJECT_ROOT / "CleanedDataSet.csv"
TEST_SIZE = 0.20
RANDOM_STATE = 42


# ============================================================
# 2. LOAD AND VALIDATE DATA
# ============================================================

def clean_dataset():

    df = loadData().copy()

    print("Original shape:", df.shape)

    required = ["Patient_ID", "Hour", "SepsisLabel"]

    for col in required:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    if df["Patient_ID"].isna().any():
        raise ValueError("Patient_ID contains missing values.")

    if df["SepsisLabel"].isna().any():
        raise ValueError("SepsisLabel contains missing values.")

    if not set(df["SepsisLabel"].unique()).issubset({0, 1}):
        raise ValueError("SepsisLabel must contain only 0 and 1.")

    if df.duplicated(["Patient_ID", "Hour"]).any():
        raise ValueError("Duplicate patient-hour records found.")

    # Sort chronologically within each patient.
    df = df.sort_values(
        ["Patient_ID", "Hour"]
    ).reset_index(drop=True)

    # Remove a saved index only if it is just a redundant index.
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])

    # Treat infinite numeric values as missing.
    numeric_cols = df.select_dtypes(
        include=np.number
    ).columns

    df[numeric_cols] = df[numeric_cols].replace(
        [np.inf, -np.inf], np.nan
    )


    # ========================================================
    # 3. SPLIT PATIENTS BEFORE IMPUTATION
    # ========================================================

    patient_ids = df["Patient_ID"].unique()

    train_ids, test_ids = train_test_split(
        patient_ids,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE
    )

    train_mask = df["Patient_ID"].isin(train_ids)
    test_mask = ~train_mask

    df["DatasetSplit"] = np.where(
        train_mask, "train", "test"
    )

    print("Training patients:", len(train_ids))
    print("Testing patients:", len(test_ids))


    # ========================================================
    # 4. IDENTIFY MODEL INPUT FEATURES
    # ========================================================

    excluded = [
        "Patient_ID",
        "SepsisLabel",
        "DatasetSplit"
    ]

    feature_cols = [
        col for col in df.columns
        if col not in excluded
    ]

    # These columns are binary/unit indicators, not
    # continuous clinical measurements.
    categorical_cols = [
        col for col in ["Gender", "Unit1", "Unit2"]
        if col in feature_cols
    ]

    numeric_cols = [
        col for col in feature_cols
        if col not in categorical_cols
    ]

    # Remove features entirely missing in training data.
    all_missing = [
        col for col in feature_cols
        if df.loc[train_mask, col].isna().all()
    ]

    if all_missing:
        print("Removing features entirely missing in training:")
        print(all_missing)

        df = df.drop(columns=all_missing)

        feature_cols = [
            col for col in feature_cols
            if col not in all_missing
        ]

        categorical_cols = [
            col for col in categorical_cols
            if col not in all_missing
        ]

        numeric_cols = [
            col for col in numeric_cols
            if col not in all_missing
        ]


    # ========================================================
    # 5. IMPUTE NUMERIC FEATURES USING TRAINING DATA ONLY
    # ========================================================

    if numeric_cols:
        numeric_imputer = SimpleImputer(
            strategy="median",
            add_indicator=True
        )

        train_values = numeric_imputer.fit_transform(
            df.loc[train_mask, numeric_cols]
        )

        all_values = numeric_imputer.transform(
            df[numeric_cols]
        )

        # Add missingness indicators for columns that
        # actually had missing values in training.
        indicator_cols = [
            f"{numeric_cols[i]}_was_missing"
            for i in numeric_imputer.indicator_.features_
        ]

        df[numeric_cols] = all_values[
            :, :len(numeric_cols)
        ]

        if indicator_cols:
            indicator_values = all_values[
                :, len(numeric_cols):
            ]

            for i, name in enumerate(indicator_cols):
                df[name] = indicator_values[:, i].astype(int)


    # ========================================================
    # 6. IMPUTE BINARY/UNIT FEATURES
    # ========================================================

    if categorical_cols:
        categorical_imputer = SimpleImputer(
            strategy="most_frequent",
            add_indicator=True
        )

        categorical_values = categorical_imputer.fit_transform(
            df.loc[train_mask, categorical_cols]
        )

        all_categorical_values = categorical_imputer.transform(
            df[categorical_cols]
        )

        df[categorical_cols] = all_categorical_values[
            :, :len(categorical_cols)
        ]

        for i in categorical_imputer.indicator_.features_:
            name = f"{categorical_cols[i]}_was_missing"

            df[name] = all_categorical_values[
                :, len(categorical_cols) +
                list(categorical_imputer.indicator_.features_).index(i)
            ].astype(int)


    # ========================================================
    # 7. VERIFY AND SAVE ONE DATASET
    # ========================================================

    print("\nRemaining missing values:")
    remaining = df.isna().sum()
    print(remaining[remaining > 0])

    print("\nFinal shape:", df.shape)
    print("Target distribution:")
    print(df["SepsisLabel"].value_counts())

    if df.isna().any().any():
        raise ValueError(
            "Missing values remain. Inspect the columns above."
        )

    df.to_csv(OUTPUT_FILE, index=False)

    print("\nSaved cleaned dataset to:")
    print(OUTPUT_FILE)


# ============================================================
# 8. RUN
# ============================================================

if __name__ == "__main__":
    clean_dataset()
