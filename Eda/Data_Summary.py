
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent))

from Datasets.Data1 import loadData


df = loadData()

print("Dataset Overview")
print(df.shape)

print("Column Names")
print(df.columns.tolist())

print("First 5 Rows")
print(df.head())

print("Last 5 Rows")
print(df.tail())

print("Random 5 Rows")
print(df.sample(min(5, len(df))))

print("Dataset Information")
df.info()

print("Data Types")
print(df.dtypes)

print("Statistical Summary")
print(df.describe(include="all").T)



print("Missing Values Percentage")
print((df.isnull().mean() * 100).round(2))

print("Total Missing Values")
print(df.isnull().sum().sum())

print("Duplicate Rows")
print(df.duplicated().sum())

print("Unique Values Per Column")
print(df.nunique())

print("Numeric Columns")
print(df.select_dtypes(include=np.number).columns.tolist())

print("Categorical Columns")
print(df.select_dtypes(include=["object", "category"]).columns.tolist())

print("Sepsis Label Distribution")
print(df["SepsisLabel"].value_counts(dropna=False))

print("Sepsis Label Percentage")
print(
    (df["SepsisLabel"].value_counts(normalize=True, dropna=False) * 100).round(2)
)

print("Unique Patients")
if "Patient_ID" in df.columns:
    print(df["Patient_ID"].nunique())

print("Dataset Memory Usage")
print(f"{df.memory_usage(deep=True).sum() / (1024 ** 2):.2f} MB")
