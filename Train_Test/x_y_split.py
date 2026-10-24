
import pandas as pd
from sklearn.model_selection import train_test_split

def split_data():
    df = pd.read_csv(
        r"C:\Users\ARYAN MAHAKUR\Desktop\VS CODE\SepAnalysis\CleanedDataSet.csv"
    )

    y = df["SepsisLabel"]

    X = df.drop(columns=["SepsisLabel", "Patient_ID"])

    x_train, x_test, y_train, y_test = train_test_split(
        X, y, random_state=42, test_size=0.2
    )

    return x_train, x_test, y_train, y_test
a