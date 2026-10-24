import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.append(str(Path(__file__).resolve().parent.parent))
from Datasets.Data1 import loadData


# ============================================================
# 1. SEPSIS LABEL DISTRIBUTION
# ============================================================

def labeldist():
    df = loadData()

    label_counts = df["SepsisLabel"].value_counts().sort_index()

    plt.figure(figsize=(7, 5))
    bars = plt.bar(label_counts.index, label_counts.values)

    for bar in bars:
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{int(bar.get_height()):,}",
            ha="center",
            va="bottom"
        )

    plt.xlabel("Sepsis Label")
    plt.xticks([0, 1])
    plt.ylabel("Number of Records")
    plt.title("Sepsis Label Distribution")
    plt.tight_layout()
    plt.show()


# ============================================================
# 2. MISSING VALUE PERCENTAGE
# ============================================================

def missing_value_plot():
    df = loadData()

    missing_percentage = (
        df.isnull().sum() / len(df)
    ) * 100

    missing_percentage = missing_percentage.sort_values(
        ascending=False
    )

    plt.figure(figsize=(12, 14))

    bars = plt.barh(
        missing_percentage.index,
        missing_percentage.values
    )

    for bar, percentage in zip(
        bars, missing_percentage.values
    ):
        if percentage > 0:
            plt.text(
                percentage + 0.5,
                bar.get_y() + bar.get_height() / 2,
                f"{percentage:.2f}%",
                va="center",
                fontsize=8
            )

    plt.xlabel("Missing Values (%)")
    plt.ylabel("Columns")
    plt.title("Missing Value Percentage for All Columns")
    plt.xlim(0, 110)
    plt.tight_layout()
    plt.show()


# ============================================================
# 3. HISTOGRAMS OF IMPORTANT VITAL SIGNS
# ============================================================

def vital_signs_histograms():
    df = loadData()

    columns = [
        "HR", "O2Sat", "Temp",
        "SBP", "MAP", "Resp"
    ]

    df[columns].hist(
        bins=40,
        figsize=(14, 9),
        edgecolor="black"
    )

    plt.suptitle(
        "Distribution of Important Vital Signs",
        fontsize=16
    )

    plt.tight_layout()
    plt.show()


# ============================================================
# 4. BOX PLOTS OF IMPORTANT VITAL SIGNS
# ============================================================

def vital_signs_boxplots():
    df = loadData()

    columns = [
        "HR", "O2Sat", "Temp",
        "SBP", "MAP", "Resp"
    ]

    plt.figure(figsize=(12, 7))

    df[columns].boxplot()

    plt.title("Box Plots of Important Vital Signs")
    plt.xlabel("Vital Signs")
    plt.ylabel("Measured Value")
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.show()


# ============================================================
# 5. CORRELATION HEATMAP
# ============================================================

def correlation_heatmap():
    df = loadData()

    columns = [
        "HR", "O2Sat", "Temp",
        "SBP", "MAP", "DBP", "Resp"
    ]

    correlation = df[columns].corr()

    plt.figure(figsize=(9, 7))

    image = plt.imshow(
        correlation,
        cmap="coolwarm",
        vmin=-1,
        vmax=1
    )

    plt.colorbar(image, label="Correlation")

    plt.xticks(
        range(len(columns)),
        columns,
        rotation=45
    )

    plt.yticks(range(len(columns)), columns)

    for i in range(len(columns)):
        for j in range(len(columns)):
            plt.text(
                j, i,
                f"{correlation.iloc[i, j]:.2f}",
                ha="center",
                va="center",
                fontsize=8
            )

    plt.title("Correlation Heatmap of Vital Signs")
    plt.tight_layout()
    plt.show()


# ============================================================
# 6. VITAL SIGNS BY SEPSIS LABEL
# ============================================================

def vital_signs_by_sepsis():
    df = loadData()

    columns = [
        "HR", "O2Sat", "Temp",
        "SBP", "Resp"
    ]

    fig, axes = plt.subplots(
        2, 3,
        figsize=(14, 9)
    )

    axes = axes.flatten()

    for i, column in enumerate(columns):
        data = [
            df.loc[
                df["SepsisLabel"] == label, column
            ].dropna().sample(
                n=min(
                    5000,
                    df.loc[
                        df["SepsisLabel"] == label,
                        column
                    ].notna().sum()
                ),
                random_state=42
            )
            for label in [0, 1]
        ]

        axes[i].boxplot(
            data,
            tick_labels=["No Sepsis (0)", "Sepsis (1)"]
        )

        axes[i].set_title(column)
        axes[i].set_ylabel("Measured Value")
        axes[i].tick_params(
            axis="x",
            labelrotation=15
        )

    axes[5].axis("off")

    fig.suptitle(
        "Vital Signs Compared by Sepsis Label",
        fontsize=16
    )

    plt.tight_layout()
    plt.show()


# ============================================================
# 7. MISSING VALUE COUNTS
# ============================================================

def missing_value_counts():
    df = loadData()

    missing_counts = df.isnull().sum()
    missing_counts = missing_counts[
        missing_counts > 0
    ].sort_values(ascending=False)

    plt.figure(figsize=(12, 10))

    bars = plt.barh(
        missing_counts.index,
        missing_counts.values
    )

    plt.gca().invert_yaxis()

    for bar, count in zip(
        bars, missing_counts.values
    ):
        plt.text(
            bar.get_width(),
            bar.get_y() + bar.get_height() / 2,
            f" {count:,}",
            va="center",
            fontsize=8
        )

    plt.xlabel("Number of Missing Values")
    plt.ylabel("Columns")
    plt.title("Missing Value Counts by Column")
    plt.tight_layout()
    plt.show()


# ============================================================
# 8. CALL ALL FUNCTIONS ONE BY ONE
# ============================================================

if __name__ == "__main__":

    labeldist()

    missing_value_plot()

    vital_signs_histograms()

    vital_signs_boxplots()

    correlation_heatmap()

    vital_signs_by_sepsis()

    missing_value_counts()