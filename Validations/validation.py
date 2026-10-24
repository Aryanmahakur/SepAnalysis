
import os
from datetime import datetime
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    average_precision_score,
    classification_report
)


def save_validation(y_test, y_pred, y_prob):

    os.makedirs("Validations/Files", exist_ok=True)

    file_path = os.path.join(
        "Validations",
        "Files",
        "validation_results.txt"
    )

    with open(file_path, "a", encoding="utf-8") as file:
        file.write("\n" + "=" * 60 + "\n")
        file.write(
            f"Validation Time: "
            f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        )
        file.write("=" * 60 + "\n")

        file.write(
            f"Accuracy: {accuracy_score(y_test, y_pred):.6f}\n"
        )
        file.write(
            f"Precision: "
            f"{precision_score(y_test, y_pred, zero_division=0):.6f}\n"
        )
        file.write(
            f"Recall: {recall_score(y_test, y_pred, zero_division=0):.6f}\n"
        )
        file.write(
            f"F1 Score: {f1_score(y_test, y_pred, zero_division=0):.6f}\n"
        )
        file.write(
            f"PR-AUC (Average Precision): "
            f"{average_precision_score(y_test, y_prob):.6f}\n"
        )

        file.write("\nClassification Report:\n")
        file.write(
            classification_report(y_test, y_pred, zero_division=0)
        )
        file.write("\n")

    print(f"Validation saved to: {file_path}")
a