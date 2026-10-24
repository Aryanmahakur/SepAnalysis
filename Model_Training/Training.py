
import os
import joblib
from lightgbm import LGBMClassifier
from sklearn.model_selection import GridSearchCV


def train_model(x_train, x_test, y_train, y_test):

    model = LGBMClassifier(
        objective="binary",
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
        verbosity=-1
    )

    param_grid = {
        "n_estimators": [100, 300, 500],
        "learning_rate": [0.01, 0.05, 0.1],
        "num_leaves": [15, 31, 63],
        "max_depth": [-1, 10, 20],
        "min_child_samples": [20, 50, 100]
    }

    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        scoring="average_precision",
        cv=3,
        n_jobs=-1,
        verbose=1
    )

    grid_search.fit(x_train, y_train)

    print("Best Parameters:", grid_search.best_params_)
    print("Best CV Score:", grid_search.best_score_)

    best_model = grid_search.best_estimator_

    os.makedirs("PickleFiles", exist_ok=True)
    joblib.dump(best_model, "PickleFiles/best_sepsis_model.pkl")

    print("Model saved successfully!")

    return best_model, x_test, y_test
