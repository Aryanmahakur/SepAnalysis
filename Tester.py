
from Train_Test.x_y_split import split_data
from Model_Training.Training import train_model
from Validations.validation import save_validation


def tester():
    print("\n" + "=" * 60)
    print("SEPSIS PREDICTION - MODEL TESTING PIPELINE")
    print("=" * 60)

    # 1. Load and split data
    print("\n[1/4] Loading and splitting dataset...")
    x_train, x_test, y_train, y_test = split_data()

    print(f"Training features shape: {x_train.shape}")
    print(f"Testing features shape:  {x_test.shape}")
    print(f"Training labels shape:   {y_train.shape}")
    print(f"Testing labels shape:    {y_test.shape}")
    print("Data splitting completed successfully.")

    # 2. Train model
    print("\n[2/4] Training LightGBM model and tuning hyperparameters...")
    model, x_test, y_test = train_model(
        x_train, x_test, y_train, y_test
    )
    print("Model training and hyperparameter tuning completed.")
    print("Best model received successfully.")

    # 3. Generate predictions
    print("\n[3/4] Generating predictions...")
    y_pred = model.predict(x_test)
    y_prob = model.predict_proba(x_test)[:, 1]

    print("Predictions generated successfully.")
    print(f"Total predictions: {len(y_pred)}")

    # 4. Validate and save results
    print("\n[4/4] Evaluating model and saving validation results...")
    save_validation(y_test, y_pred, y_prob)

    print("\n" + "=" * 60)
    print("PIPELINE COMPLETED SUCCESSFULLY!")
    print("=" * 60)


if __name__ == "__main__":
    tester()
