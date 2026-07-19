import os
import sys
import pandas as pd
from sklearn.preprocessing import StandardScaler

def scale_numerical_features(output_dir):
    print("--- Step 3: Feature Scaling ---")

    x_train_path = os.path.join(output_dir, "X_train.csv")
    x_test_path = os.path.join(output_dir, "X_test.csv")

    if not os.path.exists(x_train_path) or not os.path.exists(x_test_path):
        raise FileNotFoundError("Training or Testing splits not found.")

    X_train = pd.read_csv(x_train_path)
    X_test = pd.read_csv(x_test_path)

    # We target columns that have actual continuous physical measurements
    numerical_cols = ["age", "trestbps", "chol", "thalch", "oldpeak"]
    numerical_cols = [col for col in numerical_cols if col in X_train.columns]

    if len(numerical_cols) == 0:
        print("ℹ️ No numerical columns found for scaling.")
    else:
        print(f"⚖️ Scaling continuous columns: {numerical_cols}")
        scaler = StandardScaler()

        # Fit ONLY on training data
        scaler.fit(X_train[numerical_cols])

        # Transform both
        X_train[numerical_cols] = scaler.transform(X_train[numerical_cols])
        X_test[numerical_cols] = scaler.transform(X_test[numerical_cols])

    print("✅ Feature scaling completed successfully.")

    # Save back to disk
    X_train.to_csv(x_train_path, index=False)
    X_test.to_csv(x_test_path, index=False)
    print(f"✨ Splits updated with scaled features at: {output_dir}\n")

if __name__ == "__main__":
    OUTPUT_DIR = os.path.join("data", "processed")
    scale_numerical_features(OUTPUT_DIR)
