import os
import sys
import pandas as pd
from sklearn.preprocessing import StandardScaler


def scale_numerical_features(processed_file_path):
    print("--- Step 3: Feature Scaling ---")

    # 1. Check if the file exists
    if not os.path.exists(processed_file_path):
        raise FileNotFoundError(
            f"Processed data file not found at {processed_file_path}."
        )

    # 2. Read the current dataset state
    df = pd.read_csv(processed_file_path)

    # 3. Define continuous numerical columns that need scaling
    # We target columns that have actual continuous physical measurements
    numerical_cols = ["age", "trestbps", "chol", "thalch", "oldpeak"]

    # Filter only columns that actually exist in the dataframe to prevent errors
    numerical_cols = [col for col in numerical_cols if col in df.columns]

    if len(numerical_cols) == 0:
        print("ℹ️ No numerical columns found for scaling.")
    else:
        print(f"⚖️ Scaling continuous columns: {numerical_cols}")
        scaler = StandardScaler()

        # Apply StandardScaler to the selected columns
        df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

    print("✅ Feature scaling completed successfully.")

    # 4. Save and overwrite the same file in data/processed
    df.to_csv(processed_file_path, index=False)
    print(f"✨ Safe copy updated with scaled features at: {processed_file_path}\n")


if __name__ == "__main__":
    PROCESSED_DATA_PATH = os.path.join("data", "processed", "dataset.csv")
    scale_numerical_features(PROCESSED_DATA_PATH)