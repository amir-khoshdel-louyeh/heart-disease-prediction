import os
import sys
import pandas as pd


def encode_categorical_data(processed_file_path):
    print("--- Step 2: Categorical Encoding ---")

    # 1. Check if the file exists
    if not os.path.exists(processed_file_path):
        raise FileNotFoundError(
            f"Processed data file not found at {processed_file_path}."
        )

    # 2. Read the current dataset state
    df = pd.read_csv(processed_file_path)

    # 3. Identify categorical columns that are text-based
    # Based on your data: 'sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal'
    # We filter only columns that actually contain text/object types to prevent errors
    categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()

    # Ensure 'target' is not treated as a dummy variable if it's text (it shouldn't be)
    if "target" in categorical_cols:
        categorical_cols.remove("target")

    if len(categorical_cols) == 0:
        print("ℹ️ No text-based categorical columns found or already encoded.")
    else:
        print(f"🔤 Encoding columns: {categorical_cols}")
        # Apply One-Hot Encoding and drop the first category to avoid multicollinearity
        df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

        # Convert True/False boolean outputs from get_dummies into 1 and 0
        df = df.astype(int, errors="ignore")

    print("✅ Categorical encoding completed successfully.")

    # 4. Save and overwrite the same file in data/processed
    df.to_csv(processed_file_path, index=False)
    print(f"✨ Safe copy updated with encodings at: {processed_file_path}\n")


if __name__ == "__main__":
    PROCESSED_DATA_PATH = os.path.join("data", "processed", "dataset.csv")
    encode_categorical_data(PROCESSED_DATA_PATH)