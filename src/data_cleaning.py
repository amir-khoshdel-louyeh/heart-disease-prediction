import os
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

pd.set_option("future.no_silent_downcasting", True)

def load_and_clean_data(processed_file_path, output_dir):
    print("--- Step 1: Loading, Cleaning, and Splitting Data ---")

    if not os.path.exists(processed_file_path):
        raise FileNotFoundError(
            f"Processed data copy not found at {processed_file_path}. Please run main.py first."
        )

    df = pd.read_csv(processed_file_path)

    # 1. Structural Cleaning
    if "num" in df.columns:
        df["target"] = df["num"].apply(lambda x: 1 if x > 0 else 0)
        df = df.drop(columns=["num"])
    elif "target" not in df.columns and "num" not in df.columns:
        df["target"] = df.iloc[:, -1].apply(lambda x: 1 if x > 0 else 0)

    cols_to_drop = ["id", "dataset"]
    df = df.drop(columns=[col for col in cols_to_drop if col in df.columns])

    # 2. Split Data FIRST (to avoid data leakage)
    X = df.drop(columns=["target"])
    y = df["target"]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # 3. Impute Missing Values based ONLY on X_train
    num_cols = ["trestbps", "chol", "thalch", "oldpeak"]
    cat_cols = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]

    # Impute numericals with median
    for col in num_cols:
        if col in X_train.columns:
            median_val = X_train[col].median()
            X_train[col] = X_train[col].fillna(median_val)
            X_test[col] = X_test[col].fillna(median_val)

    # Impute categoricals with mode
    for col in cat_cols:
        if col in X_train.columns:
            mode_series = X_train[col].mode()
            if not mode_series.empty:
                mode_val = mode_series[0]
                X_train[col] = X_train[col].fillna(mode_val)
                X_test[col] = X_test[col].fillna(mode_val)

    print("✅ Data cleaning and splitting completed successfully without data leakage.")

    # 4. Save splits to disk
    X_train.to_csv(os.path.join(output_dir, "X_train.csv"), index=False)
    X_test.to_csv(os.path.join(output_dir, "X_test.csv"), index=False)
    y_train.to_csv(os.path.join(output_dir, "y_train.csv"), index=False)
    y_test.to_csv(os.path.join(output_dir, "y_test.csv"), index=False)
    
    print(f"✨ Safe splits created at: {output_dir}\n")

if __name__ == "__main__":
    PROCESSED_DATA_PATH = os.path.join("data", "processed", "dataset.csv")
    OUTPUT_DIR = os.path.join("data", "processed")
    load_and_clean_data(PROCESSED_DATA_PATH, OUTPUT_DIR)