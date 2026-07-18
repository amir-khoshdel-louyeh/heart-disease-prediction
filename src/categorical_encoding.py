import os
import sys
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

def encode_categorical_data(output_dir):
    print("--- Step 2: Categorical Encoding ---")

    x_train_path = os.path.join(output_dir, "X_train.csv")
    x_test_path = os.path.join(output_dir, "X_test.csv")

    if not os.path.exists(x_train_path) or not os.path.exists(x_test_path):
        raise FileNotFoundError("Training or Testing splits not found.")

    X_train = pd.read_csv(x_train_path)
    X_test = pd.read_csv(x_test_path)

    categorical_cols = X_train.select_dtypes(include=["object"]).columns.tolist()

    if len(categorical_cols) == 0:
        print("ℹ️ No text-based categorical columns found or already encoded.")
    else:
        print(f"🔤 Encoding columns: {categorical_cols}")
        
        # We must keep indices aligned after encoding
        # OneHotEncoder handles unknown categories in the test set
        encoder = OneHotEncoder(drop='first', handle_unknown='ignore', sparse_output=False)
        
        # Fit ONLY on training data
        encoder.fit(X_train[categorical_cols])
        
        # Transform both
        encoded_train = encoder.transform(X_train[categorical_cols])
        encoded_test = encoder.transform(X_test[categorical_cols])
        
        # Get feature names
        new_cols = encoder.get_feature_names_out(categorical_cols)
        
        # Replace old columns with new ones
        X_train_encoded = pd.DataFrame(encoded_train, columns=new_cols, index=X_train.index)
        X_test_encoded = pd.DataFrame(encoded_test, columns=new_cols, index=X_test.index)
        
        X_train = pd.concat([X_train.drop(columns=categorical_cols), X_train_encoded], axis=1)
        X_test = pd.concat([X_test.drop(columns=categorical_cols), X_test_encoded], axis=1)

    print("✅ Categorical encoding completed successfully.")

    # Save back to disk
    X_train.to_csv(x_train_path, index=False)
    X_test.to_csv(x_test_path, index=False)
    print(f"✨ Splits updated with encodings at: {output_dir}\n")

if __name__ == "__main__":
    OUTPUT_DIR = os.path.join("data", "processed")
    encode_categorical_data(OUTPUT_DIR)
