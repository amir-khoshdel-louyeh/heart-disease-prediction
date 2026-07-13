import os
import sys
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report

def train_and_evaluate_models(processed_file_path, models_dir):
    print("--- Step 5: Model Training & Evaluation ---")

    # 1. Check if the file exists
    if not os.path.exists(processed_file_path):
        raise FileNotFoundError(f"Processed data file not found at {processed_file_path}.")

    # 2. Read the preprocessed dataset
    df = pd.read_csv(processed_file_path)

    # 3. Split features (X) and target (y)
    X = df.drop(columns=['target'])
    y = df['target']

    # Split into 80% Training and 20% Testing data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    print(f"📊 Training set size: {X_train.shape[0]} samples")
    print(f"📊 Testing set size: {X_test.shape[0]} samples\n")

    # 4. Initialize Models
    models = {
        "Support_Vector_Machine": SVC(kernel='linear', random_state=42),
        "Naive_Bayes": GaussianNB()
    }

    os.makedirs(models_dir, exist_ok=True)

    # 5. Train and Evaluate each model
    for model_name, model in models.items():
        print(f"🤖 Training {model_name}...")
        model.fit(X_train, y_train)
        
        # Predict on testing set
        y_pred = model.predict(X_test)
        
        # Calculate Accuracy
        accuracy = accuracy_score(y_test, y_pred)
        print(f"🎯 {model_name} Accuracy: {accuracy:.4f}")
        print(f"📋 Classification Report for {model_name}:")
        print(classification_report(y_test, y_pred))
        print("-" * 40)

        # 6. Save the trained model to the models/ directory for future use
        model_save_path = os.path.join(models_dir, f"{model_name}.pkl")
        with open(model_save_path, 'wb') as f:
            pickle.dump(model, f)
        print(f"💾 Saved model object to: {model_save_path}\n")

    print("✅ Model training and saving completed successfully.")

if __name__ == "__main__":
    PROCESSED_DATA_PATH = os.path.join("data", "processed", "dataset.csv")
    MODELS_OUTPUT_DIR = os.path.join("models")

    train_and_evaluate_models(PROCESSED_DATA_PATH, MODELS_OUTPUT_DIR)