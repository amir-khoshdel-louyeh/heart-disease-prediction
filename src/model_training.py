import os
import sys
import pickle
import pandas as pd
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report

def train_and_evaluate_models(output_dir, models_dir):
    print("--- Step 5: Model Training & Evaluation ---")

    x_train_path = os.path.join(output_dir, "X_train.csv")
    y_train_path = os.path.join(output_dir, "y_train.csv")
    x_test_path = os.path.join(output_dir, "X_test.csv")
    y_test_path = os.path.join(output_dir, "y_test.csv")

    if not all(os.path.exists(p) for p in [x_train_path, y_train_path, x_test_path, y_test_path]):
        raise FileNotFoundError("Processed splits not found.")

    X_train = pd.read_csv(x_train_path)
    y_train = pd.read_csv(y_train_path)
    X_test = pd.read_csv(x_test_path)
    y_test = pd.read_csv(y_test_path)

    # Flatten y correctly
    y_train = y_train.values.ravel()
    y_test = y_test.values.ravel()

    print(f"📊 Training set size: {X_train.shape[0]} samples")
    print(f"📊 Testing set size: {X_test.shape[0]} samples\n")

    # Initialize Models
    models = {
        "Support_Vector_Machine": SVC(kernel='linear', random_state=42, probability=True),
        "Naive_Bayes": GaussianNB()
    }

    os.makedirs(models_dir, exist_ok=True)

    # Train and Evaluate each model
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

        # Save the trained model
        model_save_path = os.path.join(models_dir, f"{model_name}.pkl")
        with open(model_save_path, 'wb') as f:
            pickle.dump(model, f)
        print(f"💾 Saved model object to: {model_save_path}\n")

    print("✅ Model training and saving completed successfully.")

if __name__ == "__main__":
    OUTPUT_DIR = os.path.join("data", "processed")
    MODELS_OUTPUT_DIR = os.path.join("models")
    train_and_evaluate_models(OUTPUT_DIR, MODELS_OUTPUT_DIR)