import os
import sys
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc

def evaluate_test_set(output_dir, models_dir, figures_dir):
    print("--- Step 6: Detailed Evaluation on Test Set (20%) ---")

    x_test_path = os.path.join(output_dir, "X_test.csv")
    y_test_path = os.path.join(output_dir, "y_test.csv")

    if not os.path.exists(x_test_path) or not os.path.exists(y_test_path):
        raise FileNotFoundError("Test splits not found.")

    X_test = pd.read_csv(x_test_path)
    y_test = pd.read_csv(y_test_path).values.ravel()

    models_to_evaluate = ["Support_Vector_Machine", "Naive_Bayes"]
    os.makedirs(figures_dir, exist_ok=True)

    # Setup plots
    fig_cm, axes_cm = plt.subplots(1, 2, figsize=(14, 5))
    plt.figure(figsize=(8, 6)) # for ROC curve

    for idx, model_name in enumerate(models_to_evaluate):
        model_path = os.path.join(models_dir, f"{model_name}.pkl")
        if not os.path.exists(model_path):
            print(f"⚠️ Model {model_name} not found. Skipping.")
            continue

        with open(model_path, 'rb') as f:
            model = pickle.load(f)

        # Get Predictions
        y_pred = model.predict(X_test)
        
        # --- A. CONFUSION MATRIX ---
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes_cm[idx], cbar=False,
                    xticklabels=['Healthy (0)', 'Disease (1)'],
                    yticklabels=['Healthy (0)', 'Disease (1)'])
        axes_cm[idx].set_title(f'Confusion Matrix: {model_name}')
        axes_cm[idx].set_xlabel('Predicted Label')
        axes_cm[idx].set_ylabel('True Label')

        # --- B. ROC CURVE ---
        if hasattr(model, "predict_proba"):
            y_scores = model.predict_proba(X_test)[:, 1]
        else:
            y_scores = model.decision_function(X_test)

        fpr, tpr, _ = roc_curve(y_test, y_scores)
        roc_auc = auc(fpr, tpr)
        
        plt.plot(fpr, tpr, label=f'{model_name} (AUC = {roc_auc:.2f})', linewidth=2)

    # Save Confusion Matrices
    cm_output_path = os.path.join(figures_dir, "confusion_matrices.png")
    fig_cm.savefig(cm_output_path, bbox_inches='tight', dpi=300)
    plt.close(fig_cm)
    print(f"📊 Confusion Matrices saved to: {cm_output_path}")

    # Finalize and Save ROC Curve
    plt.plot([0, 1], [0, 1], 'k--')  # Diagonal random guess line
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (1 - Specificity)')
    plt.ylabel('True Positive Rate (Sensitivity / Recall)')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    
    roc_output_path = os.path.join(figures_dir, "roc_curve.png")
    plt.savefig(roc_output_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"📈 ROC Curves saved to: {roc_output_path}\n")
    print("✅ Comprehensive testing evaluation finished.")

if __name__ == "__main__":
    OUTPUT_DIR = os.path.join("data", "processed")
    MODELS_DIR = os.path.join("models")
    FIGURES_DIR = os.path.join("reports", "figures")
    evaluate_test_set(OUTPUT_DIR, MODELS_DIR, FIGURES_DIR)