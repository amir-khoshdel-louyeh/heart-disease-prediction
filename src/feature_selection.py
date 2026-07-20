import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def select_features(output_dir, output_image_path):
    print("--- Step 4: Feature Selection & Correlation Analysis ---")

    x_train_path = os.path.join(output_dir, "X_train.csv")
    x_test_path = os.path.join(output_dir, "X_test.csv")
    y_train_path = os.path.join(output_dir, "y_train.csv")

    if not os.path.exists(x_train_path) or not os.path.exists(y_train_path):
        raise FileNotFoundError("Training splits not found.")

    X_train = pd.read_csv(x_train_path)
    X_test = pd.read_csv(x_test_path)
    y_train = pd.read_csv(y_train_path)

    # Combine X_train and y_train temporarily just to compute correlation
    train_df = pd.concat([X_train, y_train], axis=1)

    # 3. Calculate correlation matrix on TRAINING DATA ONLY
    correlation_matrix = train_df.corr()

    # 4. Generate and save Heatmap
    plt.figure(figsize=(16, 12))
    sns.heatmap(
        correlation_matrix,
        annot=True,
        cmap="coolwarm",
        fmt=".2f",
        linewidths=0.5,
    )
    plt.title("Feature Correlation Matrix - Heart Disease Dataset (Train Split Only)", fontsize=16)

    # Ensure the output directory exists
    os.makedirs(os.path.dirname(output_image_path), exist_ok=True)

    # Save the plot
    plt.savefig(output_image_path, bbox_inches="tight", dpi=300)
    plt.close()
    print(f"📊 Correlation Heatmap saved successfully to: {output_image_path}")

    # 5. Feature Selection based on correlation with target
    target_corr = correlation_matrix["target"].abs().sort_values(ascending=False)

    print("\nTop features correlated with target:")
    print(target_corr)

    # Setting a threshold: Drop features with less than 0.05 absolute correlation
    threshold = 0.05
    low_corr_features = target_corr[target_corr < threshold].index.tolist()
    
    # Exclude 'target' from low_corr_features just in case
    low_corr_features = [f for f in low_corr_features if f != 'target']

    if len(low_corr_features) > 0:
        print(f"\n✂️ Dropping low-correlation features (Threshold < {threshold}): {low_corr_features}")
        X_train = X_train.drop(columns=[col for col in low_corr_features if col in X_train.columns])
        X_test = X_test.drop(columns=[col for col in low_corr_features if col in X_test.columns])
    else:
        print(f"\nℹ️ No features found with correlation lower than {threshold}. All features retained.")

    print("✅ Feature selection completed successfully.")

    # 6. Save and overwrite back to disk
    X_train.to_csv(x_train_path, index=False)
    X_test.to_csv(x_test_path, index=False)
    print(f"✨ Splits updated after feature selection at: {output_dir}\n")

if __name__ == "__main__":
    OUTPUT_DIR = os.path.join("data", "processed")
    FIGURE_OUTPUT_PATH = os.path.join("reports", "figures", "correlation_heatmap.png")
    select_features(OUTPUT_DIR, FIGURE_OUTPUT_PATH)
