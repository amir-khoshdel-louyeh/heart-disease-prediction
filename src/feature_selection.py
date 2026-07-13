import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def select_features(processed_file_path, output_image_path):
    print("--- Step 4: Feature Selection & Correlation Analysis ---")

    # 1. Check if the file exists
    if not os.path.exists(processed_file_path):
        raise FileNotFoundError(
            f"Processed data file not found at {processed_file_path}."
        )

    # 2. Read the current dataset state
    df = pd.read_csv(processed_file_path)

    # 3. Calculate correlation matrix
    correlation_matrix = df.corr()

    # 4. Generate and save Heatmap
    plt.figure(figsize=(16, 12))
    sns.heatmap(
        correlation_matrix,
        annot=True,
        cmap="coolwarm",
        fmt=".2f",
        linewidths=0.5,
    )
    plt.title("Feature Correlation Matrix - Heart Disease Dataset", fontsize=16)

    # Ensure the output directory exists
    os.makedirs(os.path.dirname(output_image_path), exist_ok=True)

    # Save the plot
    plt.savefig(output_image_path, bbox_inches="tight", dpi=300)
    plt.close()
    print(f"📊 Correlation Heatmap saved successfully to: {output_image_path}")

    # 5. Feature Selection based on correlation with target
    # Get correlation values with the target variable
    target_corr = correlation_matrix["target"].abs().sort_values(ascending=False)

    print("\nTop features correlated with target:")
    print(target_corr)

    # Setting a threshold: Drop features with less than 0.05 absolute correlation
    threshold = 0.05
    low_corr_features = target_corr[target_corr < threshold].index.tolist()

    if len(low_corr_features) > 0:
        print(
            f"\n✂️ Dropping low-correlation features (Threshold < {threshold}): {low_corr_features}"
        )
        df = df.drop(columns=low_corr_features)
    else:
        print(
            f"\nℹ️ No features found with correlation lower than {threshold}. All features retained."
        )

    print("✅ Feature selection completed successfully.")

    # 6. Save and overwrite the file in data/processed
    df.to_csv(processed_file_path, index=False)
    print(f"✨ Safe copy updated after feature selection at: {processed_file_path}\n")


if __name__ == "__main__":
    PROCESSED_DATA_PATH = os.path.join("data", "processed", "dataset.csv")
    FIGURE_OUTPUT_PATH = os.path.join(
        "reports", "figures", "correlation_heatmap.png"
    )

    select_features(PROCESSED_DATA_PATH, FIGURE_OUTPUT_PATH)