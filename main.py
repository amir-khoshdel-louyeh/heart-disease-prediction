import os
import subprocess
import sys
import shutil

def prepare_data_copy():
    """این تابع یک نسخه کپی امن از فایل دیتا ایجاد می‌کند تا دیتای خام دست‌نخورده
    بماند.
    """
    print("📁 Preparing data copy...")
    source_dir = os.path.join("data", "raw")
    target_dir = os.path.join("data", "processed")
    file_name = "dataset.csv"

    source_path = os.path.join(source_dir, file_name)
    target_path = os.path.join(target_dir, file_name)

    os.makedirs(target_dir, exist_ok=True)

    if not os.path.exists(source_path):
        print(
            f"❌ Error: Raw data file not found at {source_path}! Pipeline stopped."
        )
        return False

    try:
        shutil.copy2(source_path, target_path)
        print(f"🔄 Safe copy created at: {target_path}")
        return True
    except Exception as e:
        print(f"❌ Failed to copy data: {e}")
        return False


def run_pipeline():
    print("==================================================")
    print("🚀 STARTING HEART DISEASE PREDICTION PIPELINE 🚀")
    print("==================================================\n")

    # --------------------------------------------------
    # STEP 0: Data Backup & Preparation
    # --------------------------------------------------
    # (shutil must be imported if used here, making sure it's active)

    if not prepare_data_copy():
        return

    print("==================================================")

    # --------------------------------------------------
    # STEP 1: Data Cleaning
    # --------------------------------------------------
    print("Running Step 1: Data Cleaning...")
    step1_script = os.path.join("src", "data_cleaning.py")
    result1 = subprocess.run([sys.executable, step1_script])

    if result1.returncode != 0:
        print("❌ Error occurred in Step 1! Pipeline stopped.")
        return

    print("✅ Step 1 completed successfully.\n")
    print("==================================================")

    # --------------------------------------------------
    # STEP 2: Categorical Encoding (NEW)
    # --------------------------------------------------
    print("Running Step 2: Categorical Encoding...")
    step2_script = os.path.join("src", "categorical_encoding.py")
    result2 = subprocess.run([sys.executable, step2_script])

    if result2.returncode != 0:
        print("❌ Error occurred in Step 2! Pipeline stopped.")
        return

    print("✅ Step 2 completed successfully.\n")
    print("==================================================")

    # --------------------------------------------------
    # STEP 3: Feature Scaling (NEW)
    # --------------------------------------------------
    print("Running Step 3: Feature Scaling...")
    step3_script = os.path.join("src", "feature_scaling.py")
    result3 = subprocess.run([sys.executable, step3_script])

    if result3.returncode != 0:
        print("❌ Error occurred in Step 3! Pipeline stopped.")
        return

    print("✅ Step 3 completed successfully.\n")
    print("==================================================")

    
    # --------------------------------------------------
    # STEP 4: Feature Selection (NEW)
    # --------------------------------------------------
    print("Running Step 4: Feature Selection & Correlation Analysis...")
    step4_script = os.path.join("src", "feature_selection.py")
    result4 = subprocess.run([sys.executable, step4_script])

    if result4.returncode != 0:
        print("❌ Error occurred in Step 4! Pipeline stopped.")
        return

    print("✅ Step 4 completed successfully.\n")
    print("==================================================")
    
    
    # --------------------------------------------------
    # STEP 5: Model Training & Evaluation (NEW)
    # --------------------------------------------------
    print("Running Step 5: Model Training & Evaluation...")
    step5_script = os.path.join("src", "model_training.py")
    result5 = subprocess.run([sys.executable, step5_script])

    if result5.returncode != 0:
        print("❌ Error occurred in Step 5! Pipeline stopped.")
        return

    print("✅ Step 5 completed successfully.\n")
    print("==================================================")
    print("🎉 Pipeline executed up to the current ready state!")
    print("==================================================")


if __name__ == "__main__":
    run_pipeline()