import os
import subprocess
import sys

def run_pipeline():
    print("==================================================")
    # Start the Machine Learning Pipeline
    print("🚀 STARTING HEART DISEASE PREDICTION PIPELINE 🚀")
    print("==================================================\n")

    # --------------------------------------------------
    # STEP 1: Data Cleaning
    # --------------------------------------------------
    print("Running Step 1: Data Cleaning...")
    
    # Path to the first script inside src/
    step1_script = os.path.join("src", "1_data_cleaning.py")
    
    # Execute the script using the current python environment
    result = subprocess.run([sys.executable, step1_script])
    
    # Check if the script executed successfully
    if result.returncode != 0:
        print("❌ Error occurred in Step 1! Pipeline stopped.")
        return
    
    print("✅ Step 1 completed successfully.\n")
    print("==================================================")
    print("🎉 Pipeline executed up to the current ready state!")
    print("==================================================")

if __name__ == "__main__":
    run_pipeline()