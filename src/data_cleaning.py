import os
import pandas as pd
import numpy as np

def load_and_clean_data(input_path, output_path):
    print("--- Step 1: Loading and Cleaning Data ---")
    
    # 1. Load the raw dataset
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Raw data not found at {input_path}")
        
    df = pd.read_csv(input_path)
    
    # 2. Convert multi-class target 'num' to binary 'target' (0 = Healthy, 1 = Disease)
    # Checking if 'num' exists in columns
    if 'num' in df.columns:
        df['target'] = df['num'].apply(lambda x: 1 if x > 0 else 0)
        df = df.drop(columns=['num'])
    elif 'target' not in df.columns and 'num' not in df.columns:
        # If the column name is already 'target' but has values 0-4
        df['target'] = df.iloc[:, -1].apply(lambda x: 1 if x > 0 else 0)
    
    # 3. Drop completely unnecessary columns if they exist
    cols_to_drop = ['id', 'dataset']
    df = df.drop(columns=[col for col in cols_to_drop if col in df.columns])
    
    # 4. Handle Missing Values
    # Fill numerical columns with Median
    num_cols = ['trestbps', 'chol', 'thalch', 'oldpeak']
    for col in num_cols:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())
        
    # Fill categorical columns with Mode
    cat_cols = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']
    for col in cat_cols:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].mode()[0])
            
    print("Data cleaning completed successfully.")
    
    # 5. Save the cleaned data to the processed folder
    df.to_csv(output_path, index=False)
    print(f"Saved intermediate cleaned data to: {output_path}\n")

if __name__ == "__main__":
    # Defining relative paths based on your project structure
    # Since this script runs inside 'src', we look for data in the sibling folders
    RAW_DATA_PATH = os.path.join("data", "raw", "dataset.csv")
    PROCESSED_DATA_PATH = os.path.join("data", "processed", "1_cleaned_data.csv")
    
    load_and_clean_data(RAW_DATA_PATH, PROCESSED_DATA_PATH)