import os
import sys
import numpy as np
import pandas as pd

# Opt into the future pandas behavior: suppress FutureWarning about silent
# downcasting in .fillna() / .ffill() / .bfill() on object-dtype arrays.
# This makes the code forward-compatible with the next major pandas release.
pd.set_option("future.no_silent_downcasting", True)


def load_and_clean_data(processed_file_path):
    print("--- Step 1: Loading and Cleaning Data ---")

    # ۱. بررسی وجود فایل کپی شده
    if not os.path.exists(processed_file_path):
        raise FileNotFoundError(
            f"Processed data copy not found at {processed_file_path}. Please run pipeline.py first."
        )

    # ۲. خواندن مستقیم دیتای کپی شده
    df = pd.read_csv(processed_file_path)

    # ۳. تبدیل کلاس‌های هدف مالتی‌کلاس به باینری (0 = Healthy, 1 = Disease)
    if "num" in df.columns:
        df["target"] = df["num"].apply(lambda x: 1 if x > 0 else 0)
        df = df.drop(columns=["num"])
    elif "target" not in df.columns and "num" not in df.columns:
        df["target"] = df.iloc[:, -1].apply(lambda x: 1 if x > 0 else 0)

    # ۴. حذف ستون‌های کاملاً بلااستفاده
    cols_to_drop = ["id", "dataset"]
    df = df.drop(columns=[col for col in cols_to_drop if col in df.columns])

    # ۵. مدیریت مقادیر گم‌شده (Missing Values)
    # پر کردن ستون‌های عددی با میانه (Median)
    num_cols = ["trestbps", "chol", "thalch", "oldpeak"]
    for col in num_cols:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())

    # پر کردن ستون‌های دسته‌ای با مد (Mode)
    cat_cols = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]
    for col in cat_cols:
        if col in df.columns:
            # مطمئن می‌شویم که مد خالی نیست
            if not df[col].mode().empty:
                df[col] = df[col].fillna(df[col].mode()[0])

    print("✅ Data cleaning calculations completed successfully.")

    # ۶. ذخیره و جایگزینی مستقیم روی همان فایل کپی شده (بدون ساخت فایل جدید)
    df.to_csv(processed_file_path, index=False)
    print(f"✨ Safe copy updated and overwritten at: {processed_file_path}\n")


if __name__ == "__main__":
    # آدرس فایل کپی شده در پوشه processed
    # توجه: نام فایل را با نام فایل کپی شده در اسکریپت اصلی (مثلا heart_disease.csv) هماهنگ نگه دارید
    PROCESSED_DATA_PATH = os.path.join("data", "processed", "dataset.csv")

    load_and_clean_data(PROCESSED_DATA_PATH)