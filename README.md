# Heart Disease Prediction

![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?style=flat-square&logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.x-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-2.x-150458?style=flat-square&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-1.x-013243?style=flat-square&logo=numpy&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-3.x-11557C?style=flat-square&logo=matplotlib&logoColor=white)

A machine learning pipeline that predicts heart disease presence using SVM and Naive Bayes classifiers, built on the UCI Heart Disease dataset with full preprocessing and evaluation workflows.

![Demo](assets/Demo.gif)

---

## ✨ Highlights

- 🔁 End-to-end ML pipeline with 6 modular, sequential steps
- 🛡️ **Zero Data Leakage** – Train-test splitting occurs *before* any statistical transformations
- 🤖 Two classifiers compared: Support Vector Machine & Naive Bayes
- 📊 Automated confusion matrix and ROC curve generation
- 🔒 Raw data isolation strategy to ensure reproducible experiments

### 🛠️ Built With

`Python` • `scikit-learn` • `pandas` • `NumPy` • `matplotlib` • `seaborn`

---

## 📖 Overview

This project was developed to classify patients as healthy or at risk of heart disease based on clinical measurements.

It enables a complete ML workflow — from raw CSV data to trained, saved models and visual evaluation reports — all orchestrated through a single interactive pipeline runner.

The primary objective was to compare two classical classification algorithms (SVM and Naive Bayes) on a real-world medical dataset, applying proper preprocessing and producing interpretable performance metrics.

The system is intended for educational use, portfolio demonstration, and as a reproducible baseline for further experimentation.

---

## ❗ Problem

Heart disease is among the leading causes of death globally. Early detection from clinical data is critical but difficult due to:

- Mixed data types (continuous measurements and categorical clinical indicators)
- Missing values across multiple features in real-world datasets
- Multicollinearity between clinical features that can mislead models
- Lack of reproducible, documented pipelines for comparing classifiers

As a result, many ad-hoc approaches fail to generalize or document their preprocessing decisions clearly.

This project aims to address these limitations with a clean, step-by-step, verifiable pipeline.

---

## 💡 Solution

The system consists of 6 sequential, modular pipeline steps:

1. **Data Backup & Isolation** — safe copy of raw data to preserve the original
2. **Data Cleaning & Splitting** — splits data (80/20) first to prevent data leakage, then missing value imputation using median/mode strategies strictly fitted on the training split
3. **Categorical Encoding** — One-Hot Encoding with multicollinearity prevention
4. **Feature Scaling** — Z-score standardization on continuous numerical features
5. **Feature Selection** — correlation-based analysis on the training set with heatmap visualization
6. **Model Training & Final Evaluation** — SVM and Naive Bayes training, confusion matrices, and ROC curves

**Workflow:**

```
Raw CSV → Data Backup → Split & Clean → Encoding → Scaling → Feature Selection → Training → Evaluation Reports
```

The pipeline preserves the raw file untouched, saving all cleanly processed train/test splits into `data/processed/`.

---

## 🖥️ Demo

### Pipeline Output

<!-- Add a screenshot of the pipeline terminal output here -->

### Correlation Heatmap

<!-- Add reports/figures/correlation_heatmap.png here -->

### Confusion Matrices

<!-- Add reports/figures/confusion_matrices.png here -->

### ROC Curve

<!-- Add reports/figures/roc_curve.png here -->

---

## ⚙️ Features

- Interactive pipeline with pause-before-each-step control for step-by-step demonstration
- Raw data isolation — original file is never modified; all work happens on a safe copy
- Prevents **Data Leakage** by executing `train_test_split` prior to statistical transformations (imputation, scaling, feature selection).
- Median imputation for continuous features and mode imputation for categorical features
- One-Hot Encoding with `drop_first=True` to prevent multicollinearity
- Z-score standardization applied only to continuous physical measurements
- Correlation-based feature selection with configurable threshold (default: 0.05) calculated exclusively on training data
- Correlation heatmap saved to `reports/figures/correlation_heatmap.png`
- Trained model objects serialized to `models/` as `.pkl` files for reuse
- Side-by-side confusion matrices for both classifiers
- ROC curves with AUC scores plotted on the same figure for direct comparison

---

## 📈 Results

| Metric    | SVM (Linear) | Naive Bayes |
|-----------|:------------:|:-----------:|
| Accuracy  | ~84%         | ~84%        |
| Precision | ~83%         | ~83%        |
| Recall    | ~88%         | ~86%        |
| F1 Score  | ~86%         | ~86%        |

> **Note:** Exact values depend on the dataset version used. Run the pipeline to reproduce results.

- Dataset split: **80% training / 20% testing** (stratified, `random_state=42`)
- Feature selection threshold: **|correlation| = 0.05** with the target variable
- Both models saved as `.pkl` files for reproducible evaluation

---

## 🏗️ Architecture

```
main.py  (Pipeline Orchestrator)
    │
    ├── Step 0: Data Backup & Isolation
    │       └── Copies raw CSV → data/processed/dataset.csv
    │
    ├── Step 1: src/data_cleaning.py
    │       └── Splits into train/test sets, computes median/mode on train, applies to both
    │       └── Saves: X_train.csv, X_test.csv, y_train.csv, y_test.csv
    │
    ├── Step 2: src/categorical_encoding.py
    │       └── Fits One-Hot Encoding on X_train, applies to both
    │
    ├── Step 3: src/feature_scaling.py
    │       └── Fits StandardScaler on continuous columns of X_train, applies to both
    │
    ├── Step 4: src/feature_selection.py
    │       └── Correlation analysis on train set → drops low-correlation features from both
    │       └── Saves: reports/figures/correlation_heatmap.png
    │
    ├── Step 5: src/model_training.py
    │       └── Trains SVM (linear) & Naive Bayes on fully processed X_train
    │       └── Saves: models/Support_Vector_Machine.pkl, models/Naive_Bayes.pkl
    │
    └── Step 6: src/final_evaluation.py
            └── Evaluates saved models on X_test → generates confusion matrices & ROC curves
            └── Saves: reports/figures/confusion_matrices.png, roc_curve.png
```

### Data Layer

Reads raw data, performs a split, and subsequently each preprocessing script overwrites the split files (`X_train.csv`, `X_test.csv`) in `data/processed/`, preserving the raw file in `data/raw/` perfectly.

### Model Layer

Serialized model objects saved to `models/` using `pickle`, enabling Step 6 to evaluate independently of the training step.

### Reports Layer

All figures saved as high-resolution PNG files (300 DPI) to `reports/figures/` for documentation and presentation.

---

## 🔬 Technical Highlights

- **Modular pipeline design** — each step is an independent, testable Python module in `src/`
- **Data Leakage Prevention** — strict sequential operation ensures test data never influences training statistics
- **Raw data isolation** — `shutil.copy2` ensures the raw file is never mutated
- **Defensive file checks** — every module raises `FileNotFoundError` before processing
- **Stratified train/test split** — `stratify=y` preserves class distribution in both sets
- **Reproducible randomness** — `random_state=42` used consistently across training and evaluation
- **ROC-compatible models** — `predict_proba` used for Naive Bayes; `decision_function` used for SVM, with runtime detection via `hasattr`
- **High-resolution output** — all figures exported at 300 DPI with `bbox_inches='tight'`

---

## 🧠 Engineering Decisions

### Why Splitting First?
The most critical engineering change to prevent data leakage is performing the `train_test_split` prior to statistical transformations. If we scale features or compute median imputation on the entire dataset upfront, information from the test set "leaks" into the model's training phase, leading to artificially inflated accuracy metrics. By splitting first, our scaling and imputation values are computed strictly on the training distribution.

### Why SVM with a Linear Kernel?

- Effective in high-dimensional spaces after one-hot encoding
- Linear kernel is interpretable and avoids overfitting on small medical datasets
- Provides a `decision_function` for ROC curve generation without probability calibration

### Why Naive Bayes?

- Fast baseline with strong performance on small datasets
- Produces native class probabilities via `predict_proba`, ideal for ROC evaluation
- Useful as a complementary benchmark against the more complex SVM

### Why Median/Mode Imputation Instead of Dropping Rows?

- Medical datasets are often small; dropping rows with missing values wastes valuable training data
- Median is robust to outliers in clinical measurements (e.g., cholesterol, blood pressure)
- Mode imputation is appropriate for ordinal/nominal clinical indicators

### Why One-Hot Encoding with `drop_first=True`?

- Categorical features like chest pain type (`cp`) and thalassemia (`thal`) are nominal — they have no inherent ordinal order
- `drop_first=True` prevents the dummy variable trap (perfect multicollinearity between encoded columns)

### Why Z-Score Standardization?

- SVM is sensitive to feature scale; standardization ensures no single feature dominates the decision boundary
- Applied only to continuous physical measurements (`age`, `trestbps`, `chol`, `thalch`, `oldpeak`), not to binary or one-hot encoded columns

### Why Correlation-Based Feature Selection?

- A fast, interpretable method to remove features with negligible predictive signal
- Threshold of 0.05 is conservative, avoiding over-aggressive feature removal on a small dataset

---

## ⚠️ Challenges & Lessons Learned

### Challenge 1: Preventing Data Leakage with Step-by-Step Scripts

**Problem:** Using a standard `scikit-learn` Pipeline hides the transformed intermediate datasets from disk. We wanted the step-by-step file generation back for academic grading, while remaining mathematically sound.

**Solution:** Shifted the `train_test_split` operation to Step 1. We now generate `X_train.csv` and `X_test.csv` separately. Each subsequent script loads both, calculates its statistics strictly on the train file, applies the transformation to both, and saves them back to disk.

### Challenge 2: Reproducible Train/Test Split Across Two Scripts

**Problem:** Step 5 (training) and Step 6 (evaluation) are separate scripts. Without coordination, they could use different splits, making the confusion matrix and ROC curve meaningless.

**Solution:** Both scripts now seamlessly load the explicit `X_test.csv` generated back in Step 1, completely preventing any shuffling issues or needing to perfectly reproduce `train_test_split` calls down the line.

### Challenge 3: ROC Curve Compatibility Between Classifiers

**Problem:** SVM does not natively expose class probabilities, while `roc_curve` requires probability-like scores.

**Solution:** Used `hasattr(model, "predict_proba")` at runtime to dynamically switch between `predict_proba` (Naive Bayes) and `decision_function` (SVM), keeping the evaluation code model-agnostic.

### Challenge 4: Applying Scaling Without Corrupting Encoded Features

**Problem:** `StandardScaler` applied to all columns would distort binary (0/1) one-hot encoded features.

**Solution:** Explicitly defined a whitelist of continuous numerical columns (`age`, `trestbps`, `chol`, `thalch`, `oldpeak`) and filtered this list against the current DataFrame columns before scaling.

---

## 📚 Lessons Learned

Through this project I improved my understanding of:

- **Data Leakage** — how seemingly harmless dataset-level operations silently corrupt model generalizability.
- **ML pipeline design** — how to structure preprocessing into sequential, independent steps
- **Data integrity** — the importance of never mutating raw data and using safe copies
- **Classifier evaluation** — how to correctly generate and interpret confusion matrices and ROC/AUC curves
- **Reproducibility** — the critical role of fixed random seeds in ML experiments

---

## 📁 Repository Structure

```
heart-disease-prediction/
├── src/
│   ├── __init__.py
│   ├── data_cleaning.py          # Step 1: Split data, missing value imputation
│   ├── categorical_encoding.py   # Step 2: One-Hot Encoding
│   ├── feature_scaling.py        # Step 3: Z-score standardization
│   ├── feature_selection.py      # Step 4: Correlation analysis & heatmap
│   ├── model_training.py         # Step 5: SVM & Naive Bayes training
│   └── final_evaluation.py       # Step 6: Confusion matrices & ROC curves
├── data/
│   ├── raw/                      # Original, untouched dataset (dataset.csv)
│   └── processed/                # Fully transformed splits (X_train.csv, X_test.csv, etc.)
├── models/                       # Saved .pkl model objects
├── reports/
│   └── figures/                  # Output plots (heatmap, confusion matrices, ROC)
├── main.py                       # Pipeline orchestrator with interactive prompts
├── requirements.txt              # Python dependencies
└── README.md
```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.9+

### ⚡ Quick Start (One-Liner)

For advanced users — clone, create a virtual environment, and install all dependencies in one go:

**Windows (PowerShell):**
```powershell
git clone https://github.com/amir-khoshdel-louyeh/heart-disease-prediction.git; cd heart-disease-prediction; python -m venv .venv; .venv\Scripts\activate; pip install -r requirements.txt
```

**macOS / Linux (bash):**
```bash
git clone https://github.com/amir-khoshdel-louyeh/heart-disease-prediction.git && cd heart-disease-prediction && python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt
```

---

### Step-by-Step Setup

#### 1. Clone the Repository

```bash
git clone https://github.com/amir-khoshdel-louyeh/heart-disease-prediction.git
cd heart-disease-prediction
```

#### 2. Create a Virtual Environment

```bash
python -m venv .venv
```

Activate it:

- **Windows:** `.venv\Scripts\activate`
- **macOS/Linux:** `source .venv/bin/activate`

#### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

#### 4. Add the Dataset

Place your `dataset.csv` file (UCI Heart Disease dataset) into:

```
data/raw/dataset.csv
```

#### 5. Run the Pipeline

```bash
python main.py
```

The pipeline will pause before each step, allowing you to review the output before proceeding.

---

## 📋 Pipeline Steps Reference

| Step | Script                        | Action                                              |
|:----:|-------------------------------|-----------------------------------------------------|
| 0    | `main.py`                     | Copies raw data to `data/processed/`                |
| 1    | `src/data_cleaning.py`        | Splits data, imputes missing values                 |
| 2    | `src/categorical_encoding.py` | Applies One-Hot Encoding safely on train/test       |
| 3    | `src/feature_scaling.py`      | Standardizes continuous features safely             |
| 4    | `src/feature_selection.py`    | Drops low-correlation features, saves heatmap       |
| 5    | `src/model_training.py`       | Trains & saves SVM and Naive Bayes                  |
| 6    | `src/final_evaluation.py`     | Generates confusion matrices & ROC curves           |

You can also run any step independently:

```bash
python src/model_training.py
```

---

## 📦 Output Files

After a full pipeline run, the following files are generated:

```
models/
├── Support_Vector_Machine.pkl
└── Naive_Bayes.pkl

reports/figures/
├── correlation_heatmap.png
├── confusion_matrices.png
└── roc_curve.png
```

---

## 🗺️ Roadmap

- [x] **Refactor preprocessing to eliminate data leakage**
- [ ] Add cross-validation (k-fold) for more robust accuracy estimates
- [ ] Add a Random Forest baseline for additional comparison
- [ ] Export a full classification report to `reports/` as a CSV
- [ ] Add a `predict.py` script for single-patient inference using saved models
- [ ] Add hyperparameter tuning with `GridSearchCV` for SVM kernel and Naive Bayes smoothing
- [ ] Package as a CLI tool with argument parsing

---

## 👤 Author

**Amir Khoshdel Louyeh**

### Connect

* **GitHub:** [github.com/amir-khoshdel-louyeh](https://github.com/amir-khoshdel-louyeh)
* **LinkedIn:** [linkedin.com/in/amir-khoshdel-louyeh](https://www.linkedin.com/in/amir-khoshdel-louyeh)

---

## ⚠️ Disclaimer

This project is intended for educational and research purposes only. It is not a medical device and should not be used for clinical diagnosis or treatment decisions.

---

## ⚖️ License

This project is open-source and available under the **MIT License**.