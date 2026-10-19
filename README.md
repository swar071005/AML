# Student Performance Prediction - Advanced Machine Learning

Project based on the proposed AML approach:
Feature Selection + SMOTE + Random Forest + XGBoost + Hyperparameter Tuning + Explainability.

## Project flow
1. Load student performance CSV
2. Automatic preprocessing of numerical/categorical columns
3. Train/test split
4. Baseline Random Forest
5. Information Gain feature selection
6. RFE feature selection
7. SMOTE on training data only
8. Random Forest + XGBoost
9. Grid Search / Randomized Search
10. Stacking ensemble
11. Evaluation: Accuracy, Precision, Recall, F1, ROC-AUC
12. Confusion matrix and feature importance


## Dataset
Put your CSV in:
data/student_performance.csv

The code tries to find the target column automatically. Preferred target names:
Performance, performance, Grade, grade, Target, target, Result, result, Class, class.

If automatic detection fails, edit TARGET_COLUMN in train.py.

## Run in VS Code

Create environment:
python -m venv .venv

Windows:
.venv\Scripts\activate

Install:
pip install -r requirements.txt

Run:
python train.py

Results are saved in:
outputs/

## Important
Do not apply SMOTE before train/test splitting. This project applies it only to training data to avoid data leakage.
