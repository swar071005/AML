import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import (
    train_test_split, GridSearchCV, RandomizedSearchCV
)
from sklearn.preprocessing import LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import mutual_info_classif, RFE
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_auc_score
)
from sklearn.inspection import permutation_importance
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier


DATA_PATH = "data/student_performance.csv"
TARGET_COLUMN = None
RANDOM_STATE = 42
OUTPUT_DIR = "outputs"

os.makedirs(OUTPUT_DIR, exist_ok=True)


def find_target(df):
    preferred = [
        "Performance", "performance", "Grade", "grade",
        "Target", "target", "Result", "result",
        "Class", "class", "Outcome", "outcome"
    ]
    for col in preferred:
        if col in df.columns:
            return col

    # fallback: use last column if it looks categorical/class-like
    last = df.columns[-1]
    if df[last].nunique() <= max(20, int(len(df) * 0.05)):
        print(f"[WARNING] Target not explicitly recognized. Using last column: {last}")
        return last

    raise ValueError(
        "Could not identify target column. Set TARGET_COLUMN at the top of train.py."
    )


def evaluate_model(name, model, X_test, y_test, results):
    pred = model.predict(X_test)

    acc = accuracy_score(y_test, pred)
    prec = precision_score(y_test, pred, average="weighted", zero_division=0)
    rec = recall_score(y_test, pred, average="weighted", zero_division=0)
    f1 = f1_score(y_test, pred, average="weighted", zero_division=0)

    auc = np.nan
    if hasattr(model, "predict_proba"):
        try:
            proba = model.predict_proba(X_test)
            if len(np.unique(y_test)) == 2:
                auc = roc_auc_score(y_test, proba[:, 1])
            else:
                auc = roc_auc_score(
                    y_test, proba, multi_class="ovr", average="weighted"
                )
        except Exception:
            pass

    results.append({
        "Model": name,
        "Accuracy": acc,
        "Weighted Precision": prec,
        "Weighted Recall": rec,
        "Weighted F1": f1,
        "ROC-AUC": auc
    })

    print(f"\n===== {name} =====")
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1       : {f1:.4f}")
    if not np.isnan(auc):
        print(f"ROC-AUC  : {auc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, pred, zero_division=0))
    return pred


print("\n========== STUDENT PERFORMANCE AML PROJECT ==========\n")

if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(
        f"{DATA_PATH} not found. Put your CSV inside data/ and name it "
        "student_performance.csv"
    )

df = pd.read_csv(DATA_PATH)

print("Dataset shape:", df.shape)
print("\nColumns:")
print(df.columns.tolist())
print("\nMissing values:")
print(df.isnull().sum())
print("\nDuplicates:", df.duplicated().sum())

# target = TARGET_COLUMN if TARGET_COLUMN else find_target(df)
# Create Performance target from the three score columns
if "Performance" not in df.columns:
    df["overall_score"] = (
        df["math score"] +
        df["reading score"] +
        df["writing score"]
    ) / 3

    def create_performance(score):
        if score < 50:
            return "Average"
        elif score < 65:
            return "Good"
        elif score < 80:
            return "Better"
        else:
            return "Best"

    df["Performance"] = df["overall_score"].apply(create_performance)

target = "Performance"

print("\nPerformance distribution:")
print(df["Performance"].value_counts())
print("\nTarget column:", target)

df = df.drop_duplicates().reset_index(drop=True)

# Explicit feature columns to avoid target leakage (exclude scores/overall)
FEATURE_COLUMNS = [
    "gender",
    "race/ethnicity",
    "parental level of education",
    "lunch",
    "test preparation course"
]

# Use only the selected feature columns for modeling
X = df[FEATURE_COLUMNS].copy()

# Target
y_raw = df[target]

# Encode target
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y_raw.astype(str))

print("\nTarget classes:")
for i, c in enumerate(label_encoder.classes_):
    print(i, "=", c)

# Remove obvious ID-like columns
id_cols = []
for col in X.columns:
    if X[col].nunique() == len(X) and (
        "id" in col.lower() or "roll" in col.lower() or "index" in col.lower()
    ):
        id_cols.append(col)

if id_cols:
    print("\nRemoving ID-like columns:", id_cols)
    X = X.drop(columns=id_cols)

numeric_cols = X.select_dtypes(include=np.number).columns.tolist()
categorical_cols = X.select_dtypes(exclude=np.number).columns.tolist()

numeric_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="median"))
])

# Build OneHotEncoder with compatibility for different scikit-learn versions
ohe_kwargs = {"handle_unknown": "ignore"}
try:
    ohe = OneHotEncoder(**ohe_kwargs, sparse_output=False)
except TypeError:
    # older sklearn versions use `sparse` instead of `sparse_output`
    ohe = OneHotEncoder(**ohe_kwargs, sparse=False)

categorical_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", ohe)
])

transformers = []
if numeric_cols:
    transformers.append(("num", numeric_pipe, numeric_cols))
if categorical_cols:
    transformers.append(("cat", categorical_pipe, categorical_cols))

preprocessor = ColumnTransformer(transformers, remainder="drop")

X_processed = preprocessor.fit_transform(X)

feature_names = []
if numeric_cols:
    feature_names.extend(numeric_cols)
# Safely extract one-hot encoded feature names (compat across sklearn versions)
if categorical_cols and "cat" in preprocessor.named_transformers_:
    onehot_step = preprocessor.named_transformers_["cat"].named_steps.get("onehot")
    if onehot_step is not None:
        try:
            names = onehot_step.get_feature_names_out(categorical_cols)
        except Exception:
            # older sklearn versions may have `get_feature_names`
            try:
                names = onehot_step.get_feature_names(categorical_cols)
            except Exception:
                names = [f for c in categorical_cols for f in [c]]
        feature_names.extend(names)

X_processed = np.asarray(X_processed, dtype=float)
feature_names = np.array(feature_names)

X_train, X_test, y_train, y_test = train_test_split(
    X_processed, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
)

print("\nTrain shape:", X_train.shape)
print("Test shape :", X_test.shape)

results = []

# ---------------------------------------------------------
# 1. Baseline Random Forest
# ---------------------------------------------------------
baseline_rf = RandomForestClassifier(
    n_estimators=300,
    random_state=RANDOM_STATE,
    class_weight="balanced"
)
baseline_rf.fit(X_train, y_train)
evaluate_model("Baseline Random Forest", baseline_rf, X_test, y_test, results)

# ---------------------------------------------------------
# 2. Information Gain / Mutual Information
# ---------------------------------------------------------
mi = mutual_info_classif(
    X_train, y_train, random_state=RANDOM_STATE
)

mi_df = pd.DataFrame({
    "Feature": feature_names,
    "Information_Gain": mi
}).sort_values("Information_Gain", ascending=False)

mi_df.to_csv(
    os.path.join(OUTPUT_DIR, "information_gain_ranking.csv"),
    index=False
)

TOP_K = min(max(10, X_train.shape[1] // 2), X_train.shape[1])
selected_mi = mi_df.head(TOP_K)["Feature"].values
mi_indices = [
    np.where(feature_names == f)[0][0] for f in selected_mi
]

X_train_mi = X_train[:, mi_indices]
X_test_mi = X_test[:, mi_indices]

print(f"\nInformation Gain selected {len(mi_indices)} features.")

# ---------------------------------------------------------
# 3. RFE
# ---------------------------------------------------------
rfe_estimator = RandomForestClassifier(
    n_estimators=150,
    random_state=RANDOM_STATE,
    n_jobs=-1
)

rfe_n = min(max(10, X_train.shape[1] // 2), X_train.shape[1])

rfe = RFE(
    estimator=rfe_estimator,
    n_features_to_select=rfe_n,
    step=0.2
)
rfe.fit(X_train, y_train)

rfe_indices = np.where(rfe.support_)[0]
rfe_features = feature_names[rfe_indices]

pd.DataFrame({
    "Feature": feature_names,
    "Selected_By_RFE": rfe.support_,
    "Ranking": rfe.ranking_
}).to_csv(
    os.path.join(OUTPUT_DIR, "rfe_ranking.csv"),
    index=False
)

X_train_rfe = X_train[:, rfe_indices]
X_test_rfe = X_test[:, rfe_indices]

print(f"RFE selected {len(rfe_indices)} features.")

# ---------------------------------------------------------
# 4. SMOTE on training data ONLY
# ---------------------------------------------------------
print("\nClass distribution before SMOTE:")
print(pd.Series(y_train).value_counts().sort_index())

smote = SMOTE(random_state=RANDOM_STATE)
X_train_smote, y_train_smote = smote.fit_resample(X_train_mi, y_train)

print("\nClass distribution after SMOTE:")
print(pd.Series(y_train_smote).value_counts().sort_index())

# ---------------------------------------------------------
# 5. Random Forest + SMOTE
# ---------------------------------------------------------
rf_smote = RandomForestClassifier(
    n_estimators=400,
    random_state=RANDOM_STATE,
    n_jobs=-1
)
rf_smote.fit(X_train_smote, y_train_smote)
pred_rf = evaluate_model(
    "Random Forest + Information Gain + SMOTE",
    rf_smote, X_test_mi, y_test, results
)

# ---------------------------------------------------------
# 6. XGBoost + SMOTE
# ---------------------------------------------------------
n_classes = len(np.unique(y))

xgb_params = {
    "n_estimators": 350,
    "max_depth": 5,
    "learning_rate": 0.05,
    "subsample": 0.85,
    "colsample_bytree": 0.85,
    "random_state": RANDOM_STATE,
    "n_jobs": -1
}

if n_classes == 2:
    xgb = XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        **xgb_params
    )
else:
    xgb = XGBClassifier(
        objective="multi:softprob",
        num_class=n_classes,
        eval_metric="mlogloss",
        **xgb_params
    )

xgb.fit(X_train_smote, y_train_smote)
pred_xgb = evaluate_model(
    "XGBoost + Information Gain + SMOTE",
    xgb, X_test_mi, y_test, results
)

# ---------------------------------------------------------
# 7. Hyperparameter Optimization - RF Grid Search
# ---------------------------------------------------------
rf_grid = {
    "n_estimators": [200, 400],
    "max_depth": [None, 8, 15],
    "min_samples_split": [2, 5],
    "min_samples_leaf": [1, 2]
}

grid = GridSearchCV(
    RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1),
    rf_grid,
    scoring="f1_weighted",
    cv=3,
    n_jobs=-1
)
grid.fit(X_train_smote, y_train_smote)

best_rf = grid.best_estimator_

print("\nBest RF parameters:")
print(grid.best_params_)

evaluate_model(
    "Optimized Random Forest",
    best_rf, X_test_mi, y_test, results
)

# ---------------------------------------------------------
# 8. XGBoost Randomized Search
# ---------------------------------------------------------
xgb_base_params = {
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
    "eval_metric": "logloss" if n_classes == 2 else "mlogloss"
}

if n_classes == 2:
    xgb_base = XGBClassifier(
        objective="binary:logistic",
        **xgb_base_params
    )
else:
    xgb_base = XGBClassifier(
        objective="multi:softprob",
        num_class=n_classes,
        **xgb_base_params
    )

xgb_random_grid = {
    "n_estimators": [200, 300, 500],
    "max_depth": [3, 4, 5, 7],
    "learning_rate": [0.02, 0.05, 0.1],
    "subsample": [0.7, 0.85, 1.0],
    "colsample_bytree": [0.7, 0.85, 1.0]
}

random_search = RandomizedSearchCV(
    xgb_base,
    xgb_random_grid,
    n_iter=12,
    scoring="f1_weighted",
    cv=3,
    random_state=RANDOM_STATE,
    n_jobs=-1
)
random_search.fit(X_train_smote, y_train_smote)

best_xgb = random_search.best_estimator_

print("\nBest XGBoost parameters:")
print(random_search.best_params_)

evaluate_model(
    "Optimized XGBoost",
    best_xgb, X_test_mi, y_test, results
)

# ---------------------------------------------------------
# 9. Stacking Ensemble
# ---------------------------------------------------------
stack_rf = RandomForestClassifier(
    n_estimators=250,
    random_state=RANDOM_STATE,
    n_jobs=-1
)

stack_xgb_params = {
    "n_estimators": 250,
    "max_depth": 5,
    "learning_rate": 0.05,
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
    "eval_metric": "logloss" if n_classes == 2 else "mlogloss"
}

if n_classes == 2:
    stack_xgb = XGBClassifier(
        objective="binary:logistic",
        **stack_xgb_params
    )
else:
    stack_xgb = XGBClassifier(
        objective="multi:softprob",
        num_class=n_classes,
        **stack_xgb_params
    )

stack = StackingClassifier(
    estimators=[
        ("rf", stack_rf),
        ("xgb", stack_xgb)
    ],
    final_estimator=LogisticRegression(max_iter=2000),
    cv=3,
    n_jobs=-1
)

stack.fit(X_train_smote, y_train_smote)

evaluate_model(
    "Stacking Ensemble",
    stack, X_test_mi, y_test, results
)

# ---------------------------------------------------------
# 10. Model comparison
# ---------------------------------------------------------
results_df = pd.DataFrame(results)
results_df.to_csv(
    os.path.join(OUTPUT_DIR, "model_comparison.csv"),
    index=False
)

print("\n========== MODEL COMPARISON ==========")
print(results_df.to_string(index=False))

plt.figure(figsize=(12, 6))
plot_df = results_df.melt(
    id_vars="Model",
    value_vars=["Accuracy", "Weighted Precision", "Weighted Recall", "Weighted F1"]
)
sns.barplot(data=plot_df, x="Model", y="value", hue="variable")
plt.xticks(rotation=30, ha="right")
plt.ylabel("Score")
plt.title("Model Performance Comparison")
plt.tight_layout()
plt.savefig(
    os.path.join(OUTPUT_DIR, "model_comparison.png"),
    dpi=300
)
plt.close()

# ---------------------------------------------------------
# 11. Confusion matrix for best model
# ---------------------------------------------------------
best_model_name = results_df.sort_values(
    "Weighted F1", ascending=False
).iloc[0]["Model"]

if best_model_name == "Baseline Random Forest":
    best_model = baseline_rf
    best_X_test = X_test
elif best_model_name == "Random Forest + Information Gain + SMOTE":
    best_model = rf_smote
    best_X_test = X_test_mi
elif best_model_name == "XGBoost + Information Gain + SMOTE":
    best_model = xgb
    best_X_test = X_test_mi
elif best_model_name == "Optimized Random Forest":
    best_model = best_rf
    best_X_test = X_test_mi
elif best_model_name == "Optimized XGBoost":
    best_model = best_xgb
    best_X_test = X_test_mi
else:
    best_model = stack
    best_X_test = X_test_mi

best_pred = best_model.predict(best_X_test)

cm = confusion_matrix(y_test, best_pred)

plt.figure(figsize=(7, 6))
sns.heatmap(
    cm, annot=True, fmt="d", cmap="Blues",
    xticklabels=label_encoder.classes_,
    yticklabels=label_encoder.classes_
)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title(f"Confusion Matrix - {best_model_name}")
plt.tight_layout()
plt.savefig(
    os.path.join(OUTPUT_DIR, "confusion_matrix_best_model.png"),
    dpi=300
)
plt.close()

# ---------------------------------------------------------
# 12. Feature importance
# ---------------------------------------------------------
importance_model = best_model
if hasattr(importance_model, "feature_importances_"):
    importances = importance_model.feature_importances_

    # Determine the correct feature name list to pair with importances
    feat_names = None

    try:
        if 'selected_mi' in globals() and selected_mi is not None and len(importances) == len(selected_mi):
            feat_names = selected_mi
        elif 'feature_names' in globals() and feature_names is not None and len(importances) == len(feature_names):
            feat_names = feature_names
        elif 'rfe_indices' in globals() and rfe_indices is not None and len(importances) == len(rfe_indices):
            feat_names = feature_names[rfe_indices]
    except Exception:
        feat_names = None

    if feat_names is None:
        # Fallback: generate placeholder feature names matching importance length
        feat_names = [f"feature_{i}" for i in range(len(importances))]

    importance_df = pd.DataFrame({
        "Feature": feat_names,
        "Importance": importances
    }).sort_values("Importance", ascending=False)

    importance_df.to_csv(
        os.path.join(OUTPUT_DIR, "feature_importance.csv"),
        index=False
    )

    top = importance_df.head(15).sort_values("Importance")

    plt.figure(figsize=(10, 7))
    plt.barh(top["Feature"], top["Importance"])
    plt.xlabel("Importance")
    plt.title(f"Top Features - {best_model_name}")
    plt.tight_layout()
    plt.savefig(
        os.path.join(OUTPUT_DIR, "feature_importance.png"),
        dpi=300
    )
    plt.close()

# SHAP section removed per request

# ---------------------------------------------------------
# 14. Save final artifacts
# ---------------------------------------------------------
joblib.dump(
    {
        "preprocessor": preprocessor,
        "target_encoder": label_encoder,
        "selected_feature_names": selected_mi,
        "selected_indices": mi_indices,
        "model": best_model,
        "target_column": target
    },
    os.path.join(OUTPUT_DIR, "final_model.pkl"),
    compress=3
)

with open(os.path.join(OUTPUT_DIR, "best_model.txt"), "w") as f:
    f.write(f"Best model by weighted F1: {best_model_name}\n")
    f.write("\nAll results:\n")
    f.write(results_df.to_string(index=False))

print("\n================================================")
print("PROJECT COMPLETE")
print("Best model:", best_model_name)
print("Results saved in:", OUTPUT_DIR)
print("================================================")
