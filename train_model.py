from pathlib import Path
import pandas as pd
import numpy as np
import joblib
import json
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)
# =====================================
# FILE PATHS
# =====================================
DATA_PATH = Path("data/diabetic_data.csv")
MODEL_PATH = Path("models/readmission_model.joblib")
METRICS_PATH = Path("models/metrics.json")
# =====================================
# FEATURES
# =====================================
FEATURES = [
    "age",
    "time_in_hospital",
    "num_lab_procedures",
    "num_procedures",
    "num_medications",
    "number_outpatient",
    "number_emergency",
    "number_inpatient",
    "number_diagnoses",
    "insulin",
    "change",
    "diabetesMed",
    "admission_type_id"
]
CAT = [
    "age",
    "insulin",
    "change",
    "diabetesMed",
    "admission_type_id"
]
NUM = [c for c in FEATURES if c not in CAT]
# =====================================
# PREPROCESSING
# =====================================
def create_preprocessor(scale_numeric=False):
    numeric_steps = [
        ("imputer", SimpleImputer(strategy="median"))
    ]
    if scale_numeric:
        numeric_steps.append(
            ("scaler", StandardScaler())
        )
    numeric_pipe = Pipeline(numeric_steps)
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])
    return ColumnTransformer([
        ("num", numeric_pipe, NUM),
        ("cat", categorical_pipe, CAT)
    ])
# =====================================
# THRESHOLD TUNING
# =====================================
def find_best_threshold(y_true, probabilities):
    best_threshold = 0.5
    best_f1 = -1
    for threshold in np.arange(0.05, 0.51, 0.01):
        predictions = (
            probabilities >= threshold
        ).astype(int)
        score = f1_score(
            y_true,
            predictions,
            zero_division=0
        )
        if score > best_f1:
            best_f1 = score
            best_threshold = threshold
    return float(best_threshold)
# =====================================
# MAIN FUNCTION
# =====================================
def main():
    print("\nStarting Improved Readmission Model Training...\n")
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            "Dataset not found inside data folder."
        )
    df = pd.read_csv(DATA_PATH)
    df = df[
        df["readmitted"].isin(["<30", ">30", "NO"])
    ].copy()
    required = FEATURES + ["patient_nbr"]
    missing = [
        c for c in required if c not in df.columns
    ]
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    # Binary target
    y = (df["readmitted"] == "<30").astype(int)
    X = df[FEATURES].copy()
    groups = df["patient_nbr"]
    print("Dataset shape:", df.shape)
    print("\nTarget distribution:")
    print(y.value_counts(normalize=True).round(3))
    # =================================
    # TRAIN / TEST SPLIT
    # =================================
    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=0.20,
        random_state=42
    )
    train_idx, test_idx = next(
        splitter.split(X, y, groups)
    )
    X_train_all = X.iloc[train_idx]
    X_test = X.iloc[test_idx]
    y_train_all = y.iloc[train_idx]
    y_test = y.iloc[test_idx]
    groups_train = groups.iloc[train_idx]
    # =================================
    # TRAIN / VALIDATION SPLIT
    # =================================
    validation_splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=0.20,
        random_state=21
    )
    train_sub_idx, val_idx = next(
        validation_splitter.split(
            X_train_all,
            y_train_all,
            groups_train
        )
    )
    X_train = X_train_all.iloc[train_sub_idx]
    X_val = X_train_all.iloc[val_idx]
    y_train = y_train_all.iloc[train_sub_idx]
    y_val = y_train_all.iloc[val_idx]
    print("\nTraining records:", len(X_train))
    print("Validation records:", len(X_val))
    print("Testing records:", len(X_test))
    # =================================
    # MODELS TO COMPARE
    # =================================
    models = {
        "Random Forest": Pipeline([
            ("preprocess", create_preprocessor()),
            ("classifier", RandomForestClassifier(
                n_estimators=200,
                class_weight="balanced_subsample",
                min_samples_leaf=3,
                random_state=42,
                n_jobs=-1
            ))
        ]),
        "Logistic Regression": Pipeline([
            ("preprocess", create_preprocessor(
                scale_numeric=True
            )),
            ("classifier", LogisticRegression(
                class_weight="balanced",
                max_iter=1000,
                random_state=42
            ))
        ])
    }
    results = {}
    best_model_name = None
    best_model = None
    best_threshold = 0.5
    best_validation_f1 = -1
    # =================================
    # MODEL TRAINING AND COMPARISON
    # =================================
    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train, y_train)
        probabilities = model.predict_proba(X_val)[:, 1]
        threshold = find_best_threshold(
            y_val,
            probabilities
        )
        predictions = (
            probabilities >= threshold
        ).astype(int)
        precision = precision_score(
            y_val, predictions, zero_division=0
        )
        recall = recall_score(
            y_val, predictions, zero_division=0
        )
        f1 = f1_score(
            y_val, predictions, zero_division=0
        )
        auc = roc_auc_score(
            y_val, probabilities
        )
        results[name] = {
            "threshold": round(threshold, 3),
            "precision": round(precision, 3),
            "recall": round(recall, 3),
            "f1_score": round(f1, 3),
            "roc_auc": round(auc, 3)
        }
        print("\nValidation results:", name)
        print(results[name])
        if f1 > best_validation_f1:
            best_validation_f1 = f1
            best_model_name = name
            best_model = model
            best_threshold = threshold
    # =================================
    # FINAL TEST EVALUATION
    # =================================
    print("\n=================================")
    print("SELECTED MODEL:", best_model_name)
    print("=================================")
    test_probabilities = best_model.predict_proba(
        X_test
    )[:, 1]
    test_predictions = (
        test_probabilities >= best_threshold
    ).astype(int)
    test_metrics = {
        "accuracy": round(
            accuracy_score(y_test, test_predictions), 3
        ),
        "precision": round(
            precision_score(
                y_test, test_predictions, zero_division=0
            ), 3
        ),
        "recall": round(
            recall_score(
                y_test, test_predictions, zero_division=0
            ), 3
        ),
        "f1_score": round(
            f1_score(
                y_test, test_predictions, zero_division=0
            ), 3
        ),
        "roc_auc": round(
            roc_auc_score(y_test, test_probabilities), 3
        )
    }
    print("\nFINAL CONFUSION MATRIX:")
    print(confusion_matrix(y_test, test_predictions))
    print("\nCLASSIFICATION REPORT:")
    print(classification_report(
        y_test,
        test_predictions,
        zero_division=0
    ))
    print("\nFINAL TEST METRICS:")
    for key, value in test_metrics.items():
        print(f"{key}: {value}")
    # =================================
    # SAVE MODEL
    # =================================
    MODEL_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )
    model_bundle = {
        "pipeline": best_model,
        "threshold": best_threshold,
        "model_name": best_model_name,
        "features": FEATURES
    }
    joblib.dump(model_bundle, MODEL_PATH)
    final_results = {
        "selected_model": best_model_name,
        "validation_results": results,
        "selected_threshold": round(best_threshold, 3),
        "test_metrics": test_metrics
    }
    METRICS_PATH.write_text(
        json.dumps(final_results, indent=4)
    )
    print("\n=================================")
    print("MODEL TRAINING COMPLETED!")
    print("=================================")
    print("Saved model:", MODEL_PATH)
    print("Saved metrics:", METRICS_PATH)
if __name__ == "__main__":
    main()