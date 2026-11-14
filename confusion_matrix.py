"""
Ganga Water Quality Forecasting - Confusion Matrix & Classification Analysis
=============================================================================
This module provides complete functionality to compute, evaluate, and visualize
Confusion Matrices for Ganga River Water Quality assessment and forecasting.

Based on Central Pollution Control Board (CPCB, India) water quality standards:
1. Binary Water Quality Classification:
   - Compliant / Safe (DO >= 5.0 mg/L): Suitable for outdoor bathing (Class B)
     and healthy aquatic ecosystems.
   - Non-Compliant / Critical (DO < 5.0 mg/L): Hypoxic or polluted conditions
     requiring ecological intervention.

2. Multi-Class Water Quality Grading (CPCB Tiers):
   - Critical / Impaired (DO < 4.0 mg/L)
   - Moderate / Acceptable (4.0 <= DO < 6.0 mg/L)
   - Good / Pristine (DO >= 6.0 mg/L)

Authors: duttadhiraj-design
Repository: https://github.com/duttadhiraj-design/Ganga_Water_Quality_Forecasting
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Optional scikit-learn and seaborn imports with robust native fallbacks
try:
    from sklearn.metrics import (
        confusion_matrix as sk_confusion_matrix,
        classification_report as sk_classification_report,
        accuracy_score,
        precision_recall_fscore_support,
    )
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.tree import DecisionTreeClassifier
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    import seaborn as sns
    SEABORN_AVAILABLE = True
except ImportError:
    SEABORN_AVAILABLE = False


# ==============================================================================
# 1. WATER QUALITY CATEGORIZATION HELPERS (CPCB STANDARDS)
# ==============================================================================

def categorize_binary(do_values, threshold=5.0):
    """
    Categorize Dissolved Oxygen (DO) values according to CPCB bathing standard.
    
    Parameters:
        do_values (array-like): Dissolved Oxygen readings in mg/L.
        threshold (float): Minimum DO threshold (default: 5.0 mg/L).
        
    Returns:
        np.ndarray: String labels ('Non-Compliant (<5.0)' or 'Compliant (>=5.0)').
    """
    arr = np.asarray(do_values, dtype=float)
    return np.where(arr >= threshold, "Compliant (>=5.0 mg/L)", "Non-Compliant (<5.0 mg/L)")


def categorize_multiclass(do_values):
    """
    Categorize Dissolved Oxygen into CPCB multi-tier water quality classes:
    - Critical / Impaired: DO < 4.0 mg/L
    - Moderate / Acceptable: 4.0 <= DO < 6.0 mg/L
    - Good / Pristine: DO >= 6.0 mg/L
    
    Parameters:
        do_values (array-like): Dissolved Oxygen readings in mg/L.
        
    Returns:
        np.ndarray: String labels of the assigned water quality class.
    """
    arr = np.asarray(do_values, dtype=float)
    classes = np.empty(arr.shape, dtype=object)
    classes[arr < 4.0] = "Critical (<4.0 mg/L)"
    classes[(arr >= 4.0) & (arr < 6.0)] = "Moderate (4.0-6.0 mg/L)"
    classes[arr >= 6.0] = "Good (>=6.0 mg/L)"
    return classes


# ==============================================================================
# 2. CONFUSION MATRIX COMPUTATION & PERFORMANCE METRICS
# ==============================================================================

def compute_confusion_matrix(y_true, y_pred, labels=None):
    """
    Compute confusion matrix where rows are actual classes and columns are predicted.
    
    Parameters:
        y_true (array-like): Ground truth target labels.
        y_pred (array-like): Estimated targets by model/classifier.
        labels (list, optional): List of unique labels to index the matrix.
        
    Returns:
        tuple: (cm_matrix (2D np.ndarray), labels_list)
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if labels is None:
        labels = sorted(list(set(y_true) | set(y_pred)))
    else:
        labels = list(labels)

    if SKLEARN_AVAILABLE:
        cm = sk_confusion_matrix(y_true, y_pred, labels=labels)
    else:
        label_to_idx = {lbl: i for i, lbl in enumerate(labels)}
        n = len(labels)
        cm = np.zeros((n, n), dtype=int)
        for t, p in zip(y_true, y_pred):
            if t in label_to_idx and p in label_to_idx:
                cm[label_to_idx[t], label_to_idx[p]] += 1

    return cm, labels


def compute_metrics(y_true, y_pred, labels=None):
    """
    Compute Accuracy, Precision, Recall, Specificity, and F1-Score.
    
    Returns:
        dict: Summary containing per-class and overall metrics.
    """
    cm, labels = compute_confusion_matrix(y_true, y_pred, labels=labels)
    n_classes = len(labels)
    total_samples = np.sum(cm)
    accuracy = float(np.trace(cm) / total_samples) if total_samples > 0 else 0.0

    per_class = {}
    precisions = []
    recalls = []
    f1s = []

    for i, lbl in enumerate(labels):
        tp = cm[i, i]
        fn = np.sum(cm[i, :]) - tp
        fp = np.sum(cm[:, i]) - tp
        tn = total_samples - (tp + fn + fp)

        precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
        f1 = float(2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        support = int(np.sum(cm[i, :]))

        per_class[lbl] = {
            "TP": int(tp),
            "FP": int(fp),
            "FN": int(fn),
            "TN": int(tn),
            "Precision": precision,
            "Recall (Sensitivity)": recall,
            "Specificity": specificity,
            "F1-Score": f1,
            "Support": support,
        }
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)

    macro_precision = float(np.mean(precisions))
    macro_recall = float(np.mean(recalls))
    macro_f1 = float(np.mean(f1s))

    return {
        "Confusion_Matrix": cm,
        "Labels": labels,
        "Accuracy": accuracy,
        "Macro_Precision": macro_precision,
        "Macro_Recall": macro_recall,
        "Macro_F1": macro_f1,
        "Per_Class": per_class,
    }


def print_classification_report(metrics_dict, title="Classification Report"):
    """
    Print a neatly formatted classification report and confusion matrix summary.
    """
    print("\n" + "=" * 76)
    print(f" {title.upper()} ")
    print("=" * 76)
    
    labels = metrics_dict["Labels"]
    cm = metrics_dict["Confusion_Matrix"]
    
    print("\n--- Confusion Matrix ---")
    header = f"{'Actual \\ Predicted':<28}" + "".join([f"{lbl[:14]:>16}" for lbl in labels])
    print(header)
    print("-" * len(header))
    for i, actual_lbl in enumerate(labels):
        row_str = f"{actual_lbl:<28}" + "".join([f"{cm[i, j]:>16d}" for j in range(len(labels))])
        print(row_str)
        
    print("\n--- Detailed Classification Metrics ---")
    fmt = "{:<28} {:>10} {:>10} {:>10} {:>10} {:>10}"
    print(fmt.format("Class", "Precision", "Recall", "F1-Score", "Specific.", "Support"))
    print("-" * 84)
    for lbl, m in metrics_dict["Per_Class"].items():
        print(fmt.format(
            lbl[:27],
            f"{m['Precision']:.4f}",
            f"{m['Recall (Sensitivity)']:.4f}",
            f"{m['F1-Score']:.4f}",
            f"{m['Specificity']:.4f}",
            str(m['Support']),
        ))
    print("-" * 84)
    print(f"{'Overall Accuracy':<28} {metrics_dict['Accuracy']:.4f} ({np.sum(cm)} samples)")
    print(f"{'Macro Avg Precision':<28} {metrics_dict['Macro_Precision']:.4f}")
    print(f"{'Macro Avg Recall':<28} {metrics_dict['Macro_Recall']:.4f}")
    print(f"{'Macro Avg F1-Score':<28} {metrics_dict['Macro_F1']:.4f}")
    print("=" * 76 + "\n")


# ==============================================================================
# 3. CONFUSION MATRIX VISUALIZATION (HEATMAPS)
# ==============================================================================

def draw_confusion_matrix_axes(ax, cm, labels, title, normalize=False, cmap_name="Blues"):
    """
    Render a polished confusion matrix on a matplotlib axes with counts and percentages.
    """
    n = len(labels)
    if normalize:
        row_sums = cm.sum(axis=1, keepdims=True)
        display_data = np.divide(cm.astype(float), row_sums, out=np.zeros_like(cm, dtype=float), where=row_sums != 0)
        vmax = 1.0
    else:
        display_data = cm
        vmax = float(np.max(cm)) if np.max(cm) > 0 else 1.0

    cmap = plt.get_cmap(cmap_name)
    im = ax.imshow(display_data, interpolation="nearest", cmap=cmap, vmin=0, vmax=vmax)

    # Set title and ticks
    ax.set_title(title, fontsize=12, pad=12, weight="bold")
    tick_marks = np.arange(n)
    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)
    ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=9)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("Predicted Label", fontsize=10, weight="semibold", labelpad=8)
    ax.set_ylabel("Actual (Ground Truth) Label", fontsize=10, weight="semibold", labelpad=8)

    # Add numeric cell annotations
    total = np.sum(cm)
    for i in range(n):
        for j in range(n):
            count = cm[i, j]
            pct = (count / total * 100) if total > 0 else 0
            
            if normalize:
                cell_text = f"{display_data[i, j]:.1%}\n({count})"
                text_color = "white" if display_data[i, j] > 0.5 * vmax else "black"
            else:
                cell_text = f"{count}\n({pct:.1f}%)"
                text_color = "white" if display_data[i, j] > 0.5 * vmax else "black"
                
            ax.text(j, i, cell_text, ha="center", va="center", color=text_color, fontsize=9, weight="bold")

    # Add a thin grid
    ax.set_xticks(np.arange(n + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(n + 1) - 0.5, minor=True)
    ax.grid(which="minor", color="w", linestyle="-", linewidth=1.5)
    ax.tick_params(which="minor", bottom=False, left=False)
    
    return im


def plot_confusion_matrices(binary_metrics, multiclass_metrics, save_path="confusion_matrix_results.png"):
    """
    Generate and save a 2x2 comprehensive dashboard:
    - Top Left: Binary Confusion Matrix (Raw counts)
    - Top Right: Binary Confusion Matrix (Normalized by true class recall)
    - Bottom Left: Multi-Class Confusion Matrix (Raw counts)
    - Bottom Right: Multi-Class Confusion Matrix (Normalized)
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # 1. Binary Raw
    im1 = draw_confusion_matrix_axes(
        axes[0, 0],
        binary_metrics["Confusion_Matrix"],
        binary_metrics["Labels"],
        title="Binary Classification (CPCB DO >= 5.0 mg/L) - Raw Counts",
        normalize=False,
        cmap_name="Blues",
    )
    plt.colorbar(im1, ax=axes[0, 0], fraction=0.046, pad=0.04)

    # 2. Binary Normalized
    im2 = draw_confusion_matrix_axes(
        axes[0, 1],
        binary_metrics["Confusion_Matrix"],
        binary_metrics["Labels"],
        title="Binary Classification (CPCB DO >= 5.0 mg/L) - Normalized (Recall)",
        normalize=True,
        cmap_name="Blues",
    )
    plt.colorbar(im2, ax=axes[0, 1], fraction=0.046, pad=0.04)

    # 3. Multi-Class Raw
    im3 = draw_confusion_matrix_axes(
        axes[1, 0],
        multiclass_metrics["Confusion_Matrix"],
        multiclass_metrics["Labels"],
        title="Multi-Class Water Quality Tiers - Raw Counts",
        normalize=False,
        cmap_name="Greens",
    )
    plt.colorbar(im3, ax=axes[1, 0], fraction=0.046, pad=0.04)

    # 4. Multi-Class Normalized
    im4 = draw_confusion_matrix_axes(
        axes[1, 1],
        multiclass_metrics["Confusion_Matrix"],
        multiclass_metrics["Labels"],
        title="Multi-Class Water Quality Tiers - Normalized (Recall)",
        normalize=True,
        cmap_name="Greens",
    )
    plt.colorbar(im4, ax=axes[1, 1], fraction=0.046, pad=0.04)

    plt.suptitle(
        "Ganga River Water Quality Forecasting - Confusion Matrix Performance Dashboard\n"
        f"Binary Accuracy: {binary_metrics['Accuracy']:.2%} | Multi-Class Accuracy: {multiclass_metrics['Accuracy']:.2%}",
        fontsize=14,
        weight="bold",
        y=0.995,
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"[+] Confusion matrix dashboard saved to: {save_path}")
    plt.close()


# ==============================================================================
# 4. DATA LOADER & DATASET RESOLUTION
# ==============================================================================

def load_or_generate_dataset():
    """
    Attempt to load preprocessed Ganga datasets:
    1. ganga_model_ready_zero_nan.csv
    2. ganga_supervised_preprocessed.csv
    3. ganga_cleaned_panel.csv
    
    If none are present, generate a realistic benchmark dataset of Ganga monitoring
    stations across Uttarakhand (UK), Uttar Pradesh (UP), and West Bengal (WB).
    """
    candidates = [
        "ganga_model_ready_zero_nan.csv",
        "ganga_supervised_preprocessed.csv",
        "ganga_cleaned_panel.csv",
    ]
    for path in candidates:
        if os.path.exists(path):
            print(f"[+] Loaded existing Ganga dataset from: {path}")
            df = pd.read_csv(path)
            if "Target_DO_next" in df.columns:
                return df

    print("[*] No pre-generated dataset file found. Generating realistic benchmark Ganga dataset...")
    np.random.seed(42)

    stations = [
        ("UK", "STN_01", "Rishikesh Upstream"),
        ("UK", "STN_02", "Haridwar D/S"),
        ("UP", "STN_03", "Kanpur Bithoor"),
        ("UP", "STN_04", "Kanpur Jajmau"),
        ("UP", "STN_05", "Prayagraj Sangam"),
        ("UP", "STN_06", "Varanasi Assi Ghat"),
        ("WB", "STN_07", "Dakshineswar"),
        ("WB", "STN_08", "Howrah Bridge"),
    ]

    records = []
    # Generate 6 years (2020-2025) x 24 half-month rounds for each station
    for state, code, name in stations:
        # State-specific baseline parameters
        base_do = 8.5 if state == "UK" else (6.2 if state == "UP" else 5.8)
        base_bod = 1.5 if state == "UK" else (3.8 if state == "UP" else 3.2)
        base_ph = 7.8 if state == "UK" else (7.6 if state == "UP" else 7.4)

        for year in range(2020, 2026):
            for period_idx in range(24):
                month = (period_idx // 2) + 1
                rnd = (period_idx % 2) + 1
                time_step = year * 24 + period_idx

                # Seasonal cyclical pattern (DO higher in winter, lower in monsoon/summer)
                season_effect = 1.2 * np.cos(2 * np.pi * period_idx / 24.0)
                do_val = max(1.5, base_do + season_effect + np.random.normal(0, 0.6))
                bod_val = max(0.5, base_bod - 0.4 * season_effect + np.random.normal(0, 0.4))
                ph_val = max(6.0, min(9.0, base_ph + np.random.normal(0, 0.2)))

                records.append({
                    "state": state,
                    "stn_code": code,
                    "stn_name": name,
                    "year": year,
                    "month": month,
                    "round": rnd,
                    "period_idx": period_idx,
                    "time_step": time_step,
                    "DO": round(do_val, 2),
                    "BOD": round(bod_val, 2),
                    "pH": round(ph_val, 2),
                })

    df = pd.DataFrame(records).sort_values(["stn_code", "time_step"]).reset_index(drop=True)
    df["DO_lag1"] = df.groupby("stn_code")["DO"].shift(1)
    df["DO_diff1"] = df["DO"] - df["DO_lag1"]
    df["BOD_to_DO_ratio"] = df["BOD"] / (df["DO"] + 1e-5)
    df["sin_season"] = np.sin(2 * np.pi * df["period_idx"] / 24.0)
    df["cos_season"] = np.cos(2 * np.pi * df["period_idx"] / 24.0)
    df["Target_DO_next"] = df.groupby("stn_code")["DO"].shift(-1)
    df["Target_year"] = df.groupby("stn_code")["year"].shift(-1)

    df = df.dropna().reset_index(drop=True)
    print(f"[+] Benchmark Ganga dataset successfully constructed: {len(df)} records across 8 stations.")
    return df


# ==============================================================================
# 5. WATER QUALITY FORECAST EVALUATION PIPELINE
# ==============================================================================

def evaluate_water_quality_confusion_matrix(df=None, save_path="confusion_matrix_results.png"):
    """
    Main evaluation pipeline:
    1. Loads Ganga river dataset (or builds benchmark).
    2. Chronologically splits Train (<=2023), Validation (2024), and Test (2025).
    3. Simulates / fits the water quality forecasting model.
    4. Evaluates predictions against ground truth target DO using Confusion Matrix:
       - Binary compliance (DO >= 5.0 mg/L)
       - Multi-class tiers (Critical, Moderate, Good)
    5. Outputs complete classification metrics and saves visual charts.
    """
    if df is None:
        df = load_or_generate_dataset()

    # Chronological train / test split
    train_df = df[df["Target_year"] <= 2023].copy()
    test_df = df[df["Target_year"] == 2025].copy()
    if len(test_df) == 0:
        test_df = df[df["year"] == df["year"].max()].copy()
        train_df = df[df["year"] < df["year"].max()].copy()

    feature_cols = [
        "DO", "BOD", "pH", "DO_lag1", "DO_diff1",
        "BOD_to_DO_ratio", "sin_season", "cos_season"
    ]
    # Filter features that exist
    features = [c for c in feature_cols if c in df.columns]

    y_train = train_df["Target_DO_next"].values
    y_test_true = test_df["Target_DO_next"].values

    # Fit a model to obtain predictions on test set
    if SKLEARN_AVAILABLE and len(features) > 0:
        X_train = train_df[features].fillna(train_df[features].median()).values
        X_test = test_df[features].fillna(train_df[features].median()).values
        
        # Train Random Forest Regressor to forecast continuous Target DO
        from sklearn.ensemble import GradientBoostingRegressor
        model = GradientBoostingRegressor(n_estimators=100, max_depth=4, random_state=42)
        model.fit(X_train, y_train)
        y_test_pred = model.predict(X_test)
        model_name = "Gradient Boosting Regressor (CPCB Standard Mapping)"
    else:
        # Autoregressive persistence baseline (DO_next ~ DO_t + trend) with slight noise
        np.random.seed(42)
        trend = (test_df["DO"] - test_df.get("DO_lag1", test_df["DO"])).fillna(0).values * 0.4
        y_test_pred = test_df["DO"].values + trend + np.random.normal(0, 0.35, size=len(test_df))
        model_name = "Autoregressive Persistence Forecast"

    print(f"\n[+] Evaluated Model: {model_name}")
    print(f"[+] Total Test Samples (2025 Round Evaluations): {len(y_test_true)}")

    # 1. Binary Classification: Compliant (>= 5.0 mg/L) vs Non-Compliant (< 5.0 mg/L)
    binary_true = categorize_binary(y_test_true, threshold=5.0)
    binary_pred = categorize_binary(y_test_pred, threshold=5.0)
    binary_labels = ["Compliant (>=5.0 mg/L)", "Non-Compliant (<5.0 mg/L)"]

    binary_metrics = compute_metrics(binary_true, binary_pred, labels=binary_labels)
    print_classification_report(
        binary_metrics,
        title="Binary Water Quality Confusion Matrix (DO >= 5.0 mg/L Compliance)"
    )

    # 2. Multi-Class Classification: Critical (<4.0), Moderate (4.0-6.0), Good (>=6.0)
    multi_true = categorize_multiclass(y_test_true)
    multi_pred = categorize_multiclass(y_test_pred)
    multi_labels = ["Good (>=6.0 mg/L)", "Moderate (4.0-6.0 mg/L)", "Critical (<4.0 mg/L)"]

    multi_metrics = compute_metrics(multi_true, multi_pred, labels=multi_labels)
    print_classification_report(
        multi_metrics,
        title="Multi-Class Water Quality Confusion Matrix (CPCB Ecological Tiers)"
    )

    # 3. Plot and save visualization dashboard
    plot_confusion_matrices(binary_metrics, multi_metrics, save_path=save_path)

    return {
        "binary_metrics": binary_metrics,
        "multiclass_metrics": multi_metrics,
        "y_true": y_test_true,
        "y_pred": y_test_pred,
    }


# ==============================================================================
# MAIN ENTRY POINT
# ==============================================================================

if __name__ == "__main__":
    print("=" * 76)
    print(" Ganga Water Quality Forecasting - Confusion Matrix Analysis ")
    print("=" * 76)
    results = evaluate_water_quality_confusion_matrix()
    print("[+] Evaluation complete!")
