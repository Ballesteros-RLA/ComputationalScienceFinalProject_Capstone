# =============================================================================
# STROKE RISK PREDICTION USING K-NEAREST NEIGHBORS (KNN)
# Bachelor of Science in Computer Science — Capstone Project
# May 2026
# =============================================================================

# =============================================================================
# STEP 1 — LIBRARY IMPORTS
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report,
)

# Global plot style
plt.rcParams.update({
    "figure.facecolor": "#f8f9fa",
    "axes.facecolor":   "#ffffff",
    "axes.edgecolor":   "#cccccc",
    "axes.labelcolor":  "#333333",
    "xtick.color":      "#555555",
    "ytick.color":      "#555555",
    "font.family":      "DejaVu Sans",
    "axes.spines.top":  False,
    "axes.spines.right": False,
})

PALETTE_MAIN  = ["#2196F3", "#F44336"]   # blue = no stroke, red = stroke
PALETTE_GRAD  = "Blues"
ACCENT        = "#1565C0"
OUTPUT_DPI    = 150

print("=" * 60)
print("  STROKE RISK PREDICTION — KNN CAPSTONE")
print("=" * 60)

# =============================================================================
# STEP 2 — DATA LOADING
# =============================================================================

df = pd.read_csv(r"c:\Users\r3nz3\OneDrive\Desktop\CompSci codes\CAPSTONE_COMP_SCI\healthcare-dataset-stroke-data.csv")

# Drop spurious extra column if present
df = df.loc[:, ~df.columns.str.startswith("Unnamed")]

print(f"\n[DATA] Shape: {df.shape}")
print(df.head())

# =============================================================================
# STEP 3 — EXPLORATORY DATA ANALYSIS (EDA)
# =============================================================================

print("\n[EDA] Missing values per column:")
print(df.isnull().sum())

print("\n[EDA] Class distribution:")
vc = df["stroke"].value_counts()
print(vc)
print(f"  No Stroke : {vc[0]} ({vc[0]/len(df)*100:.1f}%)")
print(f"  Stroke    : {vc[1]} ({vc[1]/len(df)*100:.1f}%)")

print("\n[EDA] Descriptive statistics:")
print(df.describe())

# --------------------------------------------------------------------------- #
# VISUALIZATION 1 — Class Distribution Bar Chart
# --------------------------------------------------------------------------- #
fig, ax = plt.subplots(figsize=(7, 5))
bars = ax.bar(["No Stroke (0)", "Stroke (1)"],
              [vc[0], vc[1]],
              color=PALETTE_MAIN, edgecolor="white", linewidth=1.5, width=0.5)

for bar, count in zip(bars, [vc[0], vc[1]]):
    ax.text(bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 50,
            f"{count}\n({count/len(df)*100:.1f}%)",
            ha="center", va="bottom", fontsize=11, fontweight="bold", color="#333")

ax.set_title("Class Distribution: Stroke vs No Stroke", fontsize=14, fontweight="bold", pad=15)
ax.set_ylabel("Number of Patients", fontsize=12)
ax.set_ylim(0, vc[0] * 1.15)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{int(x):,}"))
fig.tight_layout()
plt.savefig("viz1_class_distribution.png", dpi=OUTPUT_DPI, bbox_inches="tight")
plt.show()
print("[VIZ] Saved → viz1_class_distribution.png")

# =============================================================================
# STEP 4 — DATA PREPROCESSING
# =============================================================================

# 4a. Drop id (no predictive value)
df = df.drop(columns=["id"])

# 4b. Handle missing BMI via mean imputation
bmi_mean = df["bmi"].mean()
df["bmi"] = df["bmi"].fillna(bmi_mean)
print(f"\n[PREPROCESS] BMI missing values filled with mean = {bmi_mean:.2f}")

# 4c. Label-encode all categorical columns
categorical_cols = ["gender", "ever_married", "work_type", "Residence_type", "smoking_status"]
le = LabelEncoder()
encoding_map = {}
for col in categorical_cols:
    df[col] = le.fit_transform(df[col].astype(str))
    encoding_map[col] = dict(zip(le.classes_, le.transform(le.classes_)))
    print(f"  {col}: {encoding_map[col]}")

# 4d. Separate features and target
X = df.drop(columns=["stroke"])
y = df["stroke"]

# 4e. Min-Max normalization (critical for KNN distance computation)
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)
X_scaled = pd.DataFrame(X_scaled, columns=X.columns)
print("\n[PREPROCESS] Features normalized with MinMaxScaler → range [0, 1]")

# 4f. Stratified train-test split (80 / 20)
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)
print(f"[PREPROCESS] Train: {X_train.shape[0]} | Test: {X_test.shape[0]}")

# --------------------------------------------------------------------------- #
# VISUALIZATION 2 — Feature Correlation Heatmap
# --------------------------------------------------------------------------- #
fig, ax = plt.subplots(figsize=(10, 8))
corr = df.corr(numeric_only=True)
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(
    corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm",
    linewidths=0.5, linecolor="#eeeeee",
    vmin=-1, vmax=1, ax=ax,
    annot_kws={"size": 8}
)
ax.set_title("Feature Correlation Heatmap", fontsize=14, fontweight="bold", pad=15)
fig.tight_layout()
plt.savefig("viz2_correlation_heatmap.png", dpi=OUTPUT_DPI, bbox_inches="tight")
plt.show()
print("[VIZ] Saved → viz2_correlation_heatmap.png")

# --------------------------------------------------------------------------- #
# VISUALIZATION 3 — Age Distribution by Stroke Status
# --------------------------------------------------------------------------- #
fig, ax = plt.subplots(figsize=(9, 5))
for label, color, name in zip([0, 1], PALETTE_MAIN, ["No Stroke", "Stroke"]):
    subset = df[df["stroke"] == label]["age"]
    ax.hist(subset, bins=30, alpha=0.7, color=color, label=f"{name} (n={len(subset)})",
            edgecolor="white", linewidth=0.5)
ax.set_title("Age Distribution by Stroke Status", fontsize=14, fontweight="bold", pad=15)
ax.set_xlabel("Age (years)", fontsize=12)
ax.set_ylabel("Count", fontsize=12)
ax.legend(fontsize=11)
fig.tight_layout()
plt.savefig("viz3_age_distribution.png", dpi=OUTPUT_DPI, bbox_inches="tight")
plt.show()
print("[VIZ] Saved → viz3_age_distribution.png")

# --------------------------------------------------------------------------- #
# VISUALIZATION 4 — Average Glucose Level by Stroke Status (Box Plot)
# --------------------------------------------------------------------------- #
fig, ax = plt.subplots(figsize=(7, 5))
data_glucose = [
    df[df["stroke"] == 0]["avg_glucose_level"].values,
    df[df["stroke"] == 1]["avg_glucose_level"].values,
]
bp = ax.boxplot(data_glucose, patch_artist=True, notch=False, widths=0.4,
                medianprops=dict(color="white", linewidth=2.5))
for patch, color in zip(bp["boxes"], PALETTE_MAIN):
    patch.set_facecolor(color)
    patch.set_alpha(0.8)
for element in ["whiskers", "caps", "fliers"]:
    for item in bp[element]:
        item.set_color("#888888")
ax.set_xticklabels(["No Stroke", "Stroke"], fontsize=12)
ax.set_title("Avg. Glucose Level by Stroke Status", fontsize=14, fontweight="bold", pad=15)
ax.set_ylabel("Average Glucose Level (mg/dL)", fontsize=12)
fig.tight_layout()
plt.savefig("viz4_glucose_boxplot.png", dpi=OUTPUT_DPI, bbox_inches="tight")
plt.show()
print("[VIZ] Saved → viz4_glucose_boxplot.png")

# =============================================================================
# STEP 5 — K VALUE EXPERIMENTATION
# =============================================================================

k_values     = list(range(1, 22, 2))   # odd k: 1, 3, 5, … 21
k_accuracy   = []
k_precision  = []
k_recall     = []
k_f1         = []

# Handle class imbalance with SMOTE (mirrors report's "class_weight='balanced'")
try:
    from imblearn.over_sampling import SMOTE
    sm = SMOTE(random_state=42)
    X_train_res, y_train_res = sm.fit_resample(X_train, y_train)
    print(f"[PREPROCESS] SMOTE applied → Training set balanced: {dict(y_train_res.value_counts())}")
except ImportError:
    # Fallback: duplicate minority samples
    from sklearn.utils import resample
    X_tr = pd.concat([X_train, y_train], axis=1)
    minority = X_tr[X_tr["stroke"] == 1]
    majority = X_tr[X_tr["stroke"] == 0]
    minority_up = resample(minority, replace=True, n_samples=len(majority), random_state=42)
    X_tr_bal = pd.concat([majority, minority_up])
    X_train_res = X_tr_bal.drop(columns=["stroke"])
    y_train_res = X_tr_bal["stroke"]
    print(f"[PREPROCESS] Upsampling fallback applied → {dict(y_train_res.value_counts())}")

print("\n[EXPERIMENT] Testing k values …")
print(f"{'k':>4} | {'Accuracy':>9} | {'Precision':>9} | {'Recall':>7} | {'F1':>7}")
print("-" * 48)

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k, metric="euclidean",
                               weights="uniform")
    knn.fit(X_train_res, y_train_res)
    yp = knn.predict(X_test)

    acc  = accuracy_score(y_test, yp)
    prec = precision_score(y_test, yp, zero_division=0)
    rec  = recall_score(y_test, yp, zero_division=0)
    f1   = f1_score(y_test, yp, zero_division=0)

    k_accuracy.append(acc)
    k_precision.append(prec)
    k_recall.append(rec)
    k_f1.append(f1)

    print(f"  k={k:>2} | {acc*100:>8.2f}% | {prec*100:>8.2f}% | {rec*100:>6.2f}% | {f1:.4f}")

best_k_idx = k_f1.index(max(k_f1))
best_k     = k_values[best_k_idx]
print(f"\n[EXPERIMENT] Best k = {best_k}  (highest F1-Score = {max(k_f1):.4f})")

# --------------------------------------------------------------------------- #
# VISUALIZATION 5 — k vs. Performance Metrics
# --------------------------------------------------------------------------- #
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(k_values, [a * 100 for a in k_accuracy],  "o-", label="Accuracy (%)",  color="#1976D2", lw=2)
ax.plot(k_values, [p * 100 for p in k_precision], "s--", label="Precision (%)", color="#388E3C", lw=2)
ax.plot(k_values, [r * 100 for r in k_recall],    "^-.", label="Recall (%)",    color="#F44336", lw=2)
ax.plot(k_values, [f * 100 for f in k_f1],        "d:",  label="F1-Score (%)", color="#7B1FA2", lw=2)
ax.axvline(best_k, color="gray", linestyle="--", alpha=0.6, label=f"Best k={best_k}")
ax.set_title("KNN Performance Metrics vs. k Value", fontsize=14, fontweight="bold", pad=15)
ax.set_xlabel("k (Number of Neighbors)", fontsize=12)
ax.set_ylabel("Score (%)", fontsize=12)
ax.set_xticks(k_values)
ax.legend(fontsize=10)
fig.tight_layout()
plt.savefig("viz5_k_vs_metrics.png", dpi=OUTPUT_DPI, bbox_inches="tight")
plt.show()
print("[VIZ] Saved → viz5_k_vs_metrics.png")

# =============================================================================
# STEP 6 — FINAL MODEL TRAINING (k = 7)
# =============================================================================

print(f"\n[MODEL] Training final KNN with k={best_k} …")
model = KNeighborsClassifier(
    n_neighbors=best_k,
    metric="euclidean",
    weights="uniform",
)
model.fit(X_train_res, y_train_res)
y_pred = model.predict(X_test)
print("[MODEL] Training complete.")

# =============================================================================
# STEP 7 — MODEL EVALUATION
# =============================================================================

acc  = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, zero_division=0)
rec  = recall_score(y_test, y_pred, zero_division=0)
f1   = f1_score(y_test, y_pred, zero_division=0)
cm   = confusion_matrix(y_test, y_pred)

print("\n" + "=" * 60)
print("  FINAL MODEL EVALUATION")
print("=" * 60)
print(f"  Accuracy  : {acc*100:.2f}%")
print(f"  Precision : {prec*100:.2f}%")
print(f"  Recall    : {rec*100:.2f}%")
print(f"  F1-Score  : {f1:.4f}")
print("\n  Confusion Matrix:")
print(f"    TN={cm[0,0]}  FP={cm[0,1]}")
print(f"    FN={cm[1,0]}  TP={cm[1,1]}")
print("\n  Classification Report:")
print(classification_report(y_test, y_pred,
                             target_names=["No Stroke", "Stroke"]))

# --------------------------------------------------------------------------- #
# VISUALIZATION 6 — Confusion Matrix (styled)
# --------------------------------------------------------------------------- #
fig, ax = plt.subplots(figsize=(6, 5))
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                              display_labels=["No Stroke", "Stroke"])
disp.plot(ax=ax, cmap="Blues", colorbar=False, values_format="d")
ax.set_title(f"Confusion Matrix  (k={best_k})", fontsize=14,
             fontweight="bold", pad=15)
# Annotate cells
labels = [["True Negative\n(TN)", "False Positive\n(FP)"],
          ["False Negative\n(FN)", "True Positive\n(TP)"]]
for i in range(2):
    for j in range(2):
        ax.text(j, i + 0.30, labels[i][j], ha="center", va="center",
                fontsize=8, color="gray")
fig.tight_layout()
plt.savefig("viz6_confusion_matrix.png", dpi=OUTPUT_DPI, bbox_inches="tight")
plt.show()
print("[VIZ] Saved → viz6_confusion_matrix.png")

# --------------------------------------------------------------------------- #
# VISUALIZATION 7 — Performance Metrics Bar Chart (Final Model)
# --------------------------------------------------------------------------- #
metrics = {
    "Accuracy":  acc,
    "Precision": prec,
    "Recall":    rec,
    "F1-Score":  f1,
}
fig, ax = plt.subplots(figsize=(7, 5))
colors  = ["#1976D2", "#388E3C", "#F44336", "#7B1FA2"]
bars    = ax.bar(metrics.keys(), [v * 100 for v in metrics.values()],
                 color=colors, edgecolor="white", linewidth=1.5, width=0.5)
for bar, val in zip(bars, metrics.values()):
    ax.text(bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.8,
            f"{val*100:.2f}%",
            ha="center", va="bottom", fontsize=11, fontweight="bold")
ax.set_ylim(0, 115)
ax.set_title(f"Final KNN Model Performance  (k={best_k})",
             fontsize=14, fontweight="bold", pad=15)
ax.set_ylabel("Score (%)", fontsize=12)
ax.axhline(y=100, color="gray", linestyle="--", alpha=0.3)
fig.tight_layout()
plt.savefig("viz7_final_metrics.png", dpi=OUTPUT_DPI, bbox_inches="tight")
plt.show()
print("[VIZ] Saved → viz7_final_metrics.png")

# --------------------------------------------------------------------------- #
# VISUALIZATION 8 — Feature Importance via Correlation with Stroke Target
# --------------------------------------------------------------------------- #
corr_stroke = df.corr(numeric_only=True)["stroke"].drop("stroke").abs().sort_values(ascending=True)
fig, ax = plt.subplots(figsize=(8, 6))
colors_corr = [PALETTE_MAIN[1] if c >= corr_stroke.median() else PALETTE_MAIN[0]
               for c in corr_stroke]
ax.barh(corr_stroke.index, corr_stroke.values, color=colors_corr,
        edgecolor="white", linewidth=0.8)
for i, (name, val) in enumerate(corr_stroke.items()):
    ax.text(val + 0.003, i, f"{val:.3f}", va="center", fontsize=9)
ax.set_title("Feature Importance (|Correlation with Stroke|)",
             fontsize=14, fontweight="bold", pad=15)
ax.set_xlabel("Absolute Pearson Correlation", fontsize=12)
ax.set_xlim(0, corr_stroke.max() + 0.06)
fig.tight_layout()
plt.savefig("viz8_feature_importance.png", dpi=OUTPUT_DPI, bbox_inches="tight")
plt.show()
print("[VIZ] Saved → viz8_feature_importance.png")

# --------------------------------------------------------------------------- #
# VISUALIZATION 9 — BMI vs Age Scatter (colored by stroke)
# --------------------------------------------------------------------------- #
fig, ax = plt.subplots(figsize=(8, 6))
for label, color, name, alpha in zip([0, 1], PALETTE_MAIN,
                                     ["No Stroke", "Stroke"],
                                     [0.3, 0.9]):
    subset = df[df["stroke"] == label]
    ax.scatter(subset["age"], subset["bmi"], c=color, label=name,
               alpha=alpha, s=20 if label == 0 else 50,
               edgecolors="none" if label == 0 else "white",
               linewidths=0.5)
ax.set_title("Age vs. BMI by Stroke Status", fontsize=14, fontweight="bold", pad=15)
ax.set_xlabel("Age (years)", fontsize=12)
ax.set_ylabel("BMI (kg/m²)", fontsize=12)
ax.legend(fontsize=11)
fig.tight_layout()
plt.savefig("viz9_age_bmi_scatter.png", dpi=OUTPUT_DPI, bbox_inches="tight")
plt.show()
print("[VIZ] Saved → viz9_age_bmi_scatter.png")

# --------------------------------------------------------------------------- #
# VISUALIZATION 10 — Comprehensive Dashboard (summary)
# --------------------------------------------------------------------------- #
fig = plt.figure(figsize=(16, 10))
fig.patch.set_facecolor("#f8f9fa")
fig.suptitle(
    f"Stroke Risk Prediction — KNN Dashboard  (k={best_k})",
    fontsize=16, fontweight="bold", y=0.98
)
gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

# Panel 1 — Class distribution
ax1 = fig.add_subplot(gs[0, 0])
ax1.bar(["No Stroke", "Stroke"], [vc[0], vc[1]], color=PALETTE_MAIN,
        edgecolor="white", width=0.5)
ax1.set_title("Class Distribution", fontweight="bold")
ax1.set_ylabel("Count")
for i, v in enumerate([vc[0], vc[1]]):
    ax1.text(i, v + 30, f"{v}", ha="center", fontsize=9, fontweight="bold")

# Panel 2 — k vs Accuracy & Recall
ax2 = fig.add_subplot(gs[0, 1])
ax2.plot(k_values, [a * 100 for a in k_accuracy], "o-", color="#1976D2",
         label="Accuracy", lw=2)
ax2.plot(k_values, [r * 100 for r in k_recall], "^-.", color="#F44336",
         label="Recall", lw=2)
ax2.axvline(best_k, color="gray", linestyle="--", alpha=0.5)
ax2.set_title("k vs. Accuracy & Recall", fontweight="bold")
ax2.set_xlabel("k")
ax2.set_ylabel("Score (%)")
ax2.set_xticks(k_values)
ax2.tick_params(axis="x", labelsize=7)
ax2.legend(fontsize=9)

# Panel 3 — Confusion Matrix
ax3 = fig.add_subplot(gs[0, 2])
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax3,
            xticklabels=["No Stroke", "Stroke"],
            yticklabels=["No Stroke", "Stroke"],
            linewidths=0.5, linecolor="#eeeeee", cbar=False,
            annot_kws={"size": 13, "weight": "bold"})
ax3.set_title("Confusion Matrix", fontweight="bold")
ax3.set_xlabel("Predicted")
ax3.set_ylabel("Actual")

# Panel 4 — Feature correlation bar
ax4 = fig.add_subplot(gs[1, 0])
corr_vals  = corr_stroke.values
corr_feats = corr_stroke.index
colors_bar = [PALETTE_MAIN[1] if v >= corr_stroke.median() else PALETTE_MAIN[0]
              for v in corr_vals]
ax4.barh(corr_feats, corr_vals, color=colors_bar, edgecolor="white")
ax4.set_title("Feature Importance", fontweight="bold")
ax4.set_xlabel("|Correlation|")
ax4.tick_params(axis="y", labelsize=8)

# Panel 5 — Performance metrics
ax5 = fig.add_subplot(gs[1, 1])
mnames = list(metrics.keys())
mvals  = [v * 100 for v in metrics.values()]
bars5  = ax5.bar(mnames, mvals, color=colors[:4], edgecolor="white", width=0.5)
for bar, v in zip(bars5, mvals):
    ax5.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
             f"{v:.1f}%", ha="center", fontsize=9, fontweight="bold")
ax5.set_ylim(0, 115)
ax5.set_title("Model Performance Metrics", fontweight="bold")
ax5.set_ylabel("Score (%)")

# Panel 6 — Age distribution
ax6 = fig.add_subplot(gs[1, 2])
for label, color, name in zip([0, 1], PALETTE_MAIN, ["No Stroke", "Stroke"]):
    ax6.hist(df[df["stroke"] == label]["age"], bins=25, alpha=0.7,
             color=color, label=name, edgecolor="white", linewidth=0.3)
ax6.set_title("Age Distribution", fontweight="bold")
ax6.set_xlabel("Age")
ax6.set_ylabel("Count")
ax6.legend(fontsize=9)

plt.savefig("viz10_dashboard.png", dpi=OUTPUT_DPI, bbox_inches="tight")
plt.show()
print("[VIZ] Saved → viz10_dashboard.png")

# =============================================================================
# STEP 8 — SAMPLE PREDICTION (Single Patient)
# =============================================================================

print("\n" + "=" * 60)
print("  SAMPLE PREDICTION")
print("=" * 60)

# Example: 67-year-old Male, hypertension, no heart disease,
#          married, Private job, Urban, glucose=228.69, BMI=36.6, formerly smoked
sample_raw = {
    "gender":            1,   # Male = 1 (LabelEncoder order may differ)
    "age":               67,
    "hypertension":      0,
    "heart_disease":     1,
    "ever_married":      1,   # Yes
    "work_type":         2,   # Private
    "Residence_type":    1,   # Urban
    "avg_glucose_level": 228.69,
    "bmi":               36.6,
    "smoking_status":    1,   # formerly smoked
}
sample_df     = pd.DataFrame([sample_raw])
sample_scaled = scaler.transform(sample_df)
prediction    = model.predict(sample_scaled)[0]
proba_approx  = model.predict_proba(sample_scaled)[0]

print(f"  Input features : {sample_raw}")
print(f"  Prediction     : {'⚠ STROKE RISK' if prediction == 1 else '✓ NO STROKE RISK'}")
print(f"  Probabilities  : No Stroke={proba_approx[0]*100:.1f}%  |  Stroke={proba_approx[1]*100:.1f}%")

# =============================================================================
# SUMMARY
# =============================================================================

print("\n" + "=" * 60)
print("  CAPSTONE SUMMARY")
print("=" * 60)
print(f"  Dataset        : 5,110 patients, 11 features")
print(f"  Algorithm      : K-Nearest Neighbors (KNN)")
print(f"  Best k         : {best_k}")
print(f"  Accuracy       : {acc*100:.2f}%")
print(f"  Precision      : {prec*100:.2f}%")
print(f"  Recall         : {rec*100:.2f}%")
print(f"  F1-Score       : {f1:.4f}")
print(f"  True Positives : {cm[1,1]}  (stroke patients correctly identified)")
print(f"  False Negatives: {cm[1,0]}  (missed stroke cases)")
print()
print("  Visualizations saved:")
for i, name in enumerate([
    "Class Distribution",
    "Feature Correlation Heatmap",
    "Age Distribution by Stroke",
    "Glucose Box Plot",
    "k vs. Performance Metrics",
    "Confusion Matrix",
    "Final Metrics Bar Chart",
    "Feature Importance",
    "Age vs. BMI Scatter",
    "Comprehensive Dashboard",
], start=1):
    print(f"    viz{i:02d} — {name}")
print("=" * 60)