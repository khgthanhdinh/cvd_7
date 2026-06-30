import warnings
warnings.filterwarnings("ignore")
import logging
logging.getLogger("PyALE").setLevel(logging.WARNING)

import sys
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap
from PyALE import ale
from scipy.stats import kendalltau

sys.path.insert(0, ".")
from utils import ThresholdClassifier

DATA_PATH  = "./cleveland/cleveland_train_test_data.joblib" # Update to your target dataset
MODEL_PATH = "./cleveland/cleveland_CatBoost_model.joblib" # Update to chosen best model
DPI        = 150

N_BOOTSTRAP   = 20
BOOT_FRAC     = 0.8
ALE_GRID_SIZE = 20
RANDOM_SEED   = 42
TOP_K_STABILITY = 5 # Calculate overlap for Top-5 features

# load data and model
print("Loading data and model...")
X_train, X_test, y_train, y_test = joblib.load(DATA_PATH)
wrapper = joblib.load(MODEL_PATH)

calibrated_cv = wrapper.model
pipeline = calibrated_cv.estimator

preprocessor = pipeline.named_steps["preprocessor"]
base_model   = pipeline.named_steps["model"]

X_train_t_np = preprocessor.transform(X_train)
X_test_t_np  = preprocessor.transform(X_test)
raw_feat  = list(preprocessor.get_feature_names_out())
n_feat    = len(raw_feat)

# X_train_t_np = X_train_t.to_numpy()
# X_test_t_np  = X_test_t.to_numpy()

cat_feats = [col for col in X_train.columns if X_train[col].dtype == 'object' or X_train[col].dtype.name == 'category']
num_feats = [col for col in raw_feat if col not in cat_feats]

# SHAP values (test set)
print("Computing SHAP values...")
# Using the base model for TreeExplainer. If it's LogisticRegression/SVM, use LinearExplainer/KernelExplainer
if "Tree" in str(type(base_model)) or "Forest" in str(type(base_model)) or "Boost" in str(type(base_model)):
    explainer = shap.TreeExplainer(base_model)
else:
    # For non-tree models, provide the predict_proba function
    explainer = shap.Explainer(base_model.predict_proba, X_train_t_np)

shap_values = explainer.shap_values(X_test_t_np)

if isinstance(shap_values, list):
    sv = shap_values[1]
else:
    sv = shap_values[:, :, 1] if len(shap_values.shape) == 3 else shap_values

print(f"  SHAP shape: {sv.shape}")

# 1 - SHAP dependence vs. ALE

print("Computing ALE effects...")

class PredWrapper:
    def __init__(self, model):
        self.model = model

    def predict(self, X_df):
        return self.model.predict_proba(X_df)[:, 1]

model_w = PredWrapper(base_model) # Initialize with just the base_model
X_train_df = pd.DataFrame(X_train_t_np, columns=raw_feat)

ale_effects = {}
for feat in num_feats:
    try:
        ale_effects[feat] = ale(X=X_train_df, model=model_w, feature=[feat], grid_size=ALE_GRID_SIZE, include_CI=True)
        print(f"  ALE computed: {feat}")
    except Exception as e:
        print(f"  Skipping ALE for {feat}: {e}")

ncols = 4
nrows = int(np.ceil(len(num_feats) / ncols))
fig1, axes = plt.subplots(nrows, ncols, figsize=(16, nrows * 3.8))
axes = axes.flatten()

for i_f, feat in enumerate(num_feats):
    if feat not in ale_effects: continue

    ax  = axes[i_f]
    ax2 = ax.twinx()

    fi = raw_feat.index(feat)
    feat_vals = X_test_t_np[:, fi]
    shap_vals = sv[:, fi]

    ax.scatter(feat_vals, shap_vals, alpha=0.3, s=8, color="steelblue")
    ax.axhline(0, color="grey", lw=0.8, ls="--")
    ax.set_xlabel(feat, fontsize=10)
    ax.set_ylabel("SHAP value", fontsize=9, color="steelblue")

    ale_df = ale_effects[feat]
    ale_x  = ale_df.index.values.astype(float)
    ale_y  = ale_df["eff"].values

    ax2.plot(ale_x, ale_y, color="tomato", lw=2.0)
    if "lowerCI_95%" in ale_df.columns:
        ax2.fill_between(ale_x, ale_df["lowerCI_95%"], ale_df["upperCI_95%"], color="tomato", alpha=0.15)

    ax2.set_ylabel("ALE effect", fontsize=9, color="tomato")
    ax.set_title(feat, fontsize=10, fontweight="bold", pad=4)

for j in range(len(num_feats), len(axes)):
    axes[j].set_visible(False)

fig1.tight_layout()
fig1.savefig("./cleveland/validation_shap_ale_cleve.png", dpi=DPI, bbox_inches="tight")
plt.close()
print("Saved → ./cleveland/validation_shap_ale_cleve.png")

# 2 - bootstrap SHAP
print(f"Running {N_BOOTSTRAP} bootstrap runs for stability metrics...")
rng = np.random.default_rng(RANDOM_SEED)
boot_rankings = np.zeros((N_BOOTSTRAP, n_feat))
boot_mean_abs = np.zeros((N_BOOTSTRAP, n_feat))

for b in range(N_BOOTSTRAP):
    idx = rng.choice(len(X_test_t_np), size=int(len(X_test_t_np) * BOOT_FRAC), replace=False)

    sv_b = explainer.shap_values(X_test_t_np[idx])
    if isinstance(sv_b, list): sv_b = sv_b[1]
    elif len(sv_b.shape) == 3: sv_b = sv_b[:, :, 1]

    means = np.abs(sv_b).mean(axis=0)
    boot_mean_abs[b] = means
    boot_rankings[b] = n_feat - means.argsort().argsort() # 1 is highest magnitude

# Kendall's Tau & Top-K Overlap
tau_scores = []
overlap_scores = []

for i in range(N_BOOTSTRAP):
    for j in range(i + 1, N_BOOTSTRAP):
        # Kendall's Tau
        tau, _ = kendalltau(boot_rankings[i], boot_rankings[j])
        tau_scores.append(tau)

        # Top-K Overlap
        top_i = set(np.where(boot_rankings[i] <= TOP_K_STABILITY)[0])
        top_j = set(np.where(boot_rankings[j] <= TOP_K_STABILITY)[0])
        overlap = len(top_i.intersection(top_j)) / TOP_K_STABILITY
        overlap_scores.append(overlap)

mean_tau = np.mean(tau_scores)
mean_overlap = np.mean(overlap_scores)
print(f"  Mean Kendall's Tau: {mean_tau:.3f}")
print(f"  Mean Top-{TOP_K_STABILITY} Overlap: {mean_overlap*100:.1f}%")

mean_ranks = boot_rankings.mean(axis=0)
std_ranks  = boot_rankings.std(axis=0)
cv_ranks   = (std_ranks / mean_ranks) * 100

order = np.argsort(mean_ranks)
feat_ordered  = np.array(raw_feat)[order]
means_ordered = boot_mean_abs.mean(axis=0)[order]
stds_ordered  = boot_mean_abs.std(axis=0)[order]
cv_ordered    = cv_ranks[order]

fig3, axes3 = plt.subplots(1, 2, figsize=(14, 6))

# Left: mean |SHAP| bar + error bars
ax_l = axes3[0]
y_pos = np.arange(n_feat)
ax_l.barh(y_pos, means_ordered, xerr=stds_ordered, color="steelblue", ecolor="black", capsize=3, alpha=0.8)
ax_l.set_yticks(y_pos)
ax_l.set_yticklabels(feat_ordered, fontsize=9)
ax_l.set_xlabel("Mean |SHAP value|", fontsize=10)
ax_l.set_title(f"Bootstrap Feature Importance\n(Kendall's Tau = {mean_tau:.2f} | Top-{TOP_K_STABILITY} Overlap = {mean_overlap*100:.0f}%)")

# Right: rank CV% bar chart
ax_r = axes3[1]
bar_colors = ["tomato" if c > 10 else "steelblue" for c in cv_ordered]
ax_r.barh(y_pos, cv_ordered, color=bar_colors, alpha=0.8)
ax_r.set_yticks(y_pos)
ax_r.set_yticklabels(feat_ordered, fontsize=9)
ax_r.set_xlabel("Rank Coefficient of Variation (%)", fontsize=10)
ax_r.axvline(10, color="tomato", lw=1.2, ls="--", alpha=0.7)
ax_r.set_title("Feature Ranking Stability (CV)")

fig3.tight_layout()
fig3.savefig("./cleveland/validation_bootstrap_stability_cleve.png", dpi=DPI, bbox_inches="tight")
plt.close()
print("Saved → ./cleveland/validation_bootstrap_stability_cleve.png")