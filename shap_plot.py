import warnings
warnings.filterwarnings("ignore")

import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap

def get_feature_names(preprocess, X_original):
    
    try:
        raw = preprocess.get_feature_names_out()
        return [n.split("__")[-1] for n in raw]
    except Exception:
        return list(X_original.columns)

def extract_shap_values(raw, class_idx=1, model_output_prob=True):
    
    if model_output_prob:
        if isinstance(raw, list):
            return raw[class_idx]
        if raw.ndim == 3:
            return raw[:, :, class_idx]
        return raw
    else:
        if isinstance(raw, list) and len(raw) == 2:
            return raw[class_idx]
        elif raw.ndim == 3:
            return raw[:, :, class_idx]
        return raw

def extract_expected_value(ev, class_idx=1, model_output_prob=True):

    if model_output_prob:
        if isinstance(ev, (list, np.ndarray)):
            return float(ev[class_idx])
        return float(ev)
    else:
        if isinstance(ev, (list, np.ndarray)):
            return float(ev[class_idx])
        return float(ev)


def generate_shap_plots(dataset_name, data_path, model_path, local_idx=0):
    print(f"\n[{dataset_name.upper()}] Loading data and Calibrated CatBoost model...")

    # load data and model
    X_train_raw, X_test_raw, y_train, y_test = joblib.load(data_path)
    wrapper = joblib.load(model_path)

    # extract pipeline
    # wrapper -> ThresholdClassifier
    # wrapper.model -> CalibratedClassifierCV
    # wrapper.model.estimator -> sklearn.pipeline.Pipeline
    pipeline = wrapper.model.estimator
    preprocessor = pipeline.named_steps["preprocessor"]
    catboost_model = pipeline.named_steps["model"]

    # preprocess X_train and X_test
    X_train_t = preprocessor.transform(X_train_raw)
    X_test_t = preprocessor.transform(X_test_raw)

    if hasattr(X_train_t, 'values'):
        X_train_t = X_train_t.values
    if hasattr(X_test_t, 'values'):
        X_test_t = X_test_t.values

    feat_names = get_feature_names(preprocessor, X_test_raw)

    if dataset_name == 'cleveland':
        feature_perturbation = 'interventional'
        model_output = 'probability'
        use_background = True
        k_background = 50
        label = "CHD"
    elif dataset_name == 'framingham':
        feature_perturbation = 'interventional'
        use_background = True
        k_background = 50
        label = "10-year CHD"
    else:
        raise ValueError("Unknown dataset_name")

    # SHAP explainer
    print(f"[{dataset_name.upper()}] Initializing SHAP TreeExplainer...")

    explainer = None
    if use_background:
        # interventional: pass a summarised background dataset
        print(f"Building background via k-means (k={k_background}) ...")
        # Use processed training data for background
        background = shap.kmeans(X_train_t, k_background)
        explainer = shap.TreeExplainer(
            catboost_model,
            data=background.data,
            model_output=model_output,
            feature_perturbation=feature_perturbation,
        )
    else:
        # tree_path_dependent: no external background needed
        print("Building TreeExplainer (tree_path_dependent, no background needed) ...")
        explainer = shap.TreeExplainer(
            catboost_model,
            model_output=model_output,
            feature_perturbation=feature_perturbation,
        )

    # SHAP values
    print(f"[{dataset_name.upper()}] Calculating SHAP values (this may take a minute)...")
    raw_shap_values = explainer.shap_values(X_test_t)

    model_output_prob_flag = (model_output == 'probability')
    sv = extract_shap_values(raw_shap_values, class_idx=1, model_output_prob=model_output_prob_flag)
    ev = extract_expected_value(explainer.expected_value, class_idx=1, model_output_prob=model_output_prob_flag)

   
    proba = pipeline.predict_proba(X_test_raw)[:, 1]
    pred_p = proba[local_idx]
    true_l = int(y_test.iloc[local_idx])
    true_str = label if true_l == 1 else f"No {label}"
    thresh = wrapper.threshold

    output_space_str = "probability scale" if model_output_prob_flag else "raw (log-odds) scale"

    print(f"[{dataset_name.upper()}] Base value (E[P({label})]): {ev:.4f}")
    print(f"[{dataset_name.upper()}] Predicted P({label}) for sample #{local_idx}: {pred_p:.4f}")

    # --------------------------------------------------
    # (1) global - stacked bar plot
    print(f"[{dataset_name.upper()}] Generating Stacked Bar Plot...")

    sv0 = -sv

    plt.figure(figsize=(10, 6))
    shap.summary_plot(
        [sv0, sv], X_test_t,
        feature_names=feat_names,
        plot_type="bar",
        class_names=[f"No {label} (0)", f"{label} (1)"],
        show=False,
    )
    plt.title(f"Global Feature Importance ({output_space_str}) - {dataset_name.capitalize()}")
    plt.xlabel(f"Mean |SHAP value| (mean impact on {output_space_str})")
    plt.tight_layout()
    plt.savefig(f"./{dataset_name}/shap_{dataset_name}_global_bar.png", dpi=300, bbox_inches='tight')
    plt.close()

    # --------------------------------------------------
    # (2) global - beeswarm plot
    print(f"[{dataset_name.upper()}] Generating Beeswarm Plot...")
    plt.figure(figsize=(10, 6))
    shap.summary_plot(sv, X_test_t, feature_names=feat_names, plot_type="dot", show=False)
    plt.title(f"SHAP Summary (Impact on Positive Class {output_space_str}) - {dataset_name.capitalize()}")
    plt.xlabel(f"SHAP value ({output_space_str})")
    plt.tight_layout()
    plt.savefig(f"./{dataset_name}/shap_{dataset_name}_global_beeswarm.png", dpi=300, bbox_inches='tight')
    plt.close()

    # ------------------------------------------------
    # (3) local - waterfall plot
    print(f"[{dataset_name.upper()}] Generating Local Waterfall Plot for patient index {local_idx}...")
    explanation = shap.Explanation(
        values=sv[local_idx],
        base_values=ev,
        data=X_test_t[local_idx],
        feature_names=feat_names,
    )
    plt.figure(figsize=(8, 6))
    shap.plots.waterfall(explanation, show=False)
    plt.title(f"Local Waterfall Plot (Patient {local_idx}) - {dataset_name.capitalize()} ({output_space_str})")
    plt.tight_layout()
    plt.savefig(f"./{dataset_name}/shap_{dataset_name}_local_waterfall_idx{local_idx}.png", dpi=300, bbox_inches='tight')
    plt.close()

    # ------------------------------------------------
    # (4) local - force plot
    print(f"[{dataset_name.upper()}] Generating Local Force Plot for patient index {local_idx}...")

    local_info = (f"Sample #{local_idx}  ·  True label: {true_str}  ·  "\
                  f"Predicted P({label}) = {pred_p:.3f}  ·  Decision threshold = {thresh:.3f}")

    shap.plots.force(
        ev,
        sv[local_idx],
        features=X_test_t[local_idx],
        feature_names=feat_names,
        matplotlib=True,
        show=False,
    )
    fig = plt.gcf()
    fig.set_size_inches(16, 3)
    plt.suptitle(f"Local Force Plot (Patient {local_idx}) - {dataset_name.capitalize()} ({output_space_str})", y=1.5, fontsize=14, fontweight="bold")
    plt.title(local_info, pad=30, fontsize=11)
    plt.tight_layout()
    plt.savefig(f"./{dataset_name}/shap_{dataset_name}_local_force_idx{local_idx}.png", dpi=300, bbox_inches='tight')
    plt.close()

    print(f"[{dataset_name.upper()}] All SHAP plots saved successfully.\n")


if __name__ == "__main__":
    try:
        # Cleveland
        generate_shap_plots(
            dataset_name='cleveland',
            data_path='./cleveland/cleveland_train_test_data.joblib',
            model_path='./cleveland/cleveland_CatBoost_model.joblib',
            local_idx=0  # Change this integer to inspect different patients
        )

        # Framingham
        generate_shap_plots(
            dataset_name='framingham',
            data_path='./framingham/framingham_train_test_data.joblib',
            model_path='./framingham/framingham_CatBoost_model.joblib',
            local_idx=0
        )

    except FileNotFoundError as e:
        print(f"Error: Could not find the necessary .joblib files. Please ensure the paths are correct. Details: {e}")