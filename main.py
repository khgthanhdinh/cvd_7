import pandas as pd
import matplotlib.pyplot as plt
from loader import load_and_split_data
from models import get_models
from training import run_nested_cv, hypertune_model
from calibration import calibrate_and_evaluate
from threshold_tuning import tune_thresholds
import utils
import loader
import preprocessing
import training
import calibration
import threshold_tuning
import models
import importlib
importlib.reload(utils)
importlib.reload(loader)
importlib.reload(preprocessing)
importlib.reload(training)
importlib.reload(calibration)
importlib.reload(threshold_tuning)
importlib.reload(models)
from utils import calculate_metrics, plot_dca
import joblib

def main(dataset_name, csv_path):
    print(f"--- Processing {dataset_name.upper()} Dataset ---")

    # load + split
    X_train, X_test, y_train, y_test, schema = load_and_split_data(csv_path, dataset_name)
    joblib.dump((X_train, X_test, y_train, y_test), f"./{dataset_name}/{dataset_name}_train_test_data.joblib")

    models_def = get_models(dataset_name)

    # nested cv (train set)
    df_nested_summary, detailed_tables = run_nested_cv(X_train, y_train, schema, models_def, dataset_name)
    df_nested_summary.to_csv(f"./{dataset_name}/{dataset_name}_nested_cv_results.csv", index=False)

    for model_name, df_detail in detailed_tables.items():
        df_detail.to_csv(f"./{dataset_name}/{dataset_name}_{model_name}_detailed_folds.csv", index=False)

    print("Nested CV Complete. Top models summary:\n", df_nested_summary.head(3))

    # top 3 models after nested cv
    top_3_names = df_nested_summary['model'].head(3).tolist()
    if 'Logistic' not in top_3_names:
        top_3_names.append('Logistic')

    # hypertune for top 3 models (train set)
    tuned_models = {}
    for name in top_3_names:
        tuned_models[name] = hypertune_model(X_train, y_train, schema, name, models_def[name], dataset_name)

    # calibration (train set) / analysis (test set)
    calibrated_models, df_brier = calibrate_and_evaluate(X_train, y_train, X_test, y_test, tuned_models, dataset_name)
    # df_brier.to_csv(f"{dataset_name}_calibration_brier.csv", index=False)

    df_brier.to_csv(f"./{dataset_name}/{dataset_name}_calibration_brier_comparison.csv", index=False)
    print("\nCalibration Brier Scores (Test Set):\n", df_brier)

    # evaluate before threshold tuning (0.5)
    before_thresh_results = []
    for name, model in calibrated_models.items():
        y_probs = model.predict_proba(X_test)[:, 1]
        y_pred = (y_probs >= 0.5).astype(int) # Default threshold
        metrics = calculate_metrics(y_test, y_pred, y_probs)
        metrics['Model'] = name
        before_thresh_results.append(metrics)

    df_before_thresh = pd.DataFrame(before_thresh_results)
    df_before_thresh.to_csv(f"./{dataset_name}/{dataset_name}_before_threshold_tuning_eval.csv", index=False)
    print("\nEvaluation BEFORE Threshold Tuning:\n", df_before_thresh[['Model', 'TP', 'FP', 'TN', 'FN', 'f1', 'roc_auc', 'pr_auc']])

    # threshold tuning (oof pro3 from train set)
    final_models = tune_thresholds(X_train, y_train, calibrated_models, dataset_name)

    for name, model_wrapper in final_models.items():
        joblib.dump(model_wrapper, f"./{dataset_name}/{dataset_name}_{name}_model.joblib")

    # final evaluation (test set)
    final_results = []
    for name, model in final_models.items():
        y_probs = model.predict_proba(X_test)[:, 1]
        y_pred = model.predict(X_test)
        metrics = calculate_metrics(y_test, y_pred, y_probs)
        metrics['Model'] = name
        final_results.append(metrics)

    df_final = pd.DataFrame(final_results)
    df_final.to_csv(f"./{dataset_name}/{dataset_name}_final_test_evaluation.csv", index=False)
    print("\nFinal Test Evaluation AFTER Threshold Tuning:\n", df_final[['Model', 'TP', 'FP', 'TN', 'FN', 'f1', 'roc_auc', 'pr_auc']])

    # optional: DCA Plot
    plot_dca(y_test, final_models, dataset_name, X_test)

if __name__ == "__main__":
    # Replace with your actual paths
    main('cleveland', './data/cleveland.csv')
    main('framingham', './data/framingham.csv')