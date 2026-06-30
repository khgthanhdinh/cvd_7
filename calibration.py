import pandas as pd
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import brier_score_loss
import matplotlib.pyplot as plt

def calibrate_and_evaluate(X_train, y_train, X_test, y_test, tuned_models, dataset_name):
    calibrated_models = {}
    brier_results = []

    cal_method = 'isotonic' if dataset_name == 'framingham' else 'sigmoid'

    curves_before = {}
    curves_after = {}

    for model_name, tuned_pipe in tuned_models.items():
        # evaluate before calib (use test set)
        y_probs_uncal = tuned_pipe.predict_proba(X_test)[:, 1]
        brier_uncal = brier_score_loss(y_test, y_probs_uncal)
        prob_true_uncal, prob_pred_uncal = calibration_curve(y_test, y_probs_uncal, n_bins=10)
        curves_before[model_name] = (prob_true_uncal, prob_pred_uncal)

        brier_results.append({
            'model': model_name,
            'state': 'Before Calibration',
            'brier_score': brier_uncal
        })

        # evaluate after calib (use test set)
        if model_name == 'Logistic':
            # LR is naturally calibrated. Use native probabilities.
            cal_model = tuned_pipe
            y_probs_cal = y_probs_uncal
            brier_cal = brier_uncal
            prob_true_cal, prob_pred_cal = prob_true_uncal, prob_pred_uncal
        else:
            print(f"Calibrating {model_name} using method: '{cal_method}'...")
            cal_model = CalibratedClassifierCV(estimator=tuned_pipe, method=cal_method, cv=5)
            cal_model.fit(X_train, y_train)

            y_probs_cal = cal_model.predict_proba(X_test)[:, 1]
            brier_cal = brier_score_loss(y_test, y_probs_cal)
            prob_true_cal, prob_pred_cal = calibration_curve(y_test, y_probs_cal, n_bins=10)

        curves_after[model_name] = (prob_true_cal, prob_pred_cal)
        calibrated_models[model_name] = cal_model

        brier_results.append({
            'model': model_name,
            'state': 'After Calibration',
            'brier_score': brier_cal
        })

        # (A) before & after calib of 1 model
        fig, ax = plt.subplots(figsize=(7, 6))
        ax.plot([0, 1], [0, 1], linestyle='--', color='black', label='Perfectly Calibrated')
        ax.plot(prob_pred_uncal, prob_true_uncal, marker='o', linestyle='-', label=f'Before (Brier: {brier_uncal:.4f})')
        if model_name != 'Logistic':
            ax.plot(prob_pred_cal, prob_true_cal, marker='s', linestyle='-', label=f'After (Brier: {brier_cal:.4f})')

        ax.set_xlabel('Mean Predicted Probability')
        ax.set_ylabel('Fraction of Positives')
        ax.set_title(f'{model_name} Calibration Curve - {dataset_name.capitalize()}')
        ax.legend()
        plt.tight_layout()
        plt.savefig(f'./{dataset_name}/cal_curve_indiv_{dataset_name}_{model_name}.png')
        plt.close()

    # (B) all models before calib
    fig_before, ax_before = plt.subplots(figsize=(8, 6))
    ax_before.plot([0, 1], [0, 1], linestyle='--', color='black', label='Perfectly Calibrated')
    for m_name, (pt, pp) in curves_before.items():
        ax_before.plot(pp, pt, marker='o', label=m_name)
    ax_before.set_xlabel('Mean Predicted Probability')
    ax_before.set_ylabel('Fraction of Positives')
    ax_before.set_title(f'All Models BEFORE Calibration - {dataset_name.capitalize()}')
    ax_before.legend()
    plt.tight_layout()
    plt.savefig(f'./{dataset_name}/cal_curve_all_before_{dataset_name}.png')
    plt.close()

    # (C) all models after calib
    fig_after, ax_after = plt.subplots(figsize=(8, 6))
    ax_after.plot([0, 1], [0, 1], linestyle='--', color='black', label='Perfectly Calibrated')
    for m_name, (pt, pp) in curves_after.items():
        ax_after.plot(pp, pt, marker='s', label=m_name)
    ax_after.set_xlabel('Mean Predicted Probability')
    ax_after.set_ylabel('Fraction of Positives')
    ax_after.set_title(f'All Models AFTER Calibration - {dataset_name.capitalize()}')
    ax_after.legend()
    plt.tight_layout()
    plt.savefig(f'./{dataset_name}/cal_curve_all_after_{dataset_name}.png')
    plt.close()

    df_brier = pd.DataFrame(brier_results)
    return calibrated_models, df_brier