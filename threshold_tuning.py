import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import precision_recall_curve, roc_curve
from utils import ThresholdClassifier

def tune_thresholds(X_train, y_train, calibrated_models, dataset_name):
    final_models = {}

    for model_name, cal_model in calibrated_models.items():
        print(f"Tuning threshold for {model_name}...")

        # Out-of-fold pro3
        oof_probs = cross_val_predict(cal_model, X_train, y_train, cv=5, method='predict_proba')[:, 1]

        fig, ax = plt.subplots(figsize=(7, 6))

        if dataset_name == 'cleveland':
            # Youden's J (ROC curve) for diag/balanced
            fpr, tpr, thresholds = roc_curve(y_train, oof_probs)
            j_scores = tpr - fpr
            best_idx = np.argmax(j_scores)

            best_threshold = thresholds[best_idx]

            # Plot ROC
            ax.plot([0, 1], [0, 1], linestyle='--', color='black', label='Random Classifier')
            ax.plot(fpr, tpr, linewidth=2, label=f'{model_name} ROC')
            ax.scatter(fpr[best_idx], tpr[best_idx], s=50, color='red', zorder=5,
                       label=f"J-stat Optimal: {best_threshold:.3f}")

            ax.set_xlabel('False Positive Rate')
            ax.set_ylabel('True Positive Rate')
            ax.set_title(f'ROC Curve & Optimal Threshold - {model_name}')

        else:
            # F2-score (PR curve) for prog/imbalance
            precisions, recalls, thresholds = precision_recall_curve(y_train, oof_probs)

            f2_scores = (5 * precisions * recalls) / ((4 * precisions) + recalls + 1e-10)
            best_idx = np.argmax(f2_scores)

            thresh_idx = min(best_idx, len(thresholds) - 1)
            best_threshold = thresholds[thresh_idx]

            # Plot PR
            all1_precision = len(y_train[y_train == 1]) / len(y_train)
            ax.plot([0, 1], [all1_precision, all1_precision], linestyle='--', color='black', label='Always Positive')
            ax.plot(recalls, precisions, linewidth=2, label=f'{model_name} PR Curve')
            ax.scatter(recalls[best_idx], precisions[best_idx], s=50, color='red', zorder=5,
                       label=f"F2-Optimal: {best_threshold:.3f}")

            ax.set_xlabel('Recall')
            ax.set_ylabel('Precision')
            ax.set_title(f'PR Curve & Optimal Threshold - {model_name}')

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.05)
        ax.legend(loc='lower left' if dataset_name == 'framingham' else 'lower right')
        plt.tight_layout()
        plt.savefig(f'tuning_curve_{dataset_name}_{model_name}.png')
        plt.close()

        print(f"Optimal Threshold for {model_name}: {best_threshold:.4f}")
        final_models[model_name] = ThresholdClassifier(cal_model, best_threshold)

    return final_models