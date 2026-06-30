import pandas as pd
import numpy as np
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, average_precision_score, 
                             balanced_accuracy_score, confusion_matrix)
import matplotlib.pyplot as plt

class ThresholdClassifier:
    """Wrapper to force a specific probability threshold on a trained model."""
    def __init__(self, model, threshold):
        self.model = model
        self.threshold = threshold 
        if hasattr(model, 'classes_'):
            self.classes_ = model.classes_ 

    def predict(self, X):
        return (self.model.predict_proba(X)[:, 1] >= self.threshold).astype(int)

    def predict_proba(self, X):
        if isinstance(X, np.ndarray) and hasattr(self.model, 'feature_names_in_'):
            X = pd.DataFrame(X, columns=self.model.feature_names_in_)
        return self.model.predict_proba(X)
    
    def __getstate__(self):
        return self.__dict__

    def __setstate__(self, state):
        self.__dict__.update(state)
    
    def __getattr__(self, name):
        if name == 'model':
             raise AttributeError(f"'{type(self).__name__}' object has no attribute 'model'")
        if name.startswith('__') and name.endswith('__'):
             raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")
        return getattr(self.model, name)

def calculate_metrics(y_true, y_pred, y_probs):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    return {
        'TP': tp,
        'FP': fp,
        'TN': tn,
        'FN': fn,
        'sensitivity': recall_score(y_true, y_pred),
        'specificity': specificity,
        'roc_auc': roc_auc_score(y_true, y_probs),
        'pr_auc': average_precision_score(y_true, y_probs),
        'f1': f1_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred),
        'balanced_accuracy': balanced_accuracy_score(y_true, y_pred)
    }

def plot_dca(y_true, models_dict, dataset_name, X_test):
    fig, ax = plt.subplots(figsize=(8, 6))
    thresholds = np.linspace(0.01, 0.99, 100)
    n = len(y_true)
    prevalence = np.mean(y_true)
    
    max_nb = 0
    min_nb = 0

    for model_name, model in models_dict.items():
        y_probs = model.predict_proba(X_test)[:, 1]
        net_benefits = []
        
        for t in thresholds:
            y_pred = (y_probs >= t).astype(int)
            tp = np.sum((y_pred == 1) & (y_true == 1))
            fp = np.sum((y_pred == 1) & (y_true == 0))
            nb = (tp / n) - (fp / n) * (t / (1 - t))
            net_benefits.append(nb)
            
        max_nb = max(max_nb, max(net_benefits))
        min_nb = min(min_nb, min(net_benefits))
        ax.plot(thresholds, net_benefits, label=model_name)
    
    # Treat All (Calculated once)
    treat_all_nb = prevalence - (1 - prevalence) * (thresholds / (1 - thresholds))
    max_nb = max(max_nb, max(treat_all_nb))
    
    ax.plot(thresholds, np.maximum(treat_all_nb, 0), color='gray', linestyle=':', label='Treat All')
    
    # Treat None (Calculated once)
    ax.axhline(0, color='black', label='Treat None')
    
    ax.set_ylim(min(-0.05, min_nb * 1.1), max(0.25, max_nb * 1.1))
    ax.set_xlabel('Threshold Probability')
    ax.set_ylabel('Net Benefit')
    ax.legend()
    
    ax.set_title(f'Decision Curve Analysis - {dataset_name}')
    plt.tight_layout()
    plt.savefig(f'./{dataset_name}/dca_{dataset_name}.png')
