import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from sklearn.pipeline import Pipeline as SkPipeline
from preprocessing import preprocessor
from utils import calculate_metrics
from config import RANDOM_STATE

def run_nested_cv(X, y, schema, models_def, dataset_name):
    main_metric = 'roc_auc' if dataset_name == 'cleveland' else 'average_precision'

    outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    inner_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    summary_results = []
    detailed_tables = {} # Will hold the 7 tables (1 per model)

    for model_name, (model_obj, param_grid, model_type) in models_def.items():
        print(f"Running Nested CV for {model_name}...")
        prep = preprocessor(schema, model_type)
        pipe = SkPipeline([('preprocessor', prep), ('model', model_obj)])

        search = RandomizedSearchCV(
            pipe, param_grid, cv=inner_cv, n_iter=20,
            scoring=main_metric, n_jobs=-1, random_state=RANDOM_STATE
        )

        fold_metrics = []
        for fold, (train_idx, val_idx) in enumerate(outer_cv.split(X, y)):
            X_train_f, X_val_f = X.iloc[train_idx], X.iloc[val_idx]
            y_train_f, y_val_f = y.iloc[train_idx], y.iloc[val_idx]

            search.fit(X_train_f, y_train_f)
            best_pipe = search.best_estimator_

            y_probs = best_pipe.predict_proba(X_val_f)[:, 1]
            y_pred = best_pipe.predict(X_val_f)

            metrics = calculate_metrics(y_val_f, y_pred, y_probs)
            metrics['Fold'] = fold + 1
            metrics['Best_Params'] = str(search.best_params_)
            fold_metrics.append(metrics)

        # detailed table for 1 model
        df_detailed = pd.DataFrame(fold_metrics)
        detailed_tables[model_name] = df_detailed

        # mean +/- std final summary
        metric_cols = ['sensitivity', 'specificity', 'roc_auc', 'pr_auc', 'f1', 'precision', 'balanced_accuracy']
        summary_row = {'model': model_name}
        for col in metric_cols:
            mean_val = df_detailed[col].mean()
            std_val = df_detailed[col].std()
            summary_row[f"{col} Mean±std"] = f"{mean_val:.3f}±{std_val:.3f}"
            summary_row[f"{col}_mean_raw"] = mean_val

        summary_results.append(summary_row)

    df_summary = pd.DataFrame(summary_results)
    sort_col = 'roc_auc_mean_raw' if dataset_name == 'cleveland' else 'pr_auc_mean_raw'
    df_summary = df_summary.sort_values(by=sort_col, ascending=False).drop(columns=[col for col in df_summary.columns if 'raw' in col])

    return df_summary, detailed_tables

def hypertune_model(X, y, schema, model_name, model_def, dataset_name):
    model_obj, param_grid, model_type = model_def
    prep = preprocessor(schema, model_type)
    pipe = SkPipeline([('preprocessor', prep), ('model', model_obj)])

    main_metric = 'roc_auc' if dataset_name == 'cleveland' else 'average_precision'
    search = RandomizedSearchCV(
        pipe, param_grid, cv=5, n_iter=20,
        scoring=main_metric, n_jobs=-1, random_state=RANDOM_STATE
    )
    search.fit(X, y)
    return search.best_estimator_