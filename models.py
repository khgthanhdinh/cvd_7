from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
import xgboost as xgb
import catboost as cb
from sklearn.svm import SVC
from config import RANDOM_STATE

def get_models(dataset_name):
    if dataset_name == 'framingham':
        return {
            'Logistic': (
                LogisticRegression(random_state=RANDOM_STATE, class_weight='balanced', max_iter=1000),
                {'model__C': [0.001, 0.01, 0.1, 1.0, 10.0]},
                'linear'
            ),
            'DecisionTree': (
                DecisionTreeClassifier(random_state=RANDOM_STATE, class_weight='balanced'),
                {'model__max_depth': [5, 10, None], 'model__min_samples_split': [5, 10], 'model__min_samples_leaf': [2, 4]},
                'tree'
            ),
            'SVM': (
                SVC(probability=True, random_state=RANDOM_STATE, class_weight='balanced'),
                {'model__C': [0.1, 1, 10], 'model__gamma': ['scale', 0.01], 'model__kernel': ['rbf']},
                'distance'
            ),
            'RandomForest': (
                RandomForestClassifier(random_state=RANDOM_STATE, class_weight='balanced', n_jobs=-1),
                {'model__n_estimators': [100, 200], 'model__max_depth': [6, 12, None], 'model__min_samples_leaf': [2, 4], 'model__max_features': ['sqrt']},
                'tree'
            ),
            'XGBoost': (
                xgb.XGBClassifier(eval_metric='logloss', random_state=RANDOM_STATE, scale_pos_weight=2877/515),
                {'model__learning_rate': [0.05], 'model__max_depth': [3, 6], 'model__n_estimators': [100, 300], 'model__subsample': [0.8]},
                'tree'
            ),
            'CatBoost': (
                cb.CatBoostClassifier(verbose=0, random_state=RANDOM_STATE, auto_class_weights='Balanced'),
                {'model__learning_rate': [0.05], 'model__depth': [4, 7], 'model__iterations': [100, 300], 'model__l2_leaf_reg': [3, 5]},
                'tree'
            )
        }
    else:  # cleveland
        return {
            'Logistic': (
                LogisticRegression(random_state=RANDOM_STATE, max_iter=1000),
                {'model__C': [0.1, 1.0, 10.0]},
                'linear'
            ),
            'DecisionTree': (
                DecisionTreeClassifier(random_state=RANDOM_STATE),
                {'model__max_depth': [3, 5], 'model__min_samples_leaf': [2, 4], 'model__min_samples_split': [4, 8]},
                'tree'
            ),
            'SVM': (
                SVC(probability=True, random_state=RANDOM_STATE),
                {'model__C': [0.1, 1.0, 10.0], 'model__gamma': ['scale', 0.01], 'model__kernel': ['rbf', 'linear']},
                'distance'
            ),
            'RandomForest': (
                RandomForestClassifier(random_state=RANDOM_STATE),
                {'model__n_estimators': [50, 100], 'model__max_depth': [4, 6, None], 'model__min_samples_leaf': [2, 4], 'model__max_features': ['sqrt']},
                'tree'
            ),
            'XGBoost': (
                xgb.XGBClassifier(eval_metric='logloss', random_state=RANDOM_STATE),
                {'model__learning_rate': [0.05], 'model__max_depth': [2, 4], 'model__subsample': [0.7], 'model__reg_lambda': [1.0, 10.0], 'model__n_estimators': [50, 100]},
                'tree'
            ),
            'CatBoost': (
                cb.CatBoostClassifier(verbose=0, random_state=RANDOM_STATE),
                {'model__learning_rate': [0.05], 'model__depth': [3, 5], 'model__iterations': [50, 100], 'model__l2_leaf_reg': [3, 7]},
                'tree'
            )
        }