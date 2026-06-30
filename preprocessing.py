import pandas as pd
import numpy as np
from sklearn import set_config
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, SimpleImputer
from sklearn.preprocessing import RobustScaler, PowerTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline as SkPipeline
from sklearn.ensemble import RandomForestRegressor
from category_encoders.woe import WOEEncoder
from category_encoders.james_stein import JamesSteinEncoder

from config import RANDOM_STATE

set_config(transform_output="pandas")

def preprocessor(schema, model_type):
    transformers = []
    num_cols = schema.get('numerical', [])

    if num_cols:
        if model_type in ['linear', 'distance']:
            num_pipe = SkPipeline([
                ('imputer', IterativeImputer(
                    estimator=RandomForestRegressor(n_estimators=20, random_state=RANDOM_STATE),
                    max_iter=5)),
                ('yeo', PowerTransformer(method='yeo-johnson')),
                ('scaler', RobustScaler()),
            ])
        elif model_type == 'tree':
            num_pipe = SkPipeline([
                ('imputer', IterativeImputer(
                    estimator=RandomForestRegressor(n_estimators=5, random_state=RANDOM_STATE),
                    max_iter=5)),
            ])
        transformers.append(('num', num_pipe, num_cols))

    cat_cols = (schema.get('ordinal', []) +
                schema.get('binary', []) +
                schema.get('nominal', []))

    if cat_cols:
        if model_type == 'linear':
            cat_pipe = SkPipeline([
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('encoder', WOEEncoder())
            ])
        elif model_type == 'distance':
            cat_pipe = SkPipeline([
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('encoder', JamesSteinEncoder())
            ])
        elif model_type == 'tree':
            from sklearn.preprocessing import OrdinalEncoder
            cat_pipe = SkPipeline([
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('encoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
            ])
        transformers.append(('cat', cat_pipe, cat_cols))

    return ColumnTransformer(transformers=transformers, remainder='drop')