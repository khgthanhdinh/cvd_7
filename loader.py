import pandas as pd
from sklearn.model_selection import train_test_split
from config import RANDOM_STATE, CONFIG_F, CONFIG_U

def load_and_split_data(csv_path, dataset_name):
    df = pd.read_csv(csv_path)
    schema = CONFIG_F if dataset_name == 'framingham' else CONFIG_U

    cat_cols = schema.get('ordinal', []) + schema.get('binary', []) + schema.get('nominal', [])
    for col in cat_cols:
        if col in df.columns:
            df[col] = df[col].astype(str)

    X = df.drop(columns=[schema['target']])
    y = df[schema['target']]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )
    return X_train, X_test, y_train, y_test, schema