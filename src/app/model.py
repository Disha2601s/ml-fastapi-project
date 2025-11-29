from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
import joblib
import os

ARTIFACT_DIR = 'artifacts'
os.makedirs(ARTIFACT_DIR, exist_ok=True)

def load_data(path='data/dataset.csv'):
    return pd.read_csv(path)

def train_and_save(path='data/dataset.csv', model_path=f'{ARTIFACT_DIR}/model.joblib', encoder_path=f'{ARTIFACT_DIR}/encoder.joblib'):
    df = load_data(path)
    X = df.drop(columns=['target'])
    y = df['target']

    # encode categorical column 'education'
    cat_cols = ['education']
    encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
    X_cat = encoder.fit_transform(X[cat_cols])
    X_num = X.drop(columns=cat_cols).values
    import numpy as np
    X_ready = np.hstack([X_num, X_cat])

    X_train, X_test, y_train, y_test = train_test_split(X_ready, y, test_size=0.2, random_state=42)
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)

    # save
    joblib.dump(clf, model_path)
    joblib.dump(encoder, encoder_path)
    print('Saved model to', model_path, 'and encoder to', encoder_path)

    # compute metrics on test
    preds = clf.predict(X_test)
    metrics = compute_metrics(y_test, preds)
    return metrics

def load_model(model_path=f'{ARTIFACT_DIR}/model.joblib', encoder_path=f'{ARTIFACT_DIR}/encoder.joblib'):
    clf = joblib.load(model_path)
    encoder = joblib.load(encoder_path)
    return clf, encoder

def inference(input_df, model=None, encoder=None):
    # input_df is a pandas DataFrame with columns feat_0...feat_4 and education
    if model is None or encoder is None:
        model, encoder = load_model()

    cat_cols = ['education']
    X_cat = encoder.transform(input_df[cat_cols])
    X_num = input_df.drop(columns=cat_cols).values
    import numpy as np
    X_ready = np.hstack([X_num, X_cat])
    preds = model.predict(X_ready)
    return preds

def compute_metrics(y_true, y_pred):
    p = precision_score(y_true, y_pred, zero_division=0)
    r = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    return {'precision': p, 'recall': r, 'f1': f1}

def compute_slice_metrics(path='data/dataset.csv', output_path=f'{ARTIFACT_DIR}/slice_output.txt', model=None, encoder=None):
    df = load_data(path)
    cat_col = 'education'
    results = []
    for val in sorted(df[cat_col].unique()):
        sub = df[df[cat_col]==val].copy()
        y = sub['target']
        X = sub.drop(columns=['target'])
        preds = inference(X, model=model, encoder=encoder)
        metrics = compute_metrics(y, preds)
        results.append((val, metrics))
    # write to file
    with open(output_path, 'w') as f:
        for val, m in results:
            f.write(f"Category: {val} -> precision={m['precision']:.3f}, recall={m['recall']:.3f}, f1={m['f1']:.3f}\n")
    print('Wrote slice metrics to', output_path)
    return results
