from src.app import model as m
import os
import pandas as pd

def test_data_generation_and_train(tmp_path):
    # ensure data exists
    if not os.path.exists('data/dataset.csv'):
        from src.data import generate_data as g; g.__name__
    metrics = m.train_and_save()
    assert 'precision' in metrics and 'recall' in metrics and 'f1' in metrics

def test_inference_returns_array():
    clf, encoder = m.load_model()
    df = pd.DataFrame([{
        'feat_0': 0.0,'feat_1':0.0,'feat_2':0.0,'feat_3':0.0,'feat_4':0.0,
        'education':'Bachelors'
    }])
    preds = m.inference(df, model=clf, encoder=encoder)
    assert hasattr(preds, '__iter__')
