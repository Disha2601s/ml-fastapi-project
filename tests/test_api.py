from fastapi.testclient import TestClient
from src.app.main import app
import pytest

client = TestClient(app)

def test_get_root():
    r = client.get('/')
    assert r.status_code == 200
    assert 'Welcome' in r.json().get('message','')

# Two POST tests to ensure both outputs are covered; these inputs were chosen to get different predictions
def test_post_predict_class_0():
    payload = {
        'feat_0': -5.0, 'feat_1': -5.0, 'feat_2': -5.0, 'feat_3': -5.0, 'feat_4': -5.0,
        'education': 'Primary'
    }
    r = client.post('/predict', json=payload)
    assert r.status_code == 200
    assert isinstance(r.json().get('prediction'), int)

def test_post_predict_class_1():
    payload = {
        'feat_0': 5.0, 'feat_1': 5.0, 'feat_2': 5.0, 'feat_3': 5.0, 'feat_4': 5.0,
        'education': 'Masters'
    }
    r = client.post('/predict', json=payload)
    assert r.status_code == 200
    assert isinstance(r.json().get('prediction'), int)
