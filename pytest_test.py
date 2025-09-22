# pytest_test.py
import json
from app.app import app

def test_predict():
    client = app.test_client()
    response = client.post('/predict', json={'features':[{"amount":10,"is_foreign":0,"prev_frauds":0,"hour":12,"channel":"online"}]})
    assert response.status_code == 200
    data = json.loads(response.data)
    assert 'predictions' in data