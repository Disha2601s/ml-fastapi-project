import requests, json, sys
if __name__ == '__main__':
    url = sys.argv[1] if len(sys.argv)>1 else 'http://127.0.0.1:8000/predict'
    payload = {
        'feat_0': 0.1, 'feat_1': 0.0, 'feat_2': 0.5, 'feat_3': -0.1, 'feat_4': 1.0,
        'education': 'Bachelors'
    }
    r = requests.post(url, json=payload)
    print('status', r.status_code)
    print('response', r.text)
