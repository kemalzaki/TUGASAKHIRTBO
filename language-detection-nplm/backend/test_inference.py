import requests
import os

BACKEND_URL = "http://127.0.0.1:5000/predict"

dataset_dir = os.path.join(os.path.dirname(__file__), '..', 'dataset')
files = ['ind.txt', 'eng.txt', 'sun.txt']

if __name__ == '__main__':
    for fname in files:
        path = os.path.join(dataset_dir, fname)
        if not os.path.exists(path):
            print(f"Missing {path}")
            continue
        with open(path, 'r', encoding='utf-8') as f:
            # take first non-empty line
            for line in f:
                text = line.strip()
                if text:
                    break
        print(f"Testing {fname}: {text[:80]}...")
        try:
            r = requests.post(BACKEND_URL, json={'text': text}, timeout=10)
            print(r.status_code, r.text)
        except Exception as e:
            print('Error calling backend:', e)
