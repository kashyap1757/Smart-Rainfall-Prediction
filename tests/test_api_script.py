import urllib.request
import json

def fetch(url):
    req = urllib.request.Request(url)
    with urllib.request.urlopen(req) as response:
        return json.loads(response.read().decode())

def post(url, data):
    req = urllib.request.Request(url, data=json.dumps(data).encode(), headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(req) as response:
        return json.loads(response.read().decode())

out = {}
try: out['health'] = fetch('http://127.0.0.1:8000/health')
except Exception as e: out['health'] = str(e)

try: out['summary'] = fetch('http://127.0.0.1:8000/analytics/summary')
except Exception as e: out['summary'] = str(e)

try: out['predict'] = post('http://127.0.0.1:8000/quick-predict', {'temperature': 28.5, 'humidity': 78.0, 'pressure': 1006.5, 'wind_speed': 18.2, 'cloud_cover': 72.0, 'dew_point': 24.1})
except Exception as e: out['predict'] = str(e)

with open('api_responses.json', 'w') as f:
    json.dump(out, f, indent=2)
