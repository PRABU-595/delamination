import requests
import sys

try:
    response = requests.get("http://127.0.0.1:8000/health")
    if response.status_code == 200:
        data = response.json()
        if data['status'] == 'healthy' and data['model_loaded']:
            print("✅ API is Healthy and Model is Loaded!")
            sys.exit(0)
        else:
            print(f"⚠️ API responded but status is: {data}")
            sys.exit(1)
    else:
        print(f"❌ API Error: {response.status_code}")
        sys.exit(1)
except Exception as e:
    print(f"❌ Connection Failed: {e}")
    sys.exit(1)
