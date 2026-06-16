import requests
import time
from pathlib import Path

# STEALTH DOWNLOADER
# Mimics a real browser to bypass Mendeley's "Auth/Cookie" walls.

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7',
    'Accept-Language': 'en-US,en;q=0.9',
    'Referer': 'https://data.mendeley.com/',
    'Connection': 'keep-alive',
    'Upgrade-Insecure-Requests': '1',
    'Sec-Fetch-Dest': 'document',
    'Sec-Fetch-Mode': 'navigate',
    'Sec-Fetch-Site': 'same-origin',
    'Sec-Fetch-User': '?1'
}

DATASETS = {
    "TUDelft_Stealth": {
        # Try the ZIP API again with headers
        "url": "https://data.mendeley.com/api/datasets-v1/datasets/mpjgzd3k9k/zip?version=1",
        "folder": "data/raw/additional/TUDelft_AE"
    },
     "TUDelft_Direct_Fallback": {
        # Try the original direct file link with headers
        "url": "https://data.mendeley.com/public-files/datasets/mpjgzd3k9k/files/f3e8f8a8-0f0e-4b2e-9d2a-8c9b0e8c0b5d/file_downloaded",
        "folder": "data/raw/additional/TUDelft_AE/Direct"
    }
}

def stealth_download(url, folder, name_suffix=""):
    folder_path = Path(folder)
    folder_path.mkdir(parents=True, exist_ok=True)
    
    filename = "download_artifact" + name_suffix + ".zip"
    filepath = folder_path / filename
    
    print(f"🕵️‍♂️ Stealth Get: {url}")
    try:
        # Session to handle cookies
        s = requests.Session()
        s.headers.update(HEADERS)
        
        # 1. Hit the landing page first to get cookies?
        # landing = "https://data.mendeley.com/datasets/mpjgzd3k9k/1"
        # s.get(landing)
        
        # 2. Hit download
        r = s.get(url, stream=True, allow_redirects=True, timeout=30)
        
        # Check if we got final destination
        print(f"   Status: {r.status_code}")
        print(f"   Content-Type: {r.headers.get('Content-Type')}")
        print(f"   Final URL: {r.url}")
        
        if r.status_code == 200:
            if 'text/html' in r.headers.get('Content-Type', ''):
                print("❌ FAILED: Got HTML instead of Binary")
                return
            
            total = int(r.headers.get('content-length', 0))
            print(f"   Size: {total/1e6:.2f} MB")
            
            with open(filepath, 'wb') as f:
                for chunk in r.iter_content(chunk_size=1024*1024):
                    f.write(chunk)
            print(f"✅ Saved to {filepath}")
        else:
            print(f"❌ Error {r.status_code}")
            
    except Exception as e:
        print(f"❌ Exception: {e}")

if __name__ == "__main__":
    for k, v in DATASETS.items():
        stealth_download(v['url'], v['folder'], name_suffix=f"_{k}")
