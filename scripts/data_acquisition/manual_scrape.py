import requests
import time
from pathlib import Path
import json

# Target Real Dataset: Carbon Look Surface Defects
# Strategy: Use the undocumented HF Tree API to list files, then download raw LFS content.

BASE_URL = "https://huggingface.co/datasets/airtlab/surface-defect-classification-in-carbon-look-components-dataset/resolve/main"
API_TREE_URL = "https://huggingface.co/api/datasets/airtlab/surface-defect-classification-in-carbon-look-components-dataset/tree/main/data"

TARGET_DIR = Path("data/raw/additional/HF_Real/Carbon_Look")
TARGET_DIR.mkdir(parents=True, exist_ok=True)

def list_remote_files(path_url):
    print(f"🔎 Scanning remote directory: {path_url}")
    try:
        r = requests.get(path_url)
        if r.status_code != 200:
            print(f"❌ Failed to list files status {r.status_code}")
            return []
        items = r.json()
        return items
    except Exception as e:
        print(f"❌ Exception scanning: {e}")
        return []

def recursive_download(remote_path, local_base):
    # Get file list for this directory layer
    tree_url = f"https://huggingface.co/api/datasets/airtlab/surface-defect-classification-in-carbon-look-components-dataset/tree/main/{remote_path}"
    items = list_remote_files(tree_url)
    
    for item in items:
        item_type = item['type']
        item_path = item['path'] 
        
        if item_type == 'directory':
            # Recurse
            print(f"📂 Entering: {item_path}")
            recursive_download(item_path, local_base)
            
        elif item_type == 'file':
            # Check extension (images only)
            if not item_path.lower().endswith(('.jpg', '.png', '.tif')):
                continue
                
            # Construct download URL (Git LFS raw link)
            # URL structure: BASE_URL + relative path from root
            # Note: item['path'] is full path like 'data/test/defects/00.jpg'
            # BASE_URL is .../resolve/main/
            file_url = f"https://huggingface.co/datasets/airtlab/surface-defect-classification-in-carbon-look-components-dataset/resolve/main/{item_path}"
            
            # Local path
            local_file = local_base / item_path.replace("data/", "") # Flatten slightly
            local_file.parent.mkdir(parents=True, exist_ok=True)
            
            if local_file.exists():
                print(f"⏩ Skipping {local_file.name} (Exists)")
                continue
            
            # Download with Retry
            retries = 3
            while retries > 0:
                try:
                    print(f"⬇️ Downloading {local_file.name}...", end='\r')
                    r = requests.get(file_url, stream=True, timeout=10)
                    if r.status_code == 200:
                        with open(local_file, 'wb') as f:
                            for chunk in r.iter_content(chunk_size=8192):
                                f.write(chunk)
                        print(f"✅ Downloaded {local_file.name}        ")
                        break
                    else:
                        print(f"⚠️ HTTP {r.status_code} - Retrying...")
                except Exception as e:
                    print(f"⚠️ Error {e} - Retrying...")
                
                time.sleep(1)
                retries -= 1

def main():
    print("🚀 STARTING MANUAL SCRAPER (Bypassing Auth)")
    print(f"   Target: {TARGET_DIR}")
    
    # Start scan at 'data' folder
    # We know the structure is data/test/defects, data/train/defects etc.
    recursive_download("data", TARGET_DIR)
    
    print("\n✅ SCRAPE COMPLETE")

if __name__ == "__main__":
    main()
