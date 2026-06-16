import requests
import time
from pathlib import Path

# WORKAROUND SCRIPT: "Blind Iterator"
# Since API listing is 401 Blocked, we iterate through known sequential IDs.

BASE_URL = "https://huggingface.co/datasets/airtlab/surface-defect-classification-in-carbon-look-components-dataset/resolve/main/data/test/defects"
TARGET_DIR = Path("data/raw/additional/HF_Real/Carbon_Look/test_defects")

def blind_download_loop(start_id=0, count=500):
    TARGET_DIR.mkdir(parents=True, exist_ok=True)
    print(f"🚀 STARTING BLIND DOWNLOAD (Iterating {start_id} to {start_id+count})...")
    print(f"   Target: {TARGET_DIR}")
    
    success_count = 0
    fail_streak = 0
    
    for i in range(start_id, start_id + count):
        # Stop if we hit too many 404s in a row (end of dataset?)
        if fail_streak > 20:
            print("🛑 Too many failures. Ending loop.")
            break
            
        filename = f"{i}.jpg"
        url = f"{BASE_URL}/{filename}"
        local_path = TARGET_DIR / filename
        
        if local_path.exists():
            print(f"⏩ {filename} exists.")
            success_count += 1
            continue
            
        try:
            r = requests.get(url, stream=True, timeout=5)
            if r.status_code == 200:
                with open(local_path, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)
                print(f"✅ Downloaded {filename} ({r.headers.get('content-length')} bytes)")
                success_count += 1
                fail_streak = 0
            else:
                # print(f"❌ {filename}: {r.status_code}")
                fail_streak += 1
        except Exception as e:
            print(f"⚠️ Error {filename}: {e}")
            fail_streak += 1
            
    print("-" * 40)
    print(f"🏁 FINISHED. Captured {success_count} images.")

if __name__ == "__main__":
    blind_download_loop(0, 500)
