import requests
import zipfile
import io
import os
from pathlib import Path

# The "Golden Spoon" Dataset: NASA PCoE CFRP Composites
# Source: NASA Prognostics Data Repository
DATASET_URL = "https://www.nasa.gov/wp-content/uploads/static/pcoe/CFRP_Composites_Data_Set.zip"
# Fallback URL if the main one moved
FALLBACK_URL = "https://ti.arc.nasa.gov/m/project/prognostic-repository/CFRP_Composites.zip"

DEST_DIR = Path("data/raw/NASA_CFRP")

def download_and_extract(url, dest):
    print(f"🚀 Attempting to download 'Golden Standard' Dataset from:")
    print(f"   {url}")
    print("   (This is ~20MB of high-fidelity Run-to-Failure data)")
    
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        print("✅ Connection Established. Downloading...")
        
        # Extract directly from memory
        total_size = int(response.headers.get('content-length', 0))
        block_size = 1024
        
        with zipfile.ZipFile(io.BytesIO(response.content)) as z:
            print(f"📦 Extracting {len(z.namelist())} files to {dest}...")
            z.extractall(dest)
            
        print("🏆 SUCCESS: The 'Golden Spoon' dataset is ready.")
        print(f"   Location: {dest.absolute()}")
        return True
        
    except Exception as e:
        print(f"❌ Download failed: {e}")
        return False

if __name__ == "__main__":
    if not DEST_DIR.exists():
        DEST_DIR.mkdir(parents=True, exist_ok=True)
        
    success = download_and_extract(DATASET_URL, DEST_DIR)
    
    if not success:
        print("⚠️ Trying fallback URL...")
        success = download_and_extract(FALLBACK_URL, DEST_DIR)
        
    if not success:
        print("\n🛑 AUTOMATED DOWNLOAD FAILED (NASA Servers sometimes block bots).")
        print("   Please manually download the 'Golden Spoon' from:")
        print(f"   👉 {DATASET_URL}")
        print(f"   And unzip it into: {DEST_DIR.absolute()}")
