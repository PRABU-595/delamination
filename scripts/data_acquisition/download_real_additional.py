import requests
import zipfile
import io
from pathlib import Path
import time

# Target: 50 GB of Real Data
# Strategy: Download high-res X-Ray CT and Raw AE Waveforms

DATASETS = {
    # 1. TU Delft Acoustic Emission (Mendeley Data) ~10GB uncompressed
    "TUDelft_AE_Fatigue": {
        "url": "https://data.mendeley.com/public-files/datasets/mpjgzd3k9k/files/f3e8f8a8-0f0e-4b2e-9d2a-8c9b0e8c0b5d/file_downloaded", 
        "name": "AE_Fatigue_Raw_Waveforms.zip",
        "folder": "data/raw/additional/TUDelft_AE"
    },
    # 2. Bristol Composite X-Ray CT (Data.bris) ~20GB
    "Bristol_CT_Voids": {
        "url": "https://data.bris.ac.uk/datasets/101698/zenodo.4068305.zip", 
        "name": "Composite_CT_Scans.zip",
        "folder": "data/raw/additional/Bristol_CT"
    },
    # 3. KIT 3D µCT (KITopen) ~15GB
    "KIT_MicroCT_Structure": {
        "url": "https://publikationen.bibliothek.kit.edu/1000091429/files/MicroCT_Data.zip", # Example direct link structure
        "name": "KIT_MicroCT.zip",
        "folder": "data/raw/additional/KIT_MicroCT"
    },
     # 4. NASA PCoE (Waitlist / Mirror) - Placeholder for manual add
}

def download_file(url, folder, filename):
    folder_path = Path(folder)
    folder_path.mkdir(parents=True, exist_ok=True)
    file_path = folder_path / filename
    
    if file_path.exists():
        print(f"✅ {filename} already exists. Skipping.")
        return file_path
        
    print(f"⬇️ Downloading {filename} from {url}...")
    try:
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            total_size = int(r.headers.get('content-length', 0))
            downloaded = 0
            
            with open(file_path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0 and downloaded % (100 * 1024 * 1024) == 0: # Log every 100MB
                        print(f"   Progress: {downloaded/1e9:.2f} GB / {total_size/1e9:.2f} GB")
                        
        print(f"✅ Download complete: {filename}")
        return file_path
    except Exception as e:
        print(f"❌ Failed to download {filename}: {e}")
        return None

def extract_archive(file_path, folder):
    if not file_path: return
    print(f"📦 Extracting {file_path.name}...")
    try:
        with zipfile.ZipFile(file_path, 'r') as z:
            z.extractall(folder)
        print(f"✅ Extracted to {folder}")
    except Exception as e:
        print(f"❌ Extraction failed: {e}")

def main():
    print("🚀 INITIALIZING REAL DATA ACQUISITION PROTOCOL (Training v1.1)")
    print("   Target: High-Fidelity Experimental Data (AE & X-Ray CT)")
    print("   Goal: +50 GB Real Storage")
    print("-" * 60)
    
    for key, info in DATASETS.items():
        print(f"\n🔍 Processing {key}...")
        archive = download_file(info['url'], info['folder'], info['name'])
        if archive:
            extract_archive(archive, info['folder'])
            
    print("\n" + "="*60)
    print("✅ ACQUISITION COMPLETE. Please verify folder sizes.")
    
if __name__ == "__main__":
    main()
