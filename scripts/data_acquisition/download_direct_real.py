import requests
import time
from pathlib import Path

# TARGET: Real Experimental Data (Mendeley/Zenodo Direct Links)
# Goal: +50GB of Acoustic Emission (AE) & X-ray CT

DIRECT_LINKS = {
    "TUDelft_AE_Fatigue": {
        # API V1 Link (Stable Zip Download)
        "url": "https://data.mendeley.com/api/datasets-v1/datasets/mpjgzd3k9k/zip?version=1",
        "filename": "TUDelft_AE.zip",
        "folder": "data/raw/additional/TUDelft_AE" 
    },
    "Composite_Impact_Images": {
        # API V1 Link
        "url": "https://data.mendeley.com/api/datasets-v1/datasets/7f65yym3r5/zip?version=1",
        "filename": "Impact_Damage_Images.zip",
        "folder": "data/raw/additional/Mendeley_Impact"
    },
    "Compression_Fatigue_AE": {
        # API V1 Link (Version 2 as per search result date)
        "url": "https://data.mendeley.com/api/datasets-v1/datasets/28r52c3c57/zip?version=2", 
        "filename": "Compression_Fatigue_AE.zip",
        "folder": "data/raw/additional/Mendeley_Fatigue"
    },
    "Bristol_CT_Void_Data": {
         # Bristol often mirrors to Zenodo. Using the API pattern for Zenodo if possible or swapping to a known stable Mendeley substitute
         # Swapping to: "X-ray CT of carbon fiber reinforced polymer with voids" on Mendeley (Doi: 10.17632/mv4n9m5c2y.2)
         # This is a safer bet than the broken Bristol link.
         "url": "https://data.mendeley.com/api/datasets-v1/datasets/mv4n9m5c2y/zip?version=2",
         "filename": "XRay_CT_Voids.zip",
         "folder": "data/raw/additional/Mendeley_CT"
    }
}

def robust_download(url, folder, filename):
    folder_path = Path(folder)
    folder_path.mkdir(parents=True, exist_ok=True)
    local_path = folder_path / filename
    
    if local_path.exists():
        print(f"⏩ {filename} already exists. Skipping.")
        return
        
    print(f"⬇️ Downloading {filename}...")
    headers = {'User-Agent': 'Mozilla/5.0'} # Polite header
    
    try:
        r = requests.get(url, stream=True, headers=headers, timeout=30)
        if r.status_code == 200:
            total_size = int(r.headers.get('content-length', 0))
            downloaded = 0
            start_time = time.time()
            
            with open(local_path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=1024*1024): # 1MB chunks
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        
                        # Progress status every 50MB
                        if downloaded % (50 * 1024 * 1024) == 0:
                            elapsed = time.time() - start_time
                            speed = downloaded / (elapsed + 1e-9) / 1024 / 1024
                            print(f"   {downloaded/1e9:.2f} GB / {total_size/1e9:.2f} GB  ({speed:.1f} MB/s)", end='\r')
                            
            print(f"\n✅ Download Complete: {filename}")
        else:
            print(f"\n❌ Error {r.status_code} for {filename}")
            
    except Exception as e:
        print(f"\n❌ Exception: {e}")

def main():
    print("🚀 INITIALIZING DIRECT DOWNLOAD PROTOCOL")
    print("   Source: Mendeley Data & Bristol Repository (Open Access)")
    print("   Content: Raw Acoustic Emission & X-Ray CT")
    print("-" * 60)
    
    for key, info in DIRECT_LINKS.items():
        print(f"\n📦 Processing {key}...")
        robust_download(info['url'], info['folder'], info['filename'])
        
    print("\n" + "="*60)
    print("✅ ACQUISITIONS COMPLETE")

if __name__ == "__main__":
    main()
