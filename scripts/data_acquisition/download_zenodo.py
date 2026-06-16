import requests
from pathlib import Path
import time

# ZENODO DOWNLOADER
# Target: High-Fidelity Delamination & Defect Datasets from Open Science
# API: https://zenodo.org/api/records/{id}

RECORDS = {
    "Bristol_XRay_CT_Voids": {
        "id": "4068305",
        "description": "X-ray CT of carbon fiber reinforced polymer with voids (Bristol)",
        "folder": "data/raw/additional/Zenodo_Bristol_CT"
    },
    "CFRP_Ultrasonic_Wavefield": {
        "id": "5105861", 
        "description": "Full ultrasonic guided wavefield measurements of CFRP plate (Debonding/Delamination)",
        "folder": "data/raw/additional/Zenodo_Ultrasonic"
    },
    "GFRP_Acoustic_Emission": {
        "id": "14591124", # Note: Recent ID, checking validity
        "description": "GFRP Composites Damage Detection using Acoustic Emission",
        "folder": "data/raw/additional/Zenodo_GFRP_AE"
    }
}

def download_zenodo_record(record_key, record_info):
    record_id = record_info['id']
    folder = Path(record_info['folder'])
    folder.mkdir(parents=True, exist_ok=True)
    
    print(f"\n🌍 Querying Zenodo Record {record_id} ({record_info['description']})...")
    api_url = f"https://zenodo.org/api/records/{record_id}"
    
    try:
        r = requests.get(api_url, timeout=10)
        if r.status_code != 200:
            print(f"❌ Failed to fetch metadata: {r.status_code}")
            return
            
        data = r.json()
        files = data.get('files', [])
        
        print(f"   Found {len(files)} files.")
        
        for f in files:
            # Zenodo API structure varies (bucket vs links)
            # Modern: f['links']['self']
            # Older: f['bucket'] + filename logic, or direct 'links'
            
            filename = f.get('key') or f.get('filename')
            download_url = f.get('links', {}).get('self')
            if not download_url:
                # Fallback for older records
                download_url = f"https://zenodo.org/record/{record_id}/files/{filename}"
                
            local_path = folder / filename
            size = f.get('size', 0)
            
            if local_path.exists():
                stat = local_path.stat()
                if stat.st_size == size:
                    print(f"⏩ {filename} exists and matches size. Skipping.")
                    continue
                else:
                    print(f"⚠️ {filename} size mismatch (Local: {stat.st_size} vs Remote: {size}). Re-downloading.")
            
            print(f"⬇️ Downloading {filename} ({size/1e9:.2f} GB)...")
            
            # Streaming download
            with requests.get(download_url, stream=True) as stream:
                stream.raise_for_status()
                with open(local_path, 'wb') as outfile:
                    dl_current = 0
                    for chunk in stream.iter_content(chunk_size=8192):
                        outfile.write(chunk)
                        dl_current += len(chunk)
                        
            print(f"✅ Success: {filename}")
            
    except Exception as e:
        print(f"❌ Error processing record {record_id}: {e}")

def main():
    print("🚀 INITIALIZING ZENODO OPEN DATA ACQUISITION")
    print("   Target: Verified Delamination & Composite Defect Datasets")
    print("-" * 60)
    
    for key, info in RECORDS.items():
        download_zenodo_record(key, info)

    print("\n✅ ZENODO ACQUISITION COMPLETE")

if __name__ == "__main__":
    main()
