import zipfile
import sys
from pathlib import Path

def unzip_safe(zip_path, extract_to):
    zip_path = Path(zip_path)
    extract_to = Path(extract_to)
    
    if not zip_path.exists():
        print(f"❌ Zip not found: {zip_path}")
        return

    print(f"📦 Extracting {zip_path.name} to {extract_to}...")
    try:
        extract_to.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            # List first few to confirm
            print(f"   Files in archive: {len(zip_ref.namelist())}")
            zip_ref.extractall(extract_to)
        print("✅ Extraction successful.")
    except zipfile.BadZipFile:
        print("❌ Error: Bad Zip File")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    # 1. Delamination Dataset (The one that failed PS)
    unzip_safe(
        r"data\raw\additional\Delamination_Dataset.zip", 
        r"data\raw\additional\Delamination_Generic"
    )
    
    # 2. SDNET Check (Just in case PS failed silently or is partial)
    # unzip_safe(r"data\raw\additional\SDNET2021.zip", r"data\raw\additional\SDNET2021")
