import os
from pathlib import Path
import requests

def debug_download():
    # Target directory
    target_dir = Path("data/raw/additional/HF_Real/Debug_Test")
    target_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"🕵️‍♂️ DEBUG DOWNLOAD PROTOCOL STARTED")
    print(f"   Target: {target_dir.absolute()}")
    
    # URL of a single known image from the dataset (Carbon Surface)
    # Using a raw Git LFS link as a test
    test_url = "https://huggingface.co/datasets/airtlab/surface-defect-classification-in-carbon-look-components-dataset/resolve/main/data/test/defects/0.jpg"
    
    print(f"1. Attempting direct GET request to: {test_url}")
    try:
        r = requests.get(test_url, stream=True, timeout=10)
        print(f"   Status Code: {r.status_code}")
        
        if r.status_code == 200:
            dest = target_dir / "test_sample_01.jpg"
            with open(dest, 'wb') as f:
                f.write(r.content)
            print(f"✅ Success! Wrote {dest.stat().st_size} bytes to disk.")
        else:
            print(f"❌ Failed with status: {r.status_code}")
            
    except Exception as e:
        print(f"❌ Exception: {e}")
        
    print("-" * 30)
    print("2. Checking HuggingFace Hub Library...")
    try:
        from huggingface_hub import hf_hub_download
        print("   huggingface_hub detected. Attempting single file download...")
        
        # Download README.md
        path = hf_hub_download(
            repo_id="airtlab/surface-defect-classification-in-carbon-look-components-dataset",
            repo_type="dataset",
            filename="README.md",
            local_dir=str(target_dir),
            local_dir_use_symlinks=False
        )
        print(f"✅ Library Success! Downloaded README to: {path}")
        
    except Exception as e:
        print(f"❌ Library Error: {e}")

if __name__ == "__main__":
    debug_download()
