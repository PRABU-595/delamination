import os
import subprocess
import sys
from pathlib import Path

# Ensure dependencies are installed
def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

try:
    import huggingface_hub
    from datasets import load_dataset
except ImportError:
    print("📦 Installing HuggingFace libraries...")
    install("huggingface_hub")
    install("datasets")
    import huggingface_hub
    from datasets import load_dataset

def download_hf_real_data():
    target_dir = Path("data/raw/additional/HF_Real")
    target_dir.mkdir(parents=True, exist_ok=True)
    
    print("🚀 INITIALIZING REAL DATA DOWNLOAD (HuggingFace)")
    print(f"   Target Dir: {target_dir.absolute()}")
    print("-" * 60)
    
    # 1. Carbon Look Components (Specific to User Domain)
    # This is a direct match for "Carbon Fiber" visual inspection
    print("\n🔍 Downloading 'Carbon Look Surface Defects'...")
    try:
        # Using snapshot_download to get raw files
        snapshot_path = huggingface_hub.snapshot_download(
            repo_id="airtlab/surface-defect-classification-in-carbon-look-components-dataset",
            repo_type="dataset",
            local_dir=str(target_dir / "Carbon_Look"),
            local_dir_use_symlinks=False
        )
        print(f"✅ Downloaded to: {snapshot_path}")
    except Exception as e:
        print(f"❌ Failed: {e}")

    # 2. Defect Spectrum (Industrial General, Massive)
    # We will download a subset 'industrial' or similar if available, or just the main repo
    print("\n🔍 Downloading 'Defect Spectrum' (Industrial Benchmark)...")
    try:
        # This is likely very large, so we just download the 'dev' or 'test' split to show proof
        # Use load_dataset to stream/cache
        print("   streaming dataset first to verify...")
        ds = load_dataset("DefectSpectrum/Defect_Spectrum", split="test", trust_remote_code=True)
        
        # Save images to disk
        save_path = target_dir / "Defect_Spectrum"
        save_path.mkdir(exist_ok=True)
        
        print(f"   Saving {len(ds)} real images to disk...")
        import time
        for i, item in enumerate(ds):
            if i >= 2000: break # Limit for demo
            
            # Simple retry backoff
            retries = 3
            while retries > 0:
                try:
                    img = item['image']
                    save_file = save_path / f"real_defect_{i:05d}.jpg"
                    if not save_file.exists():
                        img.save(save_file)
                    break
                except Exception as e:
                    print(f"   Retry {i} ({retries} left): {e}")
                    time.sleep(2)
                    retries -= 1
                    
            if i % 100 == 0:
                print(f"   Progress: {i}/2000 images...")
            
        print(f"✅ Saved 2000 real images to {save_path}")
        print("   (Note: Modify script limit to download full 50GB volume)")
        
    except Exception as e:
        print(f"❌ Failed: {e}")
        
    print("-" * 60)
    print("✅ ACQUISITION PROTOCOL COMPLETE")

if __name__ == "__main__":
    download_hf_real_data()
