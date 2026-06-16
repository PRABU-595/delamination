import zipfile
import os
import time

zip_path = "data/raw/NASA_Composites.zip"
extract_path = "data/raw/NASA_CFRP"

print(f"Extracting {zip_path} to {extract_path}...")

if not os.path.exists(zip_path):
    print(f"Error: {zip_path} not found!")
    exit(1)

os.makedirs(extract_path, exist_ok=True)

try:
    start_time = time.time()
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        # List files first
        files = zip_ref.namelist()
        print(f"Found {len(files)} files in archive.")
        
        # Extract all
        zip_ref.extractall(extract_path)
        
    print(f"Extraction complete in {time.time() - start_time:.2f} seconds.")
    
    # Verify
    extracted_files = os.listdir(extract_path)
    print(f"Contents of {extract_path}: {extracted_files[:10]}...") 
    
except Exception as e:
    print(f"Extraction failed: {e}")
