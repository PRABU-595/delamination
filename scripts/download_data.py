import requests
import os
import sys

url = "https://phm-datasets.s3.amazonaws.com/NASA/2.+Composites.zip"
dest = "data/raw/NASA_Composites.zip"

print(f"Downloading {url} to {dest}...")

try:
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        total_length = r.headers.get('content-length')
        
        with open(dest, 'wb') as f:
            if total_length is None: # no content length header
                f.write(r.content)
            else:
                dl = 0
                total_length = int(total_length)
                for data in r.iter_content(chunk_size=4096):
                    dl += len(data)
                    f.write(data)
                    done = int(50 * dl / total_length)
                    sys.stdout.write(f"\r[{'=' * done}{' ' * (50-done)}] {dl/1024/1024:.2f} MB")
                    sys.stdout.flush()
    print("\nDownload complete.")
except Exception as e:
    print(f"\nDownload failed: {e}")
    if os.path.exists(dest):
        os.remove(dest) # Cleanup partial file
