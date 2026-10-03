import os
import urllib.request
import gzip
import shutil

def download_and_extract(url, target_path):
    print(f"Downloading {url}...")
    gz_path = target_path + '.gz'
    urllib.request.urlretrieve(url, gz_path)
    
    print(f"Extracting to {target_path}...")
    with gzip.open(gz_path, 'rb') as f_in:
        with open(target_path, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
            
    os.remove(gz_path)
    print(f"Finished {target_path}")

def main():
    networks_dir = r"e:\btp\networks"
    os.makedirs(networks_dir, exist_ok=True)
    
    datasets = {
        "facebook_combined.txt": "https://snap.stanford.edu/data/facebook_combined.txt.gz",
        "ca-GrQc.txt": "https://snap.stanford.edu/data/ca-GrQc.txt.gz",
        "ca-HepTh.txt": "https://snap.stanford.edu/data/ca-HepTh.txt.gz",
        "email-Eu-core.txt": "https://snap.stanford.edu/data/email-Eu-core.txt.gz"
    }
    
    for filename, url in datasets.items():
        target_path = os.path.join(networks_dir, filename)
        if not os.path.exists(target_path):
            try:
                download_and_extract(url, target_path)
            except Exception as e:
                print(f"Failed to download {filename}: {e}")
        else:
            print(f"File {filename} already exists, skipping.")

if __name__ == "__main__":
    main()
