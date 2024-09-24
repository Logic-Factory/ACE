import os
import sys
import gzip
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

def decompress_files_inplace(root_dir):
    gz_files = []
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file.endswith('.gz'):
                gz_files.append(os.path.join(root, file))

    # Display the progress with tqdm
    for file_path in tqdm(gz_files, desc="Decompressing files", unit="file"):
        with gzip.open(file_path, 'rb') as f_in, open(file_path[:-3], 'wb') as f_out:
            f_out.writelines(f_in)
        os.remove(file_path)

if __name__ == "__main__":
    root = sys.argv[1]
    decompress_files_inplace(root)