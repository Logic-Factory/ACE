import os
import sys
import gzip
import zstandard as zstd
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

def decompress_file_by_gzip(file_path):
    """
    Function to decompress a single gz file.
    """
    with gzip.open(file_path, 'rb') as f_in, open(file_path[:-3], 'wb') as f_out:
        f_out.writelines(f_in)
    os.remove(file_path)
    return file_path

def decompress_file_by_zstd(file_path):
    """
    Decompress a single file using Zstandard.
    """
    if not file_path.endswith('.zst'):
        return

    original_file_path = file_path[:-4]  # Remove .zst extension
    with open(file_path, 'rb') as f_in:
        with open(original_file_path, 'wb') as f_out:
            dctx = zstd.ZstdDecompressor()
            dctx.copy_stream(f_in, f_out)
    os.remove(file_path)
    return original_file_path


def decompress_files_inplace(root_dir):
    gz_files = []
    zst_files = []
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file.endswith('.gz'):
                gz_files.append(os.path.join(root, file))
            elif file.endswith('.zst'):
                zst_files.append(os.path.join(root, file))
            else:
                continue
    if len (gz_files) == 0 and len(zst_files) == 0:
        print("No .gz or .zst files found in the directory.")
        sys.exit(1)
    elif len (gz_files) > 0 and len(zst_files) > 0:
        print("Both .gz and .zst files found in the directory.")
        sys.exit(1)
    
    mode = "zst" if len(zst_files) > len(gz_files) else "gz"
    print(f"compress tool: {mode}")
    
    # Display the progress with tqdm
    total_files = len(gz_files) + len(zst_files)
    print(f"Total files to decompress: {total_files}")

    with ThreadPoolExecutor(max_workers=64) as executor:
        if mode == "gz":
            futures = {executor.submit(decompress_file_by_gzip, file_path): file_path for file_path in gz_files}
        elif mode == "zst":
            futures = {executor.submit(decompress_file_by_zstd, file_path): file_path for file_path in zst_files}
        else:
            print("Invalid mode. Please use 'gz' or 'zst'.")
            sys.exit(1)
        for future in tqdm(as_completed(futures), total=total_files, desc="Decompressing files", unit="file"):
            future.result()

if __name__ == "__main__":
    root:str = sys.argv[1]
    print("source folder: ", root)
    decompress_files_inplace(root)