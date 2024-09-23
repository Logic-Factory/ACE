import os
import sys
import gzip
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

def compress_file(file_path):
    """
    Function to compress a single file.
    """
    gzip_file_path = file_path + '.gz'
    with open(file_path, 'rb') as f_in, gzip.open(gzip_file_path, 'wb') as f_out:
        f_out.writelines(f_in)
    os.remove(file_path)
    return file_path

def compress_files_inplace(root_dir):
    file_extension = [".v", ".graphml", ".json", ".seq", ".dot"]
    
    all_files = []  # List to store all files that need to be compressed
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            extension = os.path.splitext(file)[1]
            if extension in file_extension:
                all_files.append(os.path.join(root, file))
    
    total_files = len(all_files)
    print(f"Total files to compress: {total_files}")
    # Compress each file and show progress with tqdm
    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(compress_file, file_path) for file_path in all_files]
        for future in tqdm(as_completed(futures), total=total_files, desc="compressing", unit="file"):
            try:
                future.result()  # Waiting for each future to resolve here
            except Exception as exc:
                file_path = futures[future]  # Mapping futures to file paths if needed
                print(f"{file_path} generated an exception: {exc}")

if __name__ == "__main__":
    root = sys.argv[1]
    compress_files_inplace(root)