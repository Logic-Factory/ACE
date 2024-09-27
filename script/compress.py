import os
import sys
import gzip
import zstandard as zstd

from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

def get_folder_size(folder):
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(folder):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            # skip if it is symbolic link
            if not os.path.islink(fp):
                total_size += os.path.getsize(fp)
    return total_size

def sort_designs_by_size(root_dir):
    designs = os.listdir(root_dir)
    design_sizes = {}
    for design in designs:
        design_path = os.path.join(root_dir, design)
        if os.path.isdir(design_path):  # Ensure it's a directory
            size = get_folder_size(design_path)
            design_sizes[design] = size
    sorted_designs = sorted(design_sizes.items(), key=lambda item: item[1], reverse=False)
    return sorted_designs

def compress_file_by_gzip(file_path):
    """
    Function to compress a single file.
    """
    gzip_file_path = file_path + '.gz'
        
    with open(file_path, 'rb') as f_in, gzip.open(gzip_file_path, 'wb') as f_out:
        f_out.writelines(f_in)
    os.remove(file_path)
    return file_path

def compress_file_by_zstd(file_path):
    """
    Function to compress a single file.
    """
    zst_file_path = file_path + '.zst'
    with open(file_path, 'rb') as f_in:
        with open(zst_file_path, 'wb') as f_out:
            cctx = zstd.ZstdCompressor()
            cctx.copy_stream(f_in, f_out)
    os.remove(file_path)
    return file_path

def compress_files_inplace(root_dir:str, mode:str):
    file_extension = [".v", ".graphml", ".json", ".seq", ".dot"]
    
    designs = sort_designs_by_size(root_dir)
    
    print(designs)
    
    for design, room in designs:
        all_files = []
        for root, dirs, files in os.walk(os.path.join(root_dir, design)):
            for file in files:
                extension = os.path.splitext(file)[1]
                if extension in file_extension:
                    all_files.append(os.path.join(root, file))
        total_files = len(all_files)
        if total_files == 0:
            print("No files to compress in design: ", design)
            continue
        print("Compressing {} files in design: {}".format(total_files, design))
        # Compress each file and show progress with tqdm
        with ThreadPoolExecutor(max_workers=64) as executor:
            if mode == "gz":
                futures = [executor.submit(compress_file_by_gzip, file_path) for file_path in all_files]
            elif mode == "zst":
                futures = [executor.submit(compress_file_by_zstd, file_path) for file_path in all_files]
            else:
                print("Invalid mode. Please use 'gz' or 'zst'.")
                sys.exit(1)
            for future in tqdm(as_completed(futures), total=total_files, desc="compressing at " + design, unit="file"):
                future.result()

if __name__ == "__main__":
    root:str = sys.argv[1]
    mode:str = sys.argv[2]
    if mode not in ["gz", "zst"]:
        print("mode must be gz or zst")
        sys.exit(1)
    print("source folder: ", root)
    print("compress tool: ", mode)
    compress_files_inplace(root, mode)