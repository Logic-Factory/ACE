import os
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import shutil
import argparse

# 定义参数解析
def parse_arguments():
    parser = argparse.ArgumentParser(description="Convert AIG files to GraphML format in parallel.")
    parser.add_argument("--source_dir", type=str, required=False, default="../../data/aig", help="Path to the directory containing AIG files.")
    parser.add_argument('--design_class', type=str, default='comb', choices=['comb','core'], help='Class of the AIG benchmark')
    parser.add_argument("--res_dir", type=str, required=False, default="../../data/graphml", help="Path to the directory where GraphML files will be stored.")
    parser.add_argument("--tool", type=str, default="", help="Path to the AIG to GraphML conversion tool.")
    return parser.parse_args()

def ensure_stats_folder_exists(dir_path, stats_folder_name='stats'):
    """Checks whether the stats subfolder exists in the specified directory"""
    return os.path.exists(os.path.join(dir_path, stats_folder_name))

def copy_stats_folder_if_exists(aig_folder_path, result_folder_path):
    """If the stats subfolder exists under the aig folder, move to the corresponding graphml folder"""
    stats_folder = os.path.join(aig_folder_path, 'stats')
    target_stats_folder = os.path.join(result_folder_path, os.path.basename(aig_folder_path),'stats')

    if os.path.exists(stats_folder):
        # os.makedirs(target_stats_folder, exist_ok=True)
        shutil.copytree(stats_folder, target_stats_folder)
        # print(f"Moved 'stats' folder from {stats_folder} to {target_stats_folder}")

def convert_aig_to_graphml(aig_file_path, result_folder_path, tool_path):
    """Convert a single AIG file to GraphML format"""
    folder_name = os.path.basename(os.path.dirname(aig_file_path))
    result_folder = os.path.join(result_folder_path, folder_name)
    os.makedirs(result_folder, exist_ok=True) 

    result_file_path = os.path.join(result_folder, os.path.basename(aig_file_path).replace(".aig", ".graphml"))
    
    command = f"{tool_path} {aig_file_path} {result_file_path}"
    subprocess.run(command, shell=True, check=True)
    
    return result_file_path

def aig_to_graphml_parallel(source_folder, result_folder, tool_path):
    """Convert all files in the AIG folder to GraphML format in parallel"""
    if not os.path.exists(result_folder):
        os.makedirs(result_folder)
    
    aig_files = []
    # Identify folders and.aig files to work with
    for folder_name in os.listdir(source_folder):
        aig_folder_path = os.path.join(source_folder, folder_name)
        if os.path.isdir(aig_folder_path) and ensure_stats_folder_exists(aig_folder_path):
            for aig_file in os.listdir(aig_folder_path):
                if aig_file.endswith(".aig"):
                    aig_files.append(os.path.join(aig_folder_path, aig_file))
            
            copy_stats_folder_if_exists(aig_folder_path, result_folder)
            
    with ThreadPoolExecutor(max_workers=16) as executor:
        futures = [executor.submit(convert_aig_to_graphml, aig_file, result_folder, tool_path) for aig_file in tqdm(aig_files, desc="Converting AIG files")]
    return len(aig_files)

if __name__ == '__main__':
    args = parse_arguments()
    source_dir = os.path.join(args.source_dir, args.design_class)
    res_dir = os.path.join(args.res_dir, args.design_class)
    convert_num_aig = aig_to_graphml_parallel(source_dir, res_dir, args.tool)