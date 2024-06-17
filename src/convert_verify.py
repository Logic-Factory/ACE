import os
import subprocess
import argparse
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import shutil

class ConvertTool:
    def __init__(self, convert_tool, source_folder, target_folder, folder_name, temp_folder):
        self.convert_tool = convert_tool
        self.source_folder = os.path.join(source_folder, folder_name)
        self.target_folder = target_folder
        self.folder_name = folder_name
        self.temp_folder = temp_folder
        self.success_folder = os.path.join(self.target_folder,self.folder_name,'success')
        self.error_folder = os.path.join(self.target_folder,self.folder_name,'error')
        self.success_file_path = os.path.join(self.success_folder, f'success_{self.folder_name}_files.txt')
        self.error_file_path = os.path.join(self.error_folder, f'error_{self.folder_name}_files.txt')

    def run(self):
        # Ensure folder exists
        if not os.path.exists(self.target_folder):
            os.makedirs(self.target_folder, exist_ok=True)
        if not os.path.exists(self.success_folder):
            os.makedirs(self.success_folder, exist_ok=True)
        if not os.path.exists(self.error_folder):
            os.makedirs(self.error_folder, exist_ok=True)
        if not os.path.exists(self.temp_folder):
            os.makedirs(self.temp_folder, exist_ok=True)
        
        # Create the file and close it immediately to ensure that the file exists
        with open(self.success_file_path, 'w') as success_file:
            pass
        with open(self.error_file_path, 'w') as error_file:
            pass
        
        success_file_path = os.path.join(self.success_folder, f'success_{self.folder_name}_files.txt')
        error_file_path = os.path.join(self.error_folder, f'error_{self.folder_name}_files.txt')
        files = [f for f in os.listdir(self.source_folder) if os.path.isfile(os.path.join(self.source_folder, f))]

        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(self.convert_file, os.path.join(self.source_folder, file)) for file in files]
            for future in tqdm(futures, total=len(files)):
                future.result()

        print("Conversion process completed.")

        self.copy_files()

    def convert_file(self, file_path):
        try:
            result = subprocess.run([self.convert_tool, file_path, self.temp_folder],
                                    capture_output=True, text=True)
            if "success!" in result.stdout:
                with open(self.success_file_path, 'a') as success_file:
                    success_file.write(file_path + '\n')
                # print(f"Successfully converted {file_path}")
            else:
                raise result.stderr
        except subprocess.CalledProcessError as e:
            with open(self.error_file_path, 'a') as error_file:
                error_file.write(f"{file_path}\n")
                # error_file.write(f"{file_path} failed with error: {e.stderr}\n")
            # print(f"Failed to convert {file_path} with error: {e.stderr}")
        except Exception as e:
            with open(self.error_file_path, 'a') as error_file:
                error_file.write(f"{file_path}\n")
                # error_file.write(f"{file_path} raised an unexpected error: {e}\n")
            # print(f"An unexpected error occurred while converting {file_path}: {e}")

    def copy_files(self):
        copied_files_count = 0
        with open(self.success_file_path, 'r') as file:
            for source_line in file:
                source_path = source_line.strip()
                if os.path.isfile(source_path):
                    # Constructs the full path to the object file
                    filename = os.path.basename(source_path)
                    target_path = os.path.join(self.success_folder, filename)
                    shutil.copy(source_path, target_path)
                    print(f'Copied {source_path} to {target_path}')

                    copied_files_count += 1
                else:
                    print(f'File not found: {source_path}')

        print(f'Total success files copied: {copied_files_count}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert files using a conversion tool.')
    parser.add_argument('--convert_tool', type=str, requirt = True, default = '', help='Path to the conversion tool.')
    parser.add_argument('--source_folder', type=str, requirt = True, default = '/data/gtech/', help='Source folder containing files to convert.')
    parser.add_argument('--target_folder', type=str, requirt = True, default = '/data/gtech/convert_results', help='Target folder to save converted files.')
    parser.add_argument('--folder_name', type=str, requirt = True, default = 'kdd24, openabcd ...', help='Folder name for the conversion process.')
    parser.add_argument('--temp_folder', type=str, default = './temp', help='Temporary folder for conversion process.')
    args = parser.parse_args()

    converter = ConvertTool(args.convert_tool, args.source_folder, args.target_folder, args.folder_name, args.temp_folder)
    converter.run()